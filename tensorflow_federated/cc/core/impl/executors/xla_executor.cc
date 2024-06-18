/* Copyright 2022, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/xla_executor.h"

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/global_data.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/core/impl/executors/xla_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

// clang-format off
// In TF 2.17 MultiPlatformManager was renamed to PlatformManager. Remove
// this code when the OSS build gets updated to TF 2.17+.
#include "xla/stream_executor/multi_platform_manager.h"
namespace stream_executor {
using PlatformManager = MultiPlatformManager;
} // namespace stream_executor
// clang-format on

namespace tensorflow_federated {

namespace {

// Representation of a tensor embedded in the XLA service. This class is
// responsible for owning the associated resources in the XLA service, and
// carrying sufficient information to materialize the tensor it represents into
// a TFF v0::Value.
class ServiceTensor {
 public:
  ServiceTensor(std::unique_ptr<xla::GlobalData> data, xla::PrimitiveType dtype)
      : data_(std::move(data)), dtype_(dtype) {}

  xla::PrimitiveType dtype() const { return dtype_; }
  xla::GlobalData* global_data() const { return data_.get(); }

 private:
  // XLA computations can be called with GlobalData* arguments, returning
  // GlobalData unique_ptrs. GlobalData represents an allocation of data in the
  // associated XLA service, so operating GlobalData-to-GlobalData in this way
  // minimizes transfers.
  std::unique_ptr<xla::GlobalData> data_;
  const xla::PrimitiveType dtype_;
  // Since we hold a unique pointer internally, ServiceTensor is uncopyable
  // and non-copy-constructable.
  ServiceTensor(const ServiceTensor&) = delete;
  ServiceTensor& operator=(const ServiceTensor&) = delete;
};

// Represents a computation embedded in the XLA client. Responsible for carrying
// enough information to invoke the computation and ensure results can be
// materialized at the appropriate time.
class Computation {
 public:
  Computation(xla::ExecutionHandle&& compiled_computation,
              v0::Xla::Binding arg_binding, v0::Xla::Binding result_binding,
              v0::Type computation_type)
      : xla_computation_(std::move(compiled_computation)),
        arg_binding_(std::move(arg_binding)),
        result_binding_(std::move(result_binding)),
        computation_type_(std::move(computation_type)) {}

  const xla::ExecutionHandle& xla_computation() { return xla_computation_; }
  const v0::Xla::Binding& arg_binding() { return arg_binding_; }
  const v0::Xla::Binding& result_binding() { return result_binding_; }
  const v0::Type& type() { return computation_type_; }

 private:
  Computation() = delete;
  // A handle to a compiled computation embedded in the XLA service.
  // Assuming that this computation has been previously compiled in the
  // service allows us to avoid worrying about caching computations the way
  // we need to in the TensorFlow executor. If we decide to open up the
  // TFF-JAX Python API to support unknown shapes and ranks in parameter
  // tensors, this assumption will need to be relaxed for these cases.
  // One option might involve preserving the proto and recompiling on the
  // fly, adding to an internal cache of v0::Type to xla::ExecutionHandles.
  const xla::ExecutionHandle xla_computation_;
  const v0::Xla::Binding arg_binding_;
  const v0::Xla::Binding result_binding_;
  const v0::Type computation_type_;
};

// Representation for values embedded in the XLA executor. Generally, this class
// holds handles to values embedded in the XLA client, as well as structures of
// such handles.
class XLAExecutorValue {
 public:
  enum class ValueType { TENSOR, STRUCT, COMPUTATION, UNKNOWN };

  explicit XLAExecutorValue(std::unique_ptr<xla::GlobalData> global_data,
                            xla::PrimitiveType dtype)
      : value_(std::make_shared<ServiceTensor>(std::move(global_data), dtype)) {
  }
  explicit XLAExecutorValue(absl::Span<const XLAExecutorValue> value_vector)
      : value_(std::vector<XLAExecutorValue>(value_vector.begin(),
                                             value_vector.end())) {}
  explicit XLAExecutorValue(std::shared_ptr<Computation> xla_comp)
      : value_(std::move(xla_comp)) {}

  ValueType type() const {
    if (std::holds_alternative<std::shared_ptr<ServiceTensor>>(value_)) {
      return ValueType::TENSOR;
    } else if (std::holds_alternative<std::vector<XLAExecutorValue>>(value_)) {
      return ValueType::STRUCT;
    } else if (std::holds_alternative<std::shared_ptr<Computation>>(value_)) {
      return ValueType::COMPUTATION;
    } else {
      return ValueType::UNKNOWN;
    }
  }

  // Returns a pointer to the GlobalData backing an XLAExecutorValue of tensor
  // type. Requires that type() is ValueType::TENSOR. The pointer is guaranteed
  // to be valid as long as the XLAExecutorValue exists.
  std::shared_ptr<ServiceTensor> tensor() const {
    return std::get<std::shared_ptr<ServiceTensor>>(value_);
  }
  const std::vector<XLAExecutorValue>& structure() const {
    return std::get<std::vector<XLAExecutorValue>>(value_);
  }
  std::shared_ptr<Computation> computation() const {
    return std::get<std::shared_ptr<Computation>>(value_);
  }

 private:
  XLAExecutorValue() = delete;
  using ValueVariant =
      std::variant<std::shared_ptr<ServiceTensor>,
                   std::vector<XLAExecutorValue>, std::shared_ptr<Computation>>;
  ValueVariant value_;
};

// This function, and its callers below, are used to compute flat values
// corresponding to an XLA binding from a TFF type argument. The
// vector_to_populate argument is assumed to be pre-sized to be sufficiently
// large to embed all elements specified by the binding argument. The function
// is assumed to return a StatusOr<T>, where T is the type of elements of the
// vector.
template <typename F,
          typename ReturnValueType = typename std::result_of_t<F()>::value_type>
absl::Status PopulateFlatVectorLikeBinding(
    const v0::Type& type, const v0::Xla::Binding& binding, F processing_fn,
    std::vector<ReturnValueType>* vector_to_populate) {
  switch (type.type_case()) {
    case v0::Type::kTensor: {
      if (!binding.has_tensor()) {
        return absl::InvalidArgumentError(
            "Mismatch between tensor type and non-tensor binding while "
            "computing flat info like binding.");
      }
      // The binding tells us at which vector index to populate the information
      // which is the result of processing this tensor type.
      (*vector_to_populate)[binding.tensor().index()] =
          TFF_TRY(processing_fn(type.tensor()));
      return absl::OkStatus();
    }
    case v0::Type::kStruct: {
      if (!binding.has_struct_()) {
        return absl::InvalidArgumentError(
            "Mismatch between struct type and non-struct binding while "
            "computing flat info like binding.");
      }
      if (binding.struct_().element_size() != type.struct_().element_size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Mismatch between struct with ", type.struct_().element_size(),
            " elements and binding with ", binding.struct_().element_size(),
            " elements while attempting to compute flat info like binding. "
            "These size values should match."));
      }
      int idx = 0;
      for (const auto& el : type.struct_().element()) {
        if (!(el.value().has_tensor() || el.value().has_struct_())) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Cannot flatten structure with non-tensor and struct elements "
              "to a vector corresponds to an XLA binding. Encountered type: ",
              el.value().Utf8DebugString()));
        }
        TFF_TRY(PopulateFlatVectorLikeBinding(
            el.value(), binding.struct_().element().at(idx), processing_fn,
            vector_to_populate));
        idx++;
      }
      return absl::OkStatus();
    }
    default: {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot flatten structure with non-tensor elements to "
          "a vector which corresponds to an XLA binding. Encountered type: ",
          type.Utf8DebugString()));
    }
  }
}

// Returns a vector of TFF TensorTypes which correspond to the vector of tensors
// specified by the binding argument.
absl::Status FlattenTypeToTensors(const v0::Type& type,
                                  const v0::Xla::Binding& binding,
                                  std::vector<v0::TensorType>* tensor_vector) {
  auto identity =
      [](const v0::TensorType& x) -> absl::StatusOr<v0::TensorType> {
    return x;
  };
  return PopulateFlatVectorLikeBinding(type, binding, identity, tensor_vector);
}

// Computes vector of xla::Shape pointers from the type argument, in flattened
// order determined by the binding argument. Populated in flat_shapes.
absl::Status ComputeFlatShapesFromType(const v0::Type& type,
                                       const v0::Xla::Binding& binding,
                                       std::vector<xla::Shape>* flat_shapes) {
  return PopulateFlatVectorLikeBinding(type, binding, ShapeFromTensorType,
                                       flat_shapes);
}

// Computes the number of tensor elements in a given binding. We interpret an
// unset binding to contain 0 elements, for uniformity of handling unset
// parameter bindings.
int ComputeNumElementsFromBinding(const v0::Xla::Binding& binding) {
  switch (binding.binding_case()) {
    case v0::Xla::Binding::kTensor: {
      return 1;
    }
    case v0::Xla::Binding::kStruct: {
      int num_elements = 0;
      for (const v0::Xla::Binding& el_binding : binding.struct_().element()) {
        num_elements += ComputeNumElementsFromBinding(el_binding);
      }
      return num_elements;
    }
    case v0::Xla::Binding::BINDING_NOT_SET: {
      return 0;
    }
  }
}

// Flattens an XLAExecutorValue into a vector of GlobalData pointers as
// specified by binding. This function is conceptually the inverse of the one
// below. The flat_vector argument is assumed to be presized, so that the
// indices present in the binding argument can be assigned directly to their
// appropriate locations.
absl::Status FlattenValuesIntoBinding(
    const v0::Xla::Binding& binding, const XLAExecutorValue& value,
    std::vector<xla::GlobalData*>& flat_vector) {
  switch (binding.binding_case()) {
    case v0::Xla::Binding::kTensor: {
      int32_t tensor_index_in_vector = binding.tensor().index();
      if (value.type() != XLAExecutorValue::ValueType::TENSOR) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Error encountered in FlattenValuesIntoBinding; encountered tensor "
            "binding with non-tensor XLAExecutorValue. XLAExecutorValueType: ",
            value.type()));
      }
      xla::GlobalData* tensor_data = value.tensor()->global_data();
      // The index of the binding indicates the position of this tensor in the
      // flat sequence which will be e.g. passed to XLA client's Execute method
      // as its argument.
      flat_vector[tensor_index_in_vector] = tensor_data;
      return absl::OkStatus();
    }
    case v0::Xla::Binding::kStruct: {
      if (value.type() != XLAExecutorValue::ValueType::STRUCT) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Error encountered in FlattenValuesIntoBinding; encountered struct "
            "binding with non-struct XLAExecutorValue. XLAExecutorValue type: ",
            value.type()));
      }
      const std::vector<XLAExecutorValue>& values = value.structure();
      int32_t binding_size = binding.struct_().element().size();
      if (values.size() != binding_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Error encountered in FlattenValuesIntoBinding; encountered struct "
            "binding with ",
            binding_size, "elements, and XLAExecutorValue with ", values.size(),
            " elements. These two must match."));
      }
      for (int32_t index = 0; index < values.size(); index++) {
        TFF_TRY(FlattenValuesIntoBinding(binding.struct_().element()[index],
                                         values[index], flat_vector));
      }
      return absl::OkStatus();
    }
    default:
      return absl::InternalError(
          "Encountered unset binding while flattening values into binding.");
  }
}

// Packages a vector of XLAExecutorValues of XLAExecutorValue::ValueType::TENSOR
// type as a single XLAExecutorValue whose structure and tensors match the
// binding argument. This function is conceptually the inverse of the above.
absl::StatusOr<XLAExecutorValue> PackageFlatValuesAsBinding(
    const std::vector<XLAExecutorValue>& flat_tensor_values,
    const v0::Xla::Binding& binding) {
  switch (binding.binding_case()) {
    case v0::Xla::Binding::kTensor: {
      // Simply return the (tensor) XLAExecutorValue at the index indicated by
      // the binding.
      return flat_tensor_values[binding.tensor().index()];
    }
    case v0::Xla::Binding::kStruct: {
      std::vector<XLAExecutorValue> struct_element_values;
      struct_element_values.reserve(binding.struct_().element_size());
      for (const v0::Xla::Binding& el_binding : binding.struct_().element()) {
        struct_element_values.emplace_back(TFF_TRY(
            PackageFlatValuesAsBinding(flat_tensor_values, el_binding)));
      }
      return XLAExecutorValue(struct_element_values);
    }
    default:
      return absl::InvalidArgumentError(
          "Encountered unset binding while packaging flat values into "
          "binding.");
  }
}

using ValueFuture = std::shared_future<absl::StatusOr<XLAExecutorValue>>;

class XLAExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit XLAExecutor(xla::Client* xla_client) : xla_client_(xla_client) {}

  std::string_view ExecutorName() final { return "XLAExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    return ThreadRun([value_pb, this_shared = shared_from_this()]() {
      // shared_from_this() returns the base Executor* type, so we must
      // cast to our derived type here.
      return static_cast<XLAExecutor*>(this_shared.get())
          ->CreateValueAny(value_pb);
    });
  }

  absl::StatusOr<ValueFuture> CreateCall(ValueFuture fn,
                                         std::optional<ValueFuture> arg) final {
    return ThreadRun([fn, arg, this_shared = shared_from_this()]()
                         -> absl::StatusOr<XLAExecutorValue> {
      // shared_from_this() returns the base Executor* type, so we must
      // cast to our derived type here.
      XLAExecutor* this_executor = static_cast<XLAExecutor*>(this_shared.get());
      XLAExecutorValue fn_value = TFF_TRY(Wait(fn));
      if (fn_value.type() != XLAExecutorValue::ValueType::COMPUTATION) {
        return absl::InvalidArgumentError(
            "Attempted to call a non-functional value inside the XLA "
            "Executor.");
      }
      std::shared_ptr<Computation> comp = fn_value.computation();
      if (arg.has_value()) {
        return this_executor->CallComputation(comp, TFF_TRY(Wait(arg.value())));
      }
      return this_executor->CallComputation(comp, std::nullopt);
    });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    return Map(std::move(members),
               [](std::vector<XLAExecutorValue>&& elements)
                   -> absl::StatusOr<XLAExecutorValue> {
                 return XLAExecutorValue(elements);
               });
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    return Map(std::vector<ValueFuture>({value}),
               [index](std::vector<XLAExecutorValue>&& values)
                   -> absl::StatusOr<XLAExecutorValue> {
                 XLAExecutorValue& value = values[0];
                 if (value.type() != XLAExecutorValue::ValueType::STRUCT) {
                   return absl::InvalidArgumentError(
                       "Cannot create selection on non-struct value.");
                 }
                 if (value.structure().size() <= index) {
                   return absl::InvalidArgumentError(absl::StrCat(
                       "Attempted to access index ", index, " of a ",
                       value.structure().size(), "-length struct."));
                 }
                 return XLAExecutorValue(value.structure()[index]);
               });
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final {
    XLAExecutorValue executor_value = TFF_TRY(Wait(value));
    // TODO: b/337049385 - Use of ParallelTasks here is known to potentially
    // segfault when under heavy load and large structures in the `value` future
    // because of thread exhaustion. See how the TensorFlowExecutor limits the
    // number of parallel threads for potential ideas.
    ParallelTasks tasks;
    TFF_TRY(MaterializeXLAValue(executor_value, value_pb, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }

 private:
  // Pointer to local XLA client. Assumed to be valid through the lifetime of
  // the executor.
  xla::Client* xla_client_;

  absl::StatusOr<XLAExecutorValue> CreateValueTensor(
      const v0::Value& value_pb) {
    tensorflow::Tensor t = TFF_TRY(DeserializeTensorValue(value_pb));
    xla::BorrowingLiteral tensor_literal;
    absl::Status to_literal_status =
        tensorflow::HostTensorToBorrowingLiteral(t, &tensor_literal);
    if (!to_literal_status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to convert v0::Value proto to XLA literal. Message: ",
          to_literal_status.message()));
    }
    absl::StatusOr<std::unique_ptr<xla::GlobalData>> data_in_server =
        xla_client_->TransferToServer(tensor_literal);
    if (!data_in_server.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to transfer XLA literal to local server. Message: ",
          to_literal_status.message()));
    }
    xla::PrimitiveType element_type;
    absl::Status status =
        tensorflow::DataTypeToPrimitiveType(t.dtype(), &element_type);
    if (!status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failure to convert tensorflow::DataType to XLA primitive: ",
          status.message()));
    }
    return XLAExecutorValue(std::move(*data_in_server), element_type);
  }

  absl::StatusOr<XLAExecutorValue> CreateValueComputation(
      const v0::Computation& comp_pb) {
    switch (comp_pb.computation_case()) {
      case v0::Computation::kXla: {
        if (!comp_pb.type().has_function()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Computation proto with non-functional type "
                           "encountered in XLA executor. Type: ",
                           comp_pb.type().Utf8DebugString()));
        }
        xla::HloModuleProto hlo_proto;
        comp_pb.xla().hlo_module().UnpackTo(&hlo_proto);
        xla::XlaComputation xla_comp(std::move(hlo_proto));
        // Compute the vector of flat arg shapes; these will be needed to
        // compile the computation.
        v0::Xla::Binding arg_binding = comp_pb.xla().parameter();
        int num_arg_elements = ComputeNumElementsFromBinding(arg_binding);
        // Preallocate this vector to num_arg_elements, so that we can
        // assign to these elements directly in the function call below.
        std::vector<xla::Shape> arg_shapes(num_arg_elements);
        if (comp_pb.type().function().has_parameter()) {
          TFF_TRY(ComputeFlatShapesFromType(
              comp_pb.type().function().parameter(), arg_binding, &arg_shapes));
        }
        // Compile the computation, resulting in caching the executable in
        // the XLA service.
        absl::StatusOr<xla::ExecutionHandle> computation_handle =
            xla_client_->Compile(xla_comp, arg_shapes);
        if (!computation_handle.ok()) {
          return absl::InternalError(
              absl::StrCat("Failed to compile XLA computation. Message: ",
                           computation_handle.status().message()));
        }
        // Finally, construct the representation of this computation in the
        // XLA executor.
        v0::Xla::Binding result_binding = comp_pb.xla().result();
        return XLAExecutorValue(std::make_shared<Computation>(
            std::move(*computation_handle), arg_binding, result_binding,
            comp_pb.type()));
      }
      case v0::Computation::kLiteral: {
        absl::StatusOr<std::unique_ptr<xla::GlobalData>> data =
            xla_client_->TransferToServer(
                TFF_TRY(LiteralFromArray(comp_pb.literal().value())));
        if (!data.ok()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Failed to transfer XLA literal to local server. Message: ",
              data.status().message()));
        }
        return XLAExecutorValue(std::move(data.value()),
                                TFF_TRY(PrimitiveTypeFromDataType(
                                    comp_pb.literal().value().dtype())));
      }
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Only XLA computations can be embedded in the XLA "
            "executor. Attempted to embed a computation oneof [",
            comp_pb.computation_case(), "] and TFF type signature [",
            comp_pb.type().Utf8DebugString(), "]"));
    }
  }

  absl::StatusOr<XLAExecutorValue> CreateValueStruct(
      const v0::Value::Struct& struct_pb) {
    std::vector<XLAExecutorValue> values;
    values.reserve(struct_pb.element_size());
    for (const auto& el : struct_pb.element()) {
      values.emplace_back(TFF_TRY(CreateValueAny(el.value())));
    }
    return XLAExecutorValue(values);
  }

  absl::StatusOr<XLAExecutorValue> CreateValueAny(const v0::Value& value_pb) {
    switch (value_pb.value_case()) {
      case v0::Value::ValueCase::kTensor:
        return CreateValueTensor(value_pb);
      case v0::Value::ValueCase::kComputation:
        return CreateValueComputation(value_pb.computation());
      case v0::Value::ValueCase::kStruct:
        return CreateValueStruct(value_pb.struct_());
      default:
        return absl::UnimplementedError(absl::StrCat(
            "XLA executor can only embed values of tensor, structure, or "
            "computation type. Encountered a value of type ",
            value_pb.value_case()));
    }
  }

  // NOTE: Just like in TF executor, `value` reference must remain valid until
  // `tasks.WaitAll` returns. Additionally, here the captured `this` pointer
  // must also remain valid.
  absl::Status MaterializeXLAValue(const XLAExecutorValue& executor_value,
                                   v0::Value* value_pb, ParallelTasks& tasks) {
    switch (executor_value.type()) {
      case XLAExecutorValue::ValueType::TENSOR: {
        // We add tensor materialization and serialization to the ParallelTasks
        // instance we are passed down, and avoid blocking here.
        return tasks.add_task([&executor_value, value_pb, this]() {
          std::shared_ptr<ServiceTensor> tensor_in_service =
              executor_value.tensor();
          absl::StatusOr<xla::Literal> result_literal =
              xla_client_->Transfer(*(tensor_in_service->global_data()));
          if (!result_literal.ok()) {
            return absl::InternalError(absl::StrCat(
                "Error transferring tensor from XLA service to host. Message: ",
                result_literal.status().message()));
          }
          tensorflow::Tensor tensor_out;
          absl::Status tensor_conversion = tensorflow::LiteralToHostTensor(
              *result_literal,
              TFF_TRY(tensorflow::EncodePrimitiveTypeAsDataType(
                  tensor_in_service->dtype())),
              &tensor_out);
          if (!tensor_conversion.ok()) {
            return absl::InternalError(absl::StrCat(
                "Error converting XLA literal to tensor. Message: ",
                tensor_conversion.message()));
          }
          TFF_TRY(SerializeTensorValue(tensor_out, value_pb));
          return absl::OkStatus();
        });
      }
      case XLAExecutorValue::ValueType::STRUCT: {
        v0::Value::Struct* mutable_struct = value_pb->mutable_struct_();
        for (const auto& el : executor_value.structure()) {
          TFF_TRY(MaterializeXLAValue(
              el, mutable_struct->add_element()->mutable_value(), tasks));
        }
        return absl::OkStatus();
      }
      default:
        return absl::UnimplementedError(absl::StrCat(
            "Can only materialize tensors and structures of tensors from XLA "
            "executor; attempted to materialize a value of type: ",
            executor_value.type()));
    }
  }

  absl::StatusOr<XLAExecutorValue> CallComputation(
      std::shared_ptr<Computation> fn, std::optional<XLAExecutorValue> arg) {
    int num_parameter_elements =
        ComputeNumElementsFromBinding(fn->arg_binding());
    std::vector<xla::GlobalData*> arg_vector(num_parameter_elements);
    if (arg.has_value()) {
      TFF_TRY(
          FlattenValuesIntoBinding(fn->arg_binding(), arg.value(), arg_vector));
    }
    absl::StatusOr<std::unique_ptr<xla::GlobalData>> result =
        xla_client_->Execute(fn->xla_computation(), arg_vector);
    if (!result.ok()) {
      return absl::InternalError(
          absl::StrCat("Error calling XLA computation. Message: ",
                       result.status().message()));
    }
    const v0::Xla::Binding& result_binding = fn->result_binding();
    switch (result_binding.binding_case()) {
      case v0::Xla::Binding::kTensor: {
        // JAX tracing always compiles results to be tuples, which would
        // result in length 1 tuples.
        absl::StatusOr<std::vector<std::unique_ptr<xla::GlobalData>>>
            maybe_global_data_vector = xla_client_->DeconstructTuple(**result);
        if (!maybe_global_data_vector.ok()) {
          return absl::InternalError(absl::StrCat(
              "Error destructuring tuple in XLA executor. Message: ",
              maybe_global_data_vector.status().message()));
        }
        if (maybe_global_data_vector->size() != 1) {
          return absl::InternalError(
              absl::StrCat("Expected a 1-tuple representing a single tensor "
                           "output, instead output was a tuple with",
                           maybe_global_data_vector->size(), " elements."));
        }
        return XLAExecutorValue(
            std::move(maybe_global_data_vector.value()[0]),
            TFF_TRY(PrimitiveTypeFromDataType(
                fn->type().function().result().tensor().dtype())));
      }
      case v0::Xla::Binding::kStruct: {
        absl::StatusOr<std::vector<std::unique_ptr<xla::GlobalData>>>
            global_data_vector = xla_client_->DeconstructTuple(**result);
        if (!global_data_vector.ok()) {
          return absl::InternalError(absl::StrCat(
              "Error destructuring tuple in XLA executor. Message: ",
              global_data_vector.status().message()));
        }
        // We begin by constructing a vector of tensor-backed XLAExecutorValues.
        // For this purpose, we must compute the datatypes of the GlobalData
        // elements (XLA will need them to materialize values from the XLA
        // client), from the combination of the return type of the function and
        // the result binding.
        std::vector<XLAExecutorValue> flat_value_vector;
        int result_elements = ComputeNumElementsFromBinding(result_binding);
        // Preallocate the flat types tensor as required to assign directly to
        // its elements.
        std::vector<v0::TensorType> flat_tensor_types(result_elements);
        TFF_TRY(FlattenTypeToTensors(fn->type().function().result(),
                                     result_binding, &flat_tensor_types));
        flat_value_vector.reserve(flat_tensor_types.size());
        for (int i = 0; i < flat_tensor_types.size(); i++) {
          flat_value_vector.emplace_back(XLAExecutorValue(
              std::move((*global_data_vector)[i]),
              TFF_TRY(
                  PrimitiveTypeFromDataType(flat_tensor_types[i].dtype()))));
        }
        // We repackage the flat result as an XLAExecutorValue of the same
        // structure as the result binding. This structure should additionally
        // match the structure of the return type, though we do not check this
        // here.
        return PackageFlatValuesAsBinding(flat_value_vector,
                                          fn->result_binding());
      }
      default:
        return absl::InvalidArgumentError(
            "Encountered unset result binding in calling computation.");
    }
  }
};

absl::StatusOr<xla::Client*> GetXLAClient(std::string_view platform_name) {
  absl::StatusOr<xla::se::Platform*> platform =
      xla::se::PlatformManager::PlatformWithName(platform_name);
  if (!platform.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to find specified platform ", platform_name,
                     " in PlatformManager. You may be missing a build "
                     "dependency to register the platform. Message: ",
                     platform.status().message()));
  }
  xla::LocalClientOptions options;
  options.set_platform(*platform);
  absl::StatusOr<xla::Client*> constructed_client =
      xla::ClientLibrary::GetOrCreateLocalClient(options);
  if (!constructed_client.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to construct XLA client. Message: ",
                     constructed_client.status().message()));
  }
  return *constructed_client;
}

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateXLAExecutor(
    std::string_view platform_name) {
  LOG(INFO) << "Creating XLAExecutor for platform: " << platform_name;
  xla::Client* client = TFF_TRY(GetXLAClient(platform_name));
  return std::make_shared<XLAExecutor>(client);
}

}  // namespace tensorflow_federated
