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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/type_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"

ABSL_FLAG(std::string, tff_xla_executor_test_platform, "Host",
          "The name of the XLA platform to run the tests on. By default will "
          "run on 'Host' (CPU).");

namespace tensorflow_federated {
namespace {

using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::FlatStructT;
using ::tensorflow_federated::testing::FunctionT;
using ::tensorflow_federated::testing::IdentityFunctionT;
using ::tensorflow_federated::testing::NestedStructT;
using ::tensorflow_federated::testing::NoArgFunctionT;
using ::tensorflow_federated::testing::StructT;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorT;
using ::tensorflow_federated::testing::TensorV;
using ::testing::HasSubstr;

inline absl::StatusOr<std::tuple<v0::Xla::Binding, int>> BindingFromType(
    v0::Type type, int next_unused_index) {
  switch (type.type_case()) {
    case v0::Type::kTensor: {
      v0::Xla::Binding binding;
      binding.mutable_tensor()->set_index(next_unused_index);
      return std::make_tuple(binding, next_unused_index + 1);
    }
    case v0::Type::kStruct: {
      v0::Xla::Binding binding;
      for (const auto& type_element : type.struct_().element()) {
        auto partial_binding =
            TFF_TRY(BindingFromType(type_element.value(), next_unused_index));
        next_unused_index = std::get<int>(partial_binding);
        *binding.mutable_struct_()->add_element() =
            std::get<v0::Xla::Binding>(partial_binding);
      }
      return std::make_tuple(binding, next_unused_index);
    }
    default:
      return absl::InvalidArgumentError(
          "Encountered non-tensor or struct value in attempting to construct "
          "an XLA binding.");
  }
}

inline v0::Value ComputationV(std::optional<v0::Xla::Binding> in_binding,
                              v0::Xla::Binding out_binding,
                              xla::XlaComputation xla_comp,
                              v0::Type computation_type) {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  comp_pb->mutable_xla()->mutable_hlo_module()->PackFrom(xla_comp.proto());
  *comp_pb->mutable_type() = computation_type;
  if (in_binding.has_value()) {
    *comp_pb->mutable_xla()->mutable_parameter() = in_binding.value();
  }
  *comp_pb->mutable_xla()->mutable_result() = out_binding;
  return value_pb;
}

// Creates an XLA shape via TF's TensorShapeToXLAShape with unknown rank set to
// true.
inline xla::Shape UnknownRankShapeWithDtype(tensorflow::DataType dtype) {
  tensorflow::TensorShapeProto tensor_shape;
  tensor_shape.set_unknown_rank(true);
  xla::PrimitiveType xla_type;
  tensorflow::DataTypeToPrimitiveType(dtype, &xla_type).IgnoreError();
  tensorflow::PartialTensorShape partial_shape;
  tensorflow::PartialTensorShape::BuildPartialTensorShape(tensor_shape,
                                                          &partial_shape)
      .IgnoreError();
  xla::Shape to_return =
      tensorflow::TensorShapeToXLAShape(xla_type, partial_shape);
  return to_return;
}

// Creates an XLA shape via TF's TensorShapeToXLAShape with known rank, but each
// dimension of unknown shape.
inline xla::Shape XLAShapeWithUnknownDims(tensorflow::DataType dtype,
                                          int num_dims) {
  tensorflow::TensorShapeProto tensor_shape;
  tensor_shape.set_unknown_rank(false);
  for (int i = 0; i < num_dims; i++) {
    // TensorShapeProto uses -1 to represent unknown dim, just like TFF.
    tensor_shape.add_dim()->set_size(-1);
  }
  xla::PrimitiveType xla_type;
  tensorflow::DataTypeToPrimitiveType(dtype, &xla_type).IgnoreError();
  tensorflow::PartialTensorShape partial_shape;
  tensorflow::PartialTensorShape::BuildPartialTensorShape(tensor_shape,
                                                          &partial_shape)
      .IgnoreError();
  xla::Shape to_return =
      tensorflow::TensorShapeToXLAShape(xla_type, partial_shape);
  return to_return;
}

class XLAExecutorTest : public ::testing::Test {
 public:
  XLAExecutorTest() {
    absl::StatusOr<std::vector<xla::se::Platform*>> platforms =
        xla::PlatformUtil::GetSupportedPlatforms();
    if (!platforms.ok()) {
      LOG(FATAL) << "Could not enumerate supported XLA platforms";
    }
    LOG(INFO) << "Found " << platforms->size() << " platforms";
    const std::string requested_platform_name =
        absl::GetFlag(FLAGS_tff_xla_executor_test_platform);
    for (auto* platform : *platforms) {
      LOG(INFO) << "Platform: " << platform->Name();
      if (platform->Name() == requested_platform_name) {
        if (platform->VisibleDeviceCount() > 0) {
          test_executor_ = CreateXLAExecutor(platform->Name()).value();
        }
      }
    }
    // Fail the test if we couldn't find the requested platform.
    CHECK(test_executor_ != nullptr)
        << "Could not find platform " << requested_platform_name
        << ", missing build dependency, or no devices for platform found. See "
        << "logs for registered platforms.";
  }
  std::shared_ptr<Executor> test_executor_;

  void CheckMaterializeEqual(ValueId id, v0::Value expected_result) {
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(expected_result)));
  }

  void CheckMaterializeStatusIs(
      ValueId id, ::testing::Matcher<absl::Status> status_matcher) {
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), status_matcher);
  }

  void CheckRoundTrip(v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    CheckMaterializeEqual(id, input_pb);
  }

  template <typename... Ts>
  void CheckTensorRoundTrip(Ts... tensor_constructor_args) {
    auto input_pb = TensorV(tensor_constructor_args...);
    CheckRoundTrip(input_pb);
  }

  void CheckRoundTripFails(v0::Value& input_pb,
                           ::testing::Matcher<absl::Status> status_matcher) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), status_matcher);
  }
};

TEST_F(XLAExecutorTest, RoundTripSimpleTensor) {
  int8_t input_int = 9;
  CheckTensorRoundTrip(input_int);
}

TEST_F(XLAExecutorTest, RoundTripInt64Tensor) {
  int64_t input_int = 9;
  CheckTensorRoundTrip(input_int);
}

TEST_F(XLAExecutorTest, RoundTripFloatTensor) {
  float input_float = 9;
  CheckTensorRoundTrip(input_float);
}

TEST_F(XLAExecutorTest, RoundTripNonScalarFloatTensor) {
  tensorflow::Tensor input_tensor(tensorflow::DataType::DT_FLOAT,
                                  tensorflow::TensorShape({10, 10}));
  CheckTensorRoundTrip(input_tensor);
}

TEST_F(XLAExecutorTest, RoundTripStructWithTensor) {
  v0::Value input_pb = StructV({TensorV(9)});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStructOfNestedTensors) {
  v0::Value input_pb = StructV({StructV({TensorV(24)}), TensorV(88)});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStringTensorFails) {
  // String tensors are unsupported in XLA; see
  // https://github.com/tensorflow/tensorflow/issues/19140, and the enumeration
  // of primitive dtypes at
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto
  auto string_tensor = TensorV("a_string");
  CheckRoundTripFails(
      string_tensor,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unsupported type in DataTypeToPrimitiveType: 'string'")));
}

TEST_F(XLAExecutorTest, CreateStructOneElement) {
  v0::Value input = TensorV(5);
  TFF_ASSERT_OK_AND_ASSIGN(auto value, test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_, test_executor_->CreateStruct({value}));
  CheckMaterializeEqual(struct_, StructV({input}));
}

TEST_F(XLAExecutorTest, CreateStructSeveralElements) {
  v0::Value t1 = TensorV(5);
  v0::Value t2 = TensorV(6);
  v0::Value t3 = TensorV(7);
  v0::Value struct_ = StructV({TensorV(5), TensorV(6), TensorV(7)});
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(t1));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(t2));
  TFF_ASSERT_OK_AND_ASSIGN(auto t3id, test_executor_->CreateValue(t3));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id, t3id}));
  CheckMaterializeEqual(structid, struct_);
}

TEST_F(XLAExecutorTest, CreateSelectionFromCreateValue) {
  v0::Value input = StructV({TensorV(1), TensorV(2)});
  TFF_ASSERT_OK_AND_ASSIGN(auto vid, test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateSelection(vid, 1));
  CheckMaterializeEqual(t1id, TensorV(2));
}

TEST_F(XLAExecutorTest, CreateSelectionFromCreateStruct) {
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(TensorV(1)));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(TensorV(2)));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto selectedid,
                           test_executor_->CreateSelection(structid, 1));
  CheckMaterializeEqual(selectedid, TensorV(2));
}

TEST_F(XLAExecutorTest, CreateSelectionNonStructImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(TensorV(1)));
  CheckMaterializeEqual(id, TensorV(1));
  EXPECT_THAT(
      test_executor_->CreateSelection(id, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot create selection on non-struct value.")));
}

TEST_F(XLAExecutorTest, CreateSelectionOOBImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(StructV({})));
  CheckMaterializeEqual(id, StructV({}));
  EXPECT_THAT(
      test_executor_->CreateSelection(id, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attempted to access index 0 of a 0-length struct.")));
}

TEST_F(XLAExecutorTest, CreateValueComputationTensorNonFunctionalTypeFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(v0::DataType::DT_FLOAT);

  auto tensor_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor_type, 0)));
  v0::Value computation =
      ComputationV(tensor_binding, tensor_binding, std::move(*xla_computation),
                   // Create a computation with non-functional type; the XLA
                   // executor needs the functional type to propagate shape and
                   // type information to parameters and results.
                   float_tensor_type);

  // Materialization would fail in any case; we assert here that it fails early,
  // on computation embedding.
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument,
                                    HasSubstr("non-functional type")));
}

TEST_F(XLAExecutorTest,
       CreateValueComputationTensorMismatchedTypeAndBindingFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_tensor;
  float_tensor.mutable_tensor()->set_dtype(v0::DataType::DT_FLOAT);
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_tensor;
  *function_type.mutable_function()->mutable_parameter() = float_tensor;
  // We create a binding with mismatched structure.
  v0::Type struct_type;
  *struct_type.mutable_struct_()->add_element()->mutable_value() = float_tensor;

  auto struct_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(struct_type, 0)));
  auto tensor_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor, 0)));
  v0::Value computation =
      ComputationV(struct_binding, tensor_binding, std::move(*xla_computation),
                   function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(
      embedded_fn,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Mismatch between tensor type and non-tensor binding")));
}

TEST_F(XLAExecutorTest,
       CreateValueComputationTensorParameterKnownRankUnknownDimsFails) {
  int num_dims = 3;
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0,
                 XLAShapeWithUnknownDims(tensorflow::DT_FLOAT, num_dims), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_unk_shape_tensor;
  float_unk_shape_tensor.mutable_tensor()->set_dtype(v0::DataType::DT_FLOAT);
  for (int i = 0; i < num_dims; i++) {
    float_unk_shape_tensor.mutable_tensor()->add_dims(-1);
  }
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_shape_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_shape_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_shape_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(XLAExecutorTest, CreateValueComputationTensorParameterUnknownRankFails) {
  xla::XlaBuilder builder("float_unk_rank_tensor_identity");
  xla::Parameter(&builder, 0, UnknownRankShapeWithDtype(tensorflow::DT_FLOAT),
                 "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_unk_rank_tensor;
  float_unk_rank_tensor.mutable_tensor()->set_dtype(v0::DataType::DT_FLOAT);
  float_unk_rank_tensor.mutable_tensor()->set_unknown_rank(true);
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_rank_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_rank_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_rank_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(XLAExecutorTest, CreateValueComputationLiteralReturnsResult) {
  const v0::DataType dtype = v0::DataType::DT_INT32;
  v0::ArrayShape shape_pb = testing::CreateArrayShape({});
  auto values = {1};
  v0::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(dtype, shape_pb, values));
  v0::Computation computation_pb = testing::LiteralComputation(array_pb);
  v0::Value value_pb = testing::ComputationV(computation_pb);

  const OwnedValueId& embedded_fn =
      TFF_ASSERT_OK(test_executor_->CreateValue(value_pb));

  const v0::Value& expected_pb = TensorV(1);
  CheckMaterializeEqual(embedded_fn, expected_pb);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallSingleTensor) {
  xla::XlaBuilder builder("return_two");
  xla::XlaOp constant = xla::ConstantR0<float>(&builder, 2.0);
  // To mimic the Python tracing which always returns tuples, event for single
  // element results, after passing through MLIR
  // results are always in tuples.
  xla::Tuple(&builder, {constant});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  auto tensor_type = TensorT(v0::DataType::DT_FLOAT);
  v0::Type function_type = NoArgFunctionT(tensor_type);
  v0::Value computation = ComputationV(
      std::nullopt, std::get<0>(TFF_ASSERT_OK(BindingFromType(tensor_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  v0::Value expected_result = TensorV(2.0f);

  CheckMaterializeEqual(called_fn, expected_result);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallTensorStructure) {
  xla::XlaBuilder builder("return_two_tensors");
  auto float_one = xla::ConstantR0<float>(&builder, 1.0);
  auto float_two = xla::ConstantR0<float>(&builder, 2.0);
  xla::Tuple(&builder, {float_one, float_two});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  v0::Type return_type = FlatStructT(v0::DataType::DT_FLOAT, 2);
  v0::Type function_type = NoArgFunctionT(return_type);

  v0::Value computation = ComputationV(
      std::nullopt, std::get<0>(TFF_ASSERT_OK(BindingFromType(return_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  v0::Value expected_result = StructV({TensorV(1.0f), TensorV(2.0f)});
  CheckMaterializeEqual(called_fn, expected_result);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallNestedTensorStructure) {
  xla::XlaBuilder builder("return_nested_struct");
  auto float_one = xla::ConstantR0<float>(&builder, 1.0);
  auto float_two = xla::ConstantR0<float>(&builder, 2.0);
  auto float_three = xla::ConstantR0<float>(&builder, 3.0);
  xla::Tuple(&builder, {float_one, float_two, float_three});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  // We construct a return type <tf.float32, <tf.float32, tf.float32>>
  v0::Type nested_struct_type = NestedStructT(v0::DataType::DT_FLOAT);
  v0::Type function_type = NoArgFunctionT(nested_struct_type);

  v0::Value computation = ComputationV(
      std::nullopt,
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  v0::Value expected_result =
      StructV({TensorV(1.0f), StructV({TensorV(2.0f), TensorV(3.0f)})});

  CheckMaterializeEqual(called_fn, expected_result);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeIdentityScalar) {
  xla::XlaBuilder builder("float_scalar_identity");
  xla::XlaOp parameter = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  // To mimic the Python tracing which always returns tuples, event for single
  // element results, after passing through MLIR
  // results are always in tuples.
  xla::Tuple(&builder, {parameter});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_tensor_type = TensorT(v0::DataType::DT_FLOAT);
  v0::Type function_type = IdentityFunctionT(float_tensor_type);
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor_type, 0)));
  v0::Value computation = ComputationV(
      // Identical parameter and result bindings.
      binding, binding, std::move(*xla_computation), function_type);

  v0::Value arg_value = TensorV(2.0f);
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeIdentityNestedStruct) {
  xla::XlaBuilder builder("float_nested_struct_identity");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "y");
  auto z = xla::Parameter(&builder, 2,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "z");
  xla::Tuple(&builder, {x, y, z});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  v0::Type nested_struct_type = NestedStructT(v0::DataType::DT_FLOAT);
  v0::Type function_type = IdentityFunctionT(nested_struct_type);
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  v0::Value arg_value =
      StructV({TensorV(1.0f), StructV({TensorV(2.0f), TensorV(3.0f)})});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest, CallAndMaterializeIdentityPartiallyNonScalarStruct) {
  tensorflow::TensorShape non_scalar_tf_shape =
      tensorflow::TensorShape({10, 10});
  xla::Shape non_scalar_shape =
      tensorflow::TensorShapeToXLAShape(xla::F32, non_scalar_tf_shape);
  xla::XlaBuilder builder("partially_non_scalar_struct_identity");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1, non_scalar_shape, "y");
  xla::Tuple(&builder, {x, y});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  // Create a computation type to match the above.
  v0::Type scalar = TensorT(v0::DataType::DT_FLOAT);
  v0::Type matrix = TensorT(v0::DataType::DT_FLOAT, {10, 10});
  v0::Type struct_type = StructT({scalar, matrix});
  v0::Type function_type = IdentityFunctionT(struct_type);
  auto binding = std::get<0>(TFF_ASSERT_OK(BindingFromType(struct_type, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  v0::Value arg_value = StructV(
      {TensorV(1.0f), TensorV(tensorflow::DT_FLOAT, non_scalar_tf_shape)});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest,
       CreateCallAndMaterializeDifferentParameterAndResultTypes) {
  xla::XlaBuilder builder("float_nested_struct_partial_sum");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "y");
  auto z = xla::Parameter(&builder, 2,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "z");
  xla::Tuple(&builder, {x, xla::Add(y, z)});
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type nested_struct_type = NestedStructT(v0::DataType::DT_FLOAT);
  v0::Type result_type = FlatStructT(v0::DataType::DT_FLOAT, 2);
  v0::Type function_type = FunctionT(nested_struct_type, result_type);
  auto parameter_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0)));
  auto result_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(result_type, 0)));
  v0::Value computation =
      ComputationV(parameter_binding, result_binding,
                   std::move(*xla_computation), function_type);

  v0::Value arg_value =
      StructV({TensorV(1.0f), StructV({TensorV(2.0f), TensorV(3.0f)})});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  v0::Value expected_result = StructV({TensorV(1.0f), TensorV(5.0f)});
  CheckMaterializeEqual(called_fn, expected_result);
}

}  // namespace
}  // namespace tensorflow_federated
