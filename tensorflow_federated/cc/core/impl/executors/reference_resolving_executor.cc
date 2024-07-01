/* Copyright 2021, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

////////////////////////////////////////////////////////////////////////////////
// Class Definitions
////////////////////////////////////////////////////////////////////////////////

class ExecutorValue;
class ReferenceResolvingExecutor;
class Scope;

// An object for holding references (names to values).
using NamedValue = std::tuple<std::string, std::shared_ptr<ExecutorValue>>;

// An object for tracking a lambda that was created in a specific scope.
//
// References within the lambda will be resolved using the attached scope.
class ScopedLambda {
 public:
  explicit ScopedLambda(v0::Lambda lambda_pb, std::shared_ptr<Scope> scope)
      : lambda_pb_(std::move(lambda_pb)), scope_(std::move(scope)) {}
  ScopedLambda(ScopedLambda&& other)
      : lambda_pb_(std::move(other.lambda_pb_)),
        scope_(std::move(other.scope_)) {}

  absl::StatusOr<std::shared_ptr<ExecutorValue>> Call(
      const ReferenceResolvingExecutor& rre,
      std::optional<std::shared_ptr<ExecutorValue>> arg) const;

  v0::Value as_value_pb() const {
    v0::Value value_pb;
    *value_pb.mutable_computation()->mutable_lambda() = lambda_pb_;
    return value_pb;
  }

 private:
  v0::Lambda lambda_pb_;
  std::shared_ptr<Scope> scope_;
};

// An object for tracking computation evaluation scopes.
//
// Scopes are nested, each containing a single reference. Resolution of a
// reference start with the innermost nesting and works its way up to the
// root scope, which is identified by `parent_ == nullopt`.
class Scope {
 public:
  Scope() = default;
  Scope(NamedValue binding, std::shared_ptr<Scope> parent)
      : binding_(std::move(binding)), parent_(std::move(parent)) {}
  Scope(Scope&& other)
      : binding_(std::move(other.binding_)),
        parent_(std::move(other.parent_)) {}

  // Resolves a name by traversing a possibly nested scope.
  //
  // Returns the first executor value  whose name matches the `name` argument.
  // Otherwise, returns a `NotFound` error if the `name` is not present in any
  // ancestor scope.
  absl::StatusOr<std::shared_ptr<ExecutorValue>> Resolve(
      std::string_view name) const;

  // Returns a human readable string for debugging the current scope.
  //
  // Example of two nested scopes that have binding's with the same name, and
  // a middle scope with no binding.
  //
  //   [foo=V]->[]->[foo=V]
  //
  // Parent scopes are on the left, nested child scopes on the right.
  std::string DebugString() const;

 private:
  Scope(const Scope& scope) = delete;

  std::optional<NamedValue> binding_;
  // Pointer to enclosing scope. `nullopt` iff this is the root scope.
  // If a value is present, it must not be `nullptr`.
  std::optional<std::shared_ptr<Scope>> parent_;
};

// A value object for the ReferenceResolvingExecutor.
//
// This executor may have three types of values:
//   - Lambdas
//   - Values embedded in underlying executors
//   - Structures of the above
class ExecutorValue {
 public:
  enum ValueType { UNKNOWN, EMBEDDED, STRUCTURE, LAMBDA };
  explicit ExecutorValue(OwnedValueId&& child_value_id)
      : value_(std::move(child_value_id)) {}
  explicit ExecutorValue(std::vector<std::shared_ptr<ExecutorValue>>&& elements)
      : value_(std::move(elements)) {}
  explicit ExecutorValue(ScopedLambda&& scoped_lambda)
      : value_(std::move(scoped_lambda)) {}

  ValueType type() const {
    if (std::holds_alternative<OwnedValueId>(value_)) {
      return EMBEDDED;
    } else if (std::holds_alternative<
                   std::vector<std::shared_ptr<ExecutorValue>>>(value_)) {
      return STRUCTURE;
    } else if (std::holds_alternative<ScopedLambda>(value_)) {
      return LAMBDA;
    } else {
      return UNKNOWN;
    }
  }

  const OwnedValueId& embedded() const {
    return std::get<OwnedValueId>(value_);
  }

  const std::vector<std::shared_ptr<ExecutorValue>>& structure() const {
    return std::get<std::vector<std::shared_ptr<ExecutorValue>>>(value_);
  }

  const ScopedLambda& lambda() const { return std::get<ScopedLambda>(value_); }

  // Returns a human readable debugging string for error messages.
  std::string DebugString() const;

  // Move-only.
  ExecutorValue(ExecutorValue&& other) = default;
  ExecutorValue& operator=(ExecutorValue&& other) = default;
  ExecutorValue(const ExecutorValue&) = delete;
  ExecutorValue& operator=(const ExecutorValue&) = delete;

 private:
  ExecutorValue() = delete;

  std::variant<OwnedValueId, std::vector<std::shared_ptr<ExecutorValue>>,
               ScopedLambda>
      value_;
};

// An executor for resolving computation values.
//
// Specifically this executor handles computation values Lambdas, References,
// Blocks, Calls, Struct, and Selection.
//
// Other computations, including Intrinsics and TensorFlow, and handled by child
// executors.
class ReferenceResolvingExecutor
    : public ExecutorBase<std::shared_ptr<ExecutorValue>> {
 public:
  explicit ReferenceResolvingExecutor(std::shared_ptr<Executor> child)
      : child_executor_(std::move(child)) {}
  ~ReferenceResolvingExecutor() override {
    // We must make sure to delete all of our `OwnedValueId`s, releasing them
    // from the child executor as well, before deleting the child executor.
    ClearTracked();
  }

  // Evaluates a computation in the current scope.
  //
  // Evaluating a computation may involve resolving references, calls, blocks,
  // etc. The method delegates to other Evaluate*() methods, and the result
  // depends on the type of computation being evaluated.
  absl::StatusOr<std::shared_ptr<ExecutorValue>> Evaluate(
      const v0::Computation& computation_pb,
      const std::shared_ptr<Scope>& scope) const;

 protected:
  std::string_view ExecutorName() final {
    static constexpr std::string_view kExecutorName =
        "ReferenceResolvingExecutor";
    return kExecutorName;
  }

  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateExecutorValue(
      const v0::Value& value_pb) final;

  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateCall(
      std::shared_ptr<ExecutorValue> function,
      std::optional<std::shared_ptr<ExecutorValue>> argument) final;
  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateCallInternal(
      std::shared_ptr<ExecutorValue> function,
      std::optional<std::shared_ptr<ExecutorValue>> argument) const;

  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateStruct(
      std::vector<std::shared_ptr<ExecutorValue>> members) final;

  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateSelection(
      std::shared_ptr<ExecutorValue> value, uint32_t index) final;
  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateSelectionInternal(
      std::shared_ptr<ExecutorValue> source, uint32_t index) const;

  absl::Status Materialize(std::shared_ptr<ExecutorValue> value,
                           v0::Value* value_pb) final;

 private:
  std::shared_ptr<Executor> child_executor_;

  // Converts an `ExecutorValue` into a child executor value.
  //
  // This may return the `ValueId` of the already embedded value, or in the
  // case of a struct, first call `CreateStruct` on the elements and return
  // the `ValueId` of the new struct.
  absl::StatusOr<ValueId> Embed(const ExecutorValue& value,
                                std::optional<OwnedValueId>* slot) const;

  // Evaluates a block.
  //
  // The semantics of a block are documented on the
  // `tensorflow_federated::v0::Block` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateBlock(
      const v0::Block& block_pb, const std::shared_ptr<Scope>& scope) const;

  // Evaluates a reference.
  //
  // The semantics of a reference are documented on the
  // `tensorflow_federated::v0::Reference` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateReference(
      const v0::Reference& reference_pb,
      const std::shared_ptr<Scope>& scope) const;

  // Evaluates a lambda.
  //
  // The semantics of a reference are documented on the
  // `tensorflow_federated::v0::Lambda` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateLambda(
      const v0::Lambda& lambda_pb, const std::shared_ptr<Scope>& scope) const;

  // Evaluates a call.
  //
  // The semantics of a reference are documented on the
  // `tensorflow_federated::v0::Call` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateCall(
      const v0::Call& call_pb, const std::shared_ptr<Scope>& scope) const;

  // Evaluates a struct.
  //
  // The semantics of a struct are documented on the
  // `tensorflow_federated::v0::Struct` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateStruct(
      const v0::Struct& struct_pb, const std::shared_ptr<Scope>& scope) const;

  // Evaluates a selection.
  //
  // The semantics of a selection are documented on the
  // `tensorflow_federated::v0::Selection` message defined in
  // tensorflow_federated/proto/v0/computation.proto
  absl::StatusOr<std::shared_ptr<ExecutorValue>> EvaluateSelection(
      const v0::Selection& selection_pb,
      const std::shared_ptr<Scope>& scope) const;
};

////////////////////////////////////////////////////////////////////////////////
// Method implementations
////////////////////////////////////////////////////////////////////////////////

absl::StatusOr<std::shared_ptr<ExecutorValue>> ScopedLambda::Call(
    const ReferenceResolvingExecutor& rre,
    std::optional<std::shared_ptr<ExecutorValue>> arg) const {
  if (arg.has_value()) {
    NamedValue named_value =
        std::make_tuple(lambda_pb_.parameter_name(), std::move(arg.value()));
    auto new_scope = std::make_shared<Scope>(std::move(named_value), scope_);
    return rre.Evaluate(lambda_pb_.result(), new_scope);
  } else {
    return rre.Evaluate(lambda_pb_.result(), scope_);
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>> Scope::Resolve(
    std::string_view name) const {
  if (binding_.has_value()) {
    if (std::get<0>(*binding_) == name) {
      return std::get<1>(*binding_);
    }
  }
  if (parent_.has_value()) {
    return parent_.value()->Resolve(name);
  }
  return absl::NotFoundError(
      absl::StrCat("Could not find reference [", name, "]"));
}

std::string Scope::DebugString() const {
  std::string msg;
  if (binding_.has_value()) {
    msg = absl::StrCat("[", std::get<0>(*binding_), "=",
                       std::get<1>(*binding_)->DebugString(), "]");
  } else {
    msg = "[]";
  }
  if (parent_.has_value()) {
    return absl::StrCat(parent_.value()->DebugString(), "->", msg);
  } else {
    return msg;
  }
}

std::string ExecutorValue::DebugString() const {
  if (std::holds_alternative<OwnedValueId>(value_)) {
    return "V";
  } else if (std::holds_alternative<
                 std::vector<std::shared_ptr<ExecutorValue>>>(value_)) {
    return "<V>";
  } else {
    return "invalid";
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateExecutorValue(const v0::Value& value_pb) {
  VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
  switch (value_pb.value_case()) {
    case v0::Value::kFederated:
    case v0::Value::kSequence:
    case v0::Value::kArray: {
      return std::make_shared<ExecutorValue>(
          TFF_TRY(child_executor_->CreateValue(value_pb)));
    }
    case v0::Value::kStruct: {
      std::vector<std::shared_ptr<ExecutorValue>> elements;
      elements.reserve(value_pb.struct_().element_size());
      for (const v0::Value::Struct::Element& element_pb :
           value_pb.struct_().element()) {
        elements.emplace_back(TFF_TRY(CreateExecutorValue(element_pb.value())));
      }
      return std::make_shared<ExecutorValue>(std::move(elements));
    }
    case v0::Value::kComputation: {
      return Evaluate(value_pb.computation(), std::make_shared<Scope>());
    }
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Cannot create value of type [", value_pb.value_case(), "]"));
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateCall(
    std::shared_ptr<ExecutorValue> function,
    std::optional<std::shared_ptr<ExecutorValue>> argument) {
  return CreateCallInternal(std::move(function), std::move(argument));
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateCallInternal(
    std::shared_ptr<ExecutorValue> function,
    std::optional<std::shared_ptr<ExecutorValue>> argument) const {
  switch (function->type()) {
    case ExecutorValue::EMBEDDED: {
      std::optional<OwnedValueId> slot;
      std::optional<ValueId> embedded_arg;
      if (argument.has_value()) {
        embedded_arg = TFF_TRY(Embed(*argument.value(), &slot));
      }
      return std::make_shared<ExecutorValue>(TFF_TRY(
          child_executor_->CreateCall(function->embedded(), embedded_arg)));
    }
    case ExecutorValue::LAMBDA: {
      return function->lambda().Call(*this, std::move(argument));
    }
    case ExecutorValue::STRUCTURE: {
      return absl::InvalidArgumentError(
          "Received value type [STRUCTURE] which is not a function.");
    }
    case ExecutorValue::UNKNOWN: {
      return absl::InternalError(
          "Unknown function type passed to CreateCall [UNKNOWN]");
    }
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateStruct(
    std::vector<std::shared_ptr<ExecutorValue>> members) {
  return std::make_shared<ExecutorValue>(std::move(members));
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateSelection(
    std::shared_ptr<ExecutorValue> value, const uint32_t index) {
  return CreateSelectionInternal(std::move(value), index);
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::CreateSelectionInternal(
    std::shared_ptr<ExecutorValue> source, const uint32_t index) const {
  switch (source->type()) {
    case ExecutorValue::ValueType::EMBEDDED: {
      const OwnedValueId& child_id = source->embedded();
      return std::make_shared<ExecutorValue>(
          TFF_TRY(child_executor_->CreateSelection(child_id, index)));
    }
    case ExecutorValue::ValueType::STRUCTURE: {
      const std::vector<std::shared_ptr<ExecutorValue>>& elements =
          source->structure();
      if (index < 0 || index >= elements.size()) {
        const std::string err_msg =
            absl::StrCat("Failed to create selection for index [", index,
                         "] on structure with length [", elements.size(), "]");
        LOG(ERROR) << err_msg;
        return absl::NotFoundError(err_msg);
      }
      return elements[index];
    }
    case ExecutorValue::ValueType::LAMBDA: {
      return absl::InvalidArgumentError(
          "Cannot perform selection on Lambda value");
    }
    case ExecutorValue::ValueType::UNKNOWN: {
      return absl::InvalidArgumentError(
          "Cannot perform selection on unknown type value");
    }
  }
}

absl::Status ReferenceResolvingExecutor::Materialize(
    std::shared_ptr<ExecutorValue> value, v0::Value* value_pb) {
  // Value might still be a struct here, but the underlying child executor
  // only knows about the individual elements.  In this case we must create a
  // struct of the elements in the child executor first, and then materialize
  // that struct.
  std::optional<OwnedValueId> slot;
  ValueId child_value_id = TFF_TRY(Embed(*value, &slot));
  return child_executor_->Materialize(child_value_id, value_pb);
}

absl::StatusOr<ValueId> ReferenceResolvingExecutor::Embed(
    const ExecutorValue& value, std::optional<OwnedValueId>* slot) const {
  switch (value.type()) {
    case ExecutorValue::ValueType::EMBEDDED: {
      return value.embedded();
    }
    case ExecutorValue::ValueType::STRUCTURE: {
      // Container for fields that may themselves require construction of new
      // child values for structs.
      std::vector<OwnedValueId> field_slots;
      std::vector<ValueId> field_ids;
      for (const auto& field_value : value.structure()) {
        std::optional<OwnedValueId> field_slot;
        ValueId child_field_value_id =
            TFF_TRY(Embed(*field_value, &field_slot));
        if (field_slot.has_value()) {
          field_slots.emplace_back(std::move(field_slot.value()));
        }
        field_ids.push_back(child_field_value_id);
      }
      slot->emplace(TFF_TRY(child_executor_->CreateStruct(field_ids)));
      return slot->value().ref();
    }
    case ExecutorValue::ValueType::LAMBDA: {
      // Forward a lambda to the child executor. An example of this situation is
      // when a Lambda is an argument to an intrinsic call.
      OwnedValueId embedded_value_id =
          TFF_TRY(child_executor_->CreateValue(value.lambda().as_value_pb()));
      ValueId value_id = embedded_value_id.ref();
      slot->emplace(std::move(embedded_value_id));
      return value_id;
    }
    case ExecutorValue::ValueType::UNKNOWN: {
      return absl::InternalError("Tried to embed unknown ValueType [UNKNOWN]");
    }
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::Evaluate(
    const v0::Computation& computation_pb,
    const std::shared_ptr<Scope>& scope) const {
  switch (computation_pb.computation_case()) {
    case v0::Computation::kTensorflow:
    case v0::Computation::kIntrinsic:
    case v0::Computation::kData:
    case v0::Computation::kPlacement:
    case v0::Computation::kLiteral:
    case v0::Computation::kXla: {
      // Note: we're copying the Computation proto here, possibly a TensorFlow
      // graph which might have large constants, possibly making it expensive.
      // However, we've taken this approach because we don't always have a
      // `Value` for each `Computation` proto (see `v0::Block::local`); this
      // code is simpler and more homogenous. If profiling shows this is a
      // hotspot we can optimize.
      v0::Value child_value_pb;
      *child_value_pb.mutable_computation() = computation_pb;
      return std::make_shared<ExecutorValue>(
          TFF_TRY(child_executor_->CreateValue(child_value_pb)));
    }
    case v0::Computation::kReference: {
      return EvaluateReference(computation_pb.reference(), scope);
    }
    case v0::Computation::kBlock: {
      return EvaluateBlock(computation_pb.block(), scope);
    }
    case v0::Computation::kLambda: {
      return EvaluateLambda(computation_pb.lambda(), scope);
    }
    case v0::Computation::kCall: {
      return EvaluateCall(computation_pb.call(), scope);
    }
    case v0::Computation::kStruct: {
      return EvaluateStruct(computation_pb.struct_(), scope);
    }
    case v0::Computation::kSelection: {
      return EvaluateSelection(computation_pb.selection(), scope);
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Evaluate not implemented for computation type [",
                       computation_pb.computation_case(), "]"));
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateBlock(
    const v0::Block& block_pb, const std::shared_ptr<Scope>& scope) const {
  std::shared_ptr<Scope> current_scope = scope;
  auto local_pb_formatter = [](std::string* out,
                               const v0::Block::Local& local_pb) {
    out->append(local_pb.name());
  };
  for (int i = 0; i < block_pb.local_size(); ++i) {
    const v0::Block::Local& local_pb = block_pb.local(i);
    std::shared_ptr<ExecutorValue> value = TFF_TRY(
        Evaluate(local_pb.value(), current_scope),
        absl::StrCat(
            "while evaluating local [", local_pb.name(), "] in block locals [",
            absl::StrJoin(block_pb.local(), ",", local_pb_formatter), "]"));
    current_scope = std::make_shared<Scope>(
        std::make_tuple(local_pb.name(), std::move(value)),
        std::move(current_scope));
  }
  return Evaluate(block_pb.result(), current_scope);
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateReference(
    const v0::Reference& reference_pb,
    const std::shared_ptr<Scope>& scope) const {
  std::shared_ptr<ExecutorValue> resolved_value =
      TFF_TRY(scope->Resolve(reference_pb.name()),
              absl::StrCat("while searching scope: ", scope->DebugString()));
  if (resolved_value == nullptr) {
    return absl::InternalError(
        absl::StrCat("Resolved reference [", reference_pb.name(),
                     "] was nullptr. Scope: ", scope->DebugString()));
  }
  return resolved_value;
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateLambda(
    const v0::Lambda& lambda_pb, const std::shared_ptr<Scope>& scope) const {
  return std::make_shared<ExecutorValue>(ScopedLambda{lambda_pb, scope});
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateCall(
    const v0::Call& call_pb, const std::shared_ptr<Scope>& scope) const {
  std::shared_ptr<ExecutorValue> function =
      TFF_TRY(Evaluate(call_pb.function(), scope));
  std::optional<std::shared_ptr<ExecutorValue>> argument;
  if (call_pb.has_argument()) {
    argument = TFF_TRY(Evaluate(call_pb.argument(), scope));
  }
  return CreateCallInternal(std::move(function), std::move(argument));
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateStruct(
    const v0::Struct& struct_pb, const std::shared_ptr<Scope>& scope) const {
  std::vector<std::shared_ptr<ExecutorValue>> elements;
  elements.reserve(struct_pb.element_size());
  for (const v0::Struct::Element& element_pb : struct_pb.element()) {
    elements.emplace_back(TFF_TRY(Evaluate(element_pb.value(), scope)));
  }
  return std::make_shared<ExecutorValue>(std::move(elements));
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
ReferenceResolvingExecutor::EvaluateSelection(
    const v0::Selection& selection_pb,
    const std::shared_ptr<Scope>& scope) const {
  return CreateSelectionInternal(
      TFF_TRY(Evaluate(selection_pb.source(), scope)), selection_pb.index());
}

}  // namespace

std::shared_ptr<Executor> CreateReferenceResolvingExecutor(
    std::shared_ptr<Executor> child) {
  return std::make_shared<ReferenceResolvingExecutor>(std::move(child));
}

}  // namespace tensorflow_federated
