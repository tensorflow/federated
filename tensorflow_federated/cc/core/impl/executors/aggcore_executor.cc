/* Copyright 2024, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/aggcore_executor.h"

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

class ExecutorValue;

using Structure = std::shared_ptr<std::vector<ExecutorValue>>;
struct TypedFederatedIntrinsic {
  // The Federated Intrinsic.
  FederatedIntrinsic federated_intrinsic;

  // The type signature of the FederatedIntrinsic.
  v0::FunctionType type_signature;
};
using ValueVariant =
    std::variant<tensorflow::Tensor, Structure, TypedFederatedIntrinsic>;

// Representation for values inside the AggCore Executor.
class ExecutorValue {
 public:
  // The AggCore executor executes intrinsics, so the values that the executor
  // understands can be:
  // -- intrinsics: the functions to execute
  // -- arguments to and outputs of the intrinsics: tensors or structures
  enum class ValueType { TENSOR, STRUCT, INTRINSIC };

  inline static ExecutorValue CreateTensor(const tensorflow::Tensor t) {
    return ExecutorValue(std::move(t), ValueType::TENSOR);
  }

  inline tensorflow::Tensor tensor() const {
    return std::get<tensorflow::Tensor>(value_);
  }

  inline static ExecutorValue CreateFederatedIntrinsic(
      TypedFederatedIntrinsic typed_intrinsic) {
    return ExecutorValue(std::move(typed_intrinsic), ValueType::INTRINSIC);
  }

  inline TypedFederatedIntrinsic intrinsic() const {
    return std::get<TypedFederatedIntrinsic>(value_);
  }

  inline static ExecutorValue CreateStructure(Structure elements) {
    return ExecutorValue(std::move(elements), ValueType::STRUCT);
  }

  inline const Structure& structure() const {
    return std::get<::tensorflow_federated::Structure>(value_);
  }

  // Copy constructor.
  //
  // Copies are shallow: we only have to bump the reference count for either
  // the members list or the `tensorflow::Tensor`.
  ExecutorValue(ExecutorValue& other) : value_(other.value_) {}

  // Move constructor.
  ExecutorValue(ExecutorValue&& other) : value_(std::move(other.value_)) {}

  inline ValueType type() const { return type_; }

  ExecutorValue(ValueVariant value, ValueType type)
      : value_(std::move(value)), type_(type) {}

 private:
  ExecutorValue() = delete;

  ValueVariant value_;
  ValueType type_;
};

using ValueFuture = std::shared_future<absl::StatusOr<ExecutorValue>>;

class AggCoreExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit AggCoreExecutor() = default;

 protected:
  std::string_view ExecutorName() final {
    static constexpr std::string_view kExecutorName = "AggCoreExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) override {
    return absl::UnimplementedError("CreateExecutorValue not implemented.");
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, std::optional<ValueFuture> argument) override {
    return absl::UnimplementedError("CreateCall not implemented.");
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) override {
    return absl::UnimplementedError("CreateStruct not implemented.");
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) override {
    return absl::UnimplementedError("CreateSelection not implemented.");
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) override {
    return absl::UnimplementedError("Materialize not implemented.");
  }
};

}  // namespace

std::shared_ptr<Executor> CreateAggCoreExecutor() {
  return std::make_shared<AggCoreExecutor>();
}

}  // namespace tensorflow_federated
