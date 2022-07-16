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

#include <memory>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"

namespace tensorflow_federated {
namespace {

using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::TensorV;
using ::testing::HasSubstr;

class XLAExecutorTest : public ::testing::Test {
 public:
  XLAExecutorTest() { test_executor_ = CreateXLAExecutor("Host"); }
  std::shared_ptr<Executor> test_executor_;

  void CheckRoundTrip(v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(input_pb)));
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

TEST_F(XLAExecutorTest, CreateStructFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateStruct({embedded_tensor.ref()}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateSelectionFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateSelection(embedded_tensor.ref(), 0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateCallFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateCall(embedded_tensor.ref(), absl::nullopt),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace tensorflow_federated
