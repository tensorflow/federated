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

#include "tensorflow_federated/cc/core/impl/executors/streaming_remote_executor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_grpc.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

using ::testing::_;
using testing::EqualsProto;
using testing::StructV;
using testing::TensorV;
using testing::proto::IgnoringRepeatedFieldOrdering;

inline v0::Value ServerV(v0::Value unplaced_value) {
  v0::Value server_value = testing::ServerV(unplaced_value);
  absl::StatusOr<v0::Type> inferred_type_pb =
      InferTypeFromValue(unplaced_value);
  CHECK(inferred_type_pb.ok()) << inferred_type_pb.status();
  *server_value.mutable_federated()->mutable_type()->mutable_member() =
      inferred_type_pb.value();
  return server_value;
}

inline v0::Value ClientsV(absl::Span<const v0::Value> unplaced_values) {
  v0::Value clients_value = testing::ClientsV(unplaced_values);
  if (!unplaced_values.empty()) {
    absl::StatusOr<v0::Type> inferred_type_pb =
        InferTypeFromValue(unplaced_values[0]);
    CHECK(inferred_type_pb.ok()) << inferred_type_pb.status();
    *clients_value.mutable_federated()->mutable_type()->mutable_member() =
        inferred_type_pb.value();
  }
  return clients_value;
}

constexpr char kExecutorId[] = "executor_id";

template <typename ResponseType>
auto CreateResponseForId(std::string id) {
  ResponseType response;
  response.mutable_value_ref()->mutable_id()->assign(std::move(id));
  return response;
}

template <typename ResponseType>
auto ReturnOkWithResponseId(std::string response_id) {
  return ::testing::DoAll(
      ::testing::SetArgPointee<2>(
          CreateResponseForId<ResponseType>(std::move(response_id))),
      ::testing::Return(grpc::Status::OK));
}

v0::ComputeResponse ComputeResponseForValue(const v0::Value& value_proto) {
  v0::ComputeResponse response;
  *response.mutable_value() = value_proto;
  return response;
}

auto ReturnOkWithComputeResponse(const v0::Value& value_proto) {
  return ::testing::DoAll(
      ::testing::SetArgPointee<2>(ComputeResponseForValue(value_proto)),
      ::testing::Return(grpc::Status::OK));
}

v0::CreateValueRequest CreateValueRequestForValue(
    const v0::Value& value_proto) {
  v0::CreateValueRequest request;
  request.mutable_executor()->set_id(kExecutorId);
  *request.mutable_value() = value_proto;
  return request;
}

v0::CreateStructRequest CreateStructRequestForValues(
    const std::vector<std::string_view>& ref_names) {
  v0::CreateStructRequest create_struct_request;
  create_struct_request.mutable_executor()->set_id(kExecutorId);
  for (std::string_view ref_name : ref_names) {
    create_struct_request.add_element()->mutable_value_ref()->set_id(
        std::string(ref_name));
  }
  return create_struct_request;
}

v0::ComputeRequest ComputeRequestForId(std::string ref) {
  v0::ComputeRequest request;
  request.mutable_executor()->set_id(kExecutorId);
  request.mutable_value_ref()->mutable_id()->assign(std::move(ref));
  return request;
}

std::function<grpc::Status()> NotifyAndReturnOk(
    absl::Notification& done_notification) {
  return [&done_notification] {
    done_notification.Notify();
    return grpc::Status::OK;
  };
}

constexpr char kExpectedGetExecutorRequest[] = R"pb(
  cardinalities {
    placement { uri: "clients" }
    cardinality: 1
  }
)pb";

grpc::Status UnimplementedPlaceholder() {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Test");
}

::testing::Matcher<grpc::Status> IsUnimplementedPlaceholder() {
  return GrpcStatusIs(grpc::StatusCode::UNIMPLEMENTED, "Test");
}

class StreamingRemoteExecutorBase {
 protected:
  StreamingRemoteExecutorBase()
      : mock_executor_service_(mock_executor_.service()) {
    std::unique_ptr<v0::ExecutorGroup::Stub> stub_ptr(mock_executor_.NewStub());
    CardinalityMap cardinalities = {{"clients", 1}};
    test_executor_ =
        CreateStreamingRemoteExecutor(std::move(stub_ptr), cardinalities);
  }
  virtual ~StreamingRemoteExecutorBase() { test_executor_ = nullptr; }

  // Adds expectations of calls to `GetExecutor` and `DisposeExecutor` and
  // returns a notification which notifies when `DisposeExecutor` is called.
  //
  // Tests which call this method should end with a call to
  // `WaitForDisposeExecutor`.
  void ExpectGetAndDisposeExecutor(
      absl::Notification& dispose_notification_out) {
    v0::GetExecutorResponse get_response;
    *get_response.mutable_executor()->mutable_id() = kExecutorId;
    EXPECT_CALL(*mock_executor_service_,
                GetExecutor(_,
                            IgnoringRepeatedFieldOrdering(
                                EqualsProto(kExpectedGetExecutorRequest)),
                            _))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<2>(get_response),
                                   ::testing::Return(grpc::Status::OK)));

    absl::Notification done;
    v0::DisposeExecutorRequest dispose_request;
    *dispose_request.mutable_executor()->mutable_id() = kExecutorId;
    EXPECT_CALL(*mock_executor_service_,
                DisposeExecutor(_, EqualsProto(dispose_request), _))
        .WillOnce(NotifyAndReturnOk(dispose_notification_out));
  }

  // Discards `test_executor_` and waits for `notification`. This method is
  // intended to be used at the end of a testcase in combination with a call to
  // `ExpectGetAndDisposeExecutor` at the beginning of the testcase.
  void WaitForDisposeExecutor(absl::Notification& notification) {
    // Remove the `test_executor_` to allow the underlying stub to be deleted,
    // triggering the call to `DisposeExecutor`.
    test_executor_ = nullptr;
    while (!notification.WaitForNotificationWithTimeout(absl::Seconds(1))) {
      LOG(WARNING) << "Waiting for call to `DisposeExecutor`...";
    }
  }

  MockGrpcExecutorServer mock_executor_;
  MockGrpcExecutorService* mock_executor_service_;
  std::shared_ptr<Executor> test_executor_;
};

class StreamingRemoteExecutorTest : public ::testing::Test,
                                    public StreamingRemoteExecutorBase {
 protected:
  using StreamingRemoteExecutorBase::ExpectGetAndDisposeExecutor;
  using StreamingRemoteExecutorBase::WaitForDisposeExecutor;
};

TEST_F(StreamingRemoteExecutorTest, CreateValueTensor) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);

  v0::Value tensor_two = TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>("value_ref"));

    v0::DisposeRequest expected_dispose_request;
    expected_dispose_request.mutable_executor()->set_id(kExecutorId);
    expected_dispose_request.mutable_value_ref()->Add()->set_id("value_ref");
    EXPECT_CALL(*mock_executor_service_,
                Dispose(_, EqualsProto(expected_dispose_request), _))
        .WillOnce(::testing::Return(grpc::Status::OK));

    OwnedValueId value_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_, Compute(_, _, _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(value_ref, &materialized_value);
  }

  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateValueNestedStruct) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value tensor_three = TensorV(3.0f);
  v0::Value tensor_four = TensorV(4.0f);
  v0::Value inner_struct_value;
  auto inner_struct = inner_struct_value.mutable_struct_();
  *inner_struct->add_element()->mutable_value() = tensor_two;
  *inner_struct->add_element()->mutable_value() = tensor_three;
  v0::Value outer_struct_value;
  auto outer_struct = outer_struct_value.mutable_struct_();
  *outer_struct->add_element()->mutable_value() = inner_struct_value;
  *outer_struct->add_element()->mutable_value() = tensor_four;

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    {
      EXPECT_CALL(
          *mock_executor_service_,
          CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)),
                      _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "inner_struct_elem0"));
      EXPECT_CALL(
          *mock_executor_service_,
          CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_three)),
                      _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "inner_struct_elem1"));

      v0::CreateStructRequest inner_create_struct_request;
      inner_create_struct_request.mutable_executor()->set_id(kExecutorId);
      inner_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "inner_struct_elem0");
      inner_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "inner_struct_elem1");

      EXPECT_CALL(*mock_executor_service_,
                  CreateStruct(_, EqualsProto(inner_create_struct_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateStructResponse>(
              "outer_struct_elem0"));
      EXPECT_CALL(
          *mock_executor_service_,
          CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_four)),
                      _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "outer_struct_elem1"));

      v0::CreateStructRequest outer_create_struct_request;
      outer_create_struct_request.mutable_executor()->set_id(kExecutorId);
      outer_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "outer_struct_elem0");
      outer_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "outer_struct_elem1");

      EXPECT_CALL(*mock_executor_service_,
                  CreateStruct(_, EqualsProto(outer_create_struct_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateStructResponse>(
              "outer_struct_ref"));
    }

    OwnedValueId outer_struct_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(outer_struct_value));

    {
      // We need to call Materialize to force synchronization of the calls
      // above.
      v0::CreateSelectionRequest outer_create_selection_request0;
      outer_create_selection_request0.mutable_executor()->set_id(kExecutorId);
      outer_create_selection_request0.mutable_source_ref()->set_id(
          "outer_struct_ref");
      outer_create_selection_request0.set_index(0);

      EXPECT_CALL(
          *mock_executor_service_,
          CreateSelection(_, EqualsProto(outer_create_selection_request0), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "outer_struct_elem0"));

      v0::CreateSelectionRequest outer_create_selection_request1;
      outer_create_selection_request1.mutable_executor()->set_id(kExecutorId);
      outer_create_selection_request1.mutable_source_ref()->set_id(
          "outer_struct_ref");
      outer_create_selection_request1.set_index(1);

      EXPECT_CALL(
          *mock_executor_service_,
          CreateSelection(_, EqualsProto(outer_create_selection_request1), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "outer_struct_elem1"));

      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("outer_struct_elem1")), _))
          .WillOnce(ReturnOkWithComputeResponse(tensor_four));

      v0::CreateSelectionRequest inner_create_selection_request0;
      inner_create_selection_request0.mutable_executor()->set_id(kExecutorId);
      inner_create_selection_request0.mutable_source_ref()->set_id(
          "outer_struct_elem0");
      inner_create_selection_request0.set_index(0);
      EXPECT_CALL(
          *mock_executor_service_,
          CreateSelection(_, EqualsProto(inner_create_selection_request0), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "inner_struct_elem0"));
      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("inner_struct_elem0")), _))
          .WillOnce(ReturnOkWithComputeResponse(tensor_two));

      v0::CreateSelectionRequest inner_create_selection_request1;
      inner_create_selection_request1.mutable_executor()->set_id(kExecutorId);
      inner_create_selection_request1.mutable_source_ref()->set_id(
          "outer_struct_elem0");
      inner_create_selection_request1.set_index(1);
      EXPECT_CALL(
          *mock_executor_service_,
          CreateSelection(_, EqualsProto(inner_create_selection_request1), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "inner_struct_elem1"));
      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("inner_struct_elem1")), _))
          .WillOnce(ReturnOkWithComputeResponse(tensor_three));
    }
    materialize_status =
        test_executor_->Materialize(outer_struct_ref, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(outer_struct_value));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateValueWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);

  {
    v0::Value tensor_two = testing::TensorV(2.0f);
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));

    OwnedValueId value_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    v0::Value materialized_value;
    // If the CreateValue fails we dont expect a Compute call on the other side.
    // Nor do we expect a dispose, because no value has been created.
    absl::Status materialize_status =
        test_executor_->Materialize(value_ref, &materialized_value);
    EXPECT_THAT(materialize_status,
                StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  }
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, MaterializeWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>("value_ref"));
    OwnedValueId value_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_, Compute(_, _, _))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));
    materialize_status =
        test_executor_->Materialize(value_ref, &materialized_value);
    EXPECT_THAT(materialize_status,
                StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  }
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateCallFnWithArg) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value tensor_three = TensorV(3.0f);

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_three)),
                    _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("argument_ref"));

    // The literal argument itself is thrown away, but is used to identify which
    // CreateValue represents the function and which the argument.
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    OwnedValueId arg = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_three));

    v0::CreateCallRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_function_ref()->set_id("function_ref");
    expected_request.mutable_argument_ref()->set_id("argument_ref");

    EXPECT_CALL(*mock_executor_service_,
                CreateCall(_, EqualsProto(expected_request), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>("call_ref"));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, arg));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(*mock_executor_service_,
                Compute(_, EqualsProto(ComputeRequestForId("call_ref")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateCallNoArgFn) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  EXPECT_CALL(*mock_executor_service_, CreateValue(_, _, _))
      .WillOnce(
          ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));

  // The literal argument itself is unused.
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    v0::CreateCallRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_function_ref()->set_id("function_ref");

    EXPECT_CALL(*mock_executor_service_,
                CreateCall(_, EqualsProto(expected_request), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>("call_ref"));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, std::nullopt));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(*mock_executor_service_,
                Compute(_, EqualsProto(ComputeRequestForId("call_ref")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateCallFnWithStructReturnType) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value tensor_three = TensorV(3.0f);
  // Create a no-arg lambda that returns a structure of tensors.
  v0::Value fn_value;
  v0::Computation* fn_computation = fn_value.mutable_computation();
  v0::FunctionType* fn_type =
      fn_computation->mutable_type()->mutable_function();
  fn_type->mutable_parameter()->mutable_tensor()->set_dtype(
      v0::TensorType::DT_FLOAT);
  v0::StructType* result_struct_type =
      fn_type->mutable_result()->mutable_struct_();
  result_struct_type->add_element()
      ->mutable_value()
      ->mutable_tensor()
      ->set_dtype(v0::TensorType::DT_FLOAT);
  result_struct_type->add_element()
      ->mutable_value()
      ->mutable_tensor()
      ->set_dtype(v0::TensorType::DT_FLOAT);
  v0::Value arg_value = TensorV(4.0f);

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(fn_value)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));

    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(arg_value)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("argument_ref"));

    v0::CreateCallRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_function_ref()->set_id("function_ref");
    expected_request.mutable_argument_ref()->set_id("argument_ref");

    EXPECT_CALL(*mock_executor_service_,
                CreateCall(_, EqualsProto(expected_request), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>("call_ref"));

    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(fn_value));
    OwnedValueId arg = TFF_ASSERT_OK(test_executor_->CreateValue(arg_value));
    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, arg));

    // We need to call Materialize to force synchronization of the calls
    // above.

    v0::CreateSelectionRequest create_selection_request0;
    create_selection_request0.mutable_executor()->set_id(kExecutorId);
    create_selection_request0.mutable_source_ref()->set_id("call_ref");
    create_selection_request0.set_index(0);
    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(_, EqualsProto(create_selection_request0), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref0"));

    v0::CreateSelectionRequest create_selection_request1;
    create_selection_request1.mutable_executor()->set_id(kExecutorId);
    create_selection_request1.mutable_source_ref()->set_id("call_ref");
    create_selection_request1.set_index(1);
    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(_, EqualsProto(create_selection_request1), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref1"));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(_, EqualsProto(ComputeRequestForId("selection_ref0")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(_, EqualsProto(ComputeRequestForId("selection_ref1")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_three));

    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value,
              EqualsProto(StructV({tensor_two, tensor_three})));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateCallError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  EXPECT_CALL(*mock_executor_service_, CreateValue(_, _, _))
      .WillOnce(
          ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));

  v0::Value tensor_two = TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    // The literal value itself is unused.
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_, CreateCall(_, _, _))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, std::nullopt));

    // We expect the executor to shortcircuit and never call Compute if an
    // intermediate result errors out.
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  EXPECT_THAT(materialize_status,
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateStructWithTwoElements) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value tensor_three = TensorV(3.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("struct_elem1"));
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_three)),
                    _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("struct_elem2"));
    OwnedValueId first_struct_element =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    OwnedValueId second_struct_element =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_three));

    EXPECT_CALL(*mock_executor_service_,
                CreateStruct(_,
                             EqualsProto(CreateStructRequestForValues(
                                 {"struct_elem1", "struct_elem2"})),
                             _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateStructResponse>("source_ref"));

    std::vector<ValueId> struct_to_create = {std::move(first_struct_element),
                                             std::move(second_struct_element)};
    OwnedValueId struct_result =
        TFF_ASSERT_OK(test_executor_->CreateStruct(struct_to_create));

    // We need to call Materialize to force synchronization of the calls
    // above.
    v0::CreateSelectionRequest create_selection_request0;
    create_selection_request0.mutable_executor()->set_id(kExecutorId);
    create_selection_request0.mutable_source_ref()->set_id("source_ref");
    create_selection_request0.set_index(0);

    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(_, EqualsProto(create_selection_request0), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref0"));

    v0::CreateSelectionRequest create_selection_request1;
    create_selection_request1.mutable_executor()->set_id(kExecutorId);
    create_selection_request1.mutable_source_ref()->set_id("source_ref");
    create_selection_request1.set_index(1);

    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(_, EqualsProto(create_selection_request1), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref1"));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(_, EqualsProto(ComputeRequestForId("selection_ref0")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(_, EqualsProto(ComputeRequestForId("selection_ref1")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_three));

    materialize_status =
        test_executor_->Materialize(struct_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  v0::Value expected_value;
  auto expected_struct = expected_value.mutable_struct_();
  *expected_struct->add_element()->mutable_value() = tensor_two;
  *expected_struct->add_element()->mutable_value() = tensor_three;
  EXPECT_THAT(materialized_value, EqualsProto(expected_value));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateStructWithNoElements) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value empty_struct = StructV({});
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    v0::CreateStructRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    EXPECT_CALL(*mock_executor_service_,
                CreateStruct(_, EqualsProto(expected_request), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateStructResponse>("struct_ref"));

    OwnedValueId struct_result =
        TFF_ASSERT_OK(test_executor_->CreateStruct(std::vector<ValueId>{}));

    // We need to call Materialize to force synchronization of the calls above.

    // NOTE: we don't expect any RPCs because the remote executor was tracking
    // that this wasn an empty structure.
    materialize_status =
        test_executor_->Materialize(struct_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(empty_struct));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateStructWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  {
    v0::CreateStructRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    EXPECT_CALL(*mock_executor_service_,
                CreateStruct(_, EqualsProto(expected_request), _))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));

    std::vector<ValueId> struct_to_create = {};
    OwnedValueId struct_result =
        TFF_ASSERT_OK(test_executor_->CreateStruct(struct_to_create));

    // If the CreateStruct fails we dont expect a Compute call on the other side
    v0::Value materialized_value;
    absl::Status materialize_status =
        test_executor_->Materialize(struct_result, &materialized_value);
    EXPECT_THAT(materialize_status,
                StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  }
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateSelection) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("source_ref"));

    EXPECT_CALL(
        *mock_executor_service_,
        CreateStruct(
            _, EqualsProto(CreateStructRequestForValues({"source_ref"})), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateStructResponse>("struct_ref"));

    OwnedValueId source_value =
        TFF_ASSERT_OK(test_executor_->CreateValue(StructV({tensor_two})));

    v0::CreateSelectionRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_source_ref()->set_id("struct_ref");
    expected_request.set_index(0);
    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(_, EqualsProto(expected_request), _))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref"));

    OwnedValueId selection_result =
        TFF_ASSERT_OK(test_executor_->CreateSelection(source_value, 0));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(_, EqualsProto(ComputeRequestForId("selection_ref")), _))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(selection_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(StreamingRemoteExecutorTest, CreateSelectionWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(_, EqualsProto(CreateValueRequestForValue(tensor_two)), _))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("source_ref"));
    OwnedValueId source_value =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    v0::CreateSelectionRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_source_ref()->set_id("source_ref");
    expected_request.set_index(1);
    OwnedValueId selection_result =
        TFF_ASSERT_OK(test_executor_->CreateSelection(source_value, 1));

    // If the CreateSelection fails we dont expect a Compute call on the other
    // side
    materialize_status =
        test_executor_->Materialize(selection_result, &materialized_value);
  }
  EXPECT_THAT(
      materialize_status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Error selecting from non-Struct value")));
  WaitForDisposeExecutor(dispose_notification);
}

struct FederatedStructTestCase {
  std::function<v0::Value(std::vector<v0::Value>)> FederatedV;
  std::function<v0::Value(v0::FunctionType)> FederatedZipIntrinsicV;
  std::string_view placement_uri;
  bool all_equal;
};

class StreamingRemoteExecutorFederatedStructsTest
    : public ::testing::TestWithParam<FederatedStructTestCase>,
      public StreamingRemoteExecutorBase {
 protected:
  using StreamingRemoteExecutorBase::ExpectGetAndDisposeExecutor;
  using StreamingRemoteExecutorBase::WaitForDisposeExecutor;
};

TEST_P(StreamingRemoteExecutorFederatedStructsTest, RoundTripFederatedStruct) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);

  const FederatedStructTestCase& test_case = GetParam();
  // Constructs a <float32, float32>@P.
  const v0::Value tensor_two = TensorV(2.0f);
  const v0::Value tensor_three = TensorV(3.0f);
  const v0::Value struct_value = StructV({tensor_two, tensor_three});
  const v0::Value federated_struct_value = test_case.FederatedV({struct_value});

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    // Add expecations for the sequence of calls that result from a CreateValue.
    {
      EXPECT_CALL(*mock_executor_service_,
                  CreateValue(_,
                              EqualsProto(CreateValueRequestForValue(
                                  test_case.FederatedV({tensor_two}))),
                              _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "federated_elem_0"));
      EXPECT_CALL(*mock_executor_service_,
                  CreateValue(_,
                              EqualsProto(CreateValueRequestForValue(
                                  test_case.FederatedV({tensor_three}))),
                              _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "federated_elem_1"));

      v0::CreateStructRequest inner_create_struct_request;
      inner_create_struct_request.mutable_executor()->set_id(kExecutorId);
      inner_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "federated_elem_0");
      inner_create_struct_request.add_element()->mutable_value_ref()->set_id(
          "federated_elem_1");
      EXPECT_CALL(*mock_executor_service_,
                  CreateStruct(_, EqualsProto(inner_create_struct_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateStructResponse>(
              "streamed_federated_struct"));

      v0::FunctionType zip_at_placement_type_pb;
      {
        auto* param_struct_type =
            zip_at_placement_type_pb.mutable_parameter()->mutable_struct_();
        for (int32_t i = 0; i < 2; ++i) {
          auto* param_federated_type = param_struct_type->add_element()
                                           ->mutable_value()
                                           ->mutable_federated();
          param_federated_type->mutable_member()->mutable_tensor()->set_dtype(
              v0::TensorType::DT_FLOAT);
          param_federated_type->set_all_equal(test_case.all_equal);
          param_federated_type->mutable_placement()->mutable_value()->set_uri(
              std::string(test_case.placement_uri));
        }
        auto* result_federated_type =
            zip_at_placement_type_pb.mutable_result()->mutable_federated();
        result_federated_type->set_all_equal(test_case.all_equal);
        result_federated_type->mutable_placement()->mutable_value()->set_uri(
            std::string(test_case.placement_uri));
        auto* result_struct_type =
            result_federated_type->mutable_member()->mutable_struct_();
        for (int32_t i = 0; i < 2; ++i) {
          result_struct_type->add_element()
              ->mutable_value()
              ->mutable_tensor()
              ->set_dtype(v0::TensorType::DT_FLOAT);
        }
      }
      EXPECT_CALL(
          *mock_executor_service_,
          CreateValue(
              _,
              EqualsProto(CreateValueRequestForValue(
                  test_case.FederatedZipIntrinsicV(zip_at_placement_type_pb))),
              _))
          .WillOnce(
              ReturnOkWithResponseId<v0::CreateValueResponse>("federated_zip"));

      v0::CreateCallRequest zip_call_request;
      zip_call_request.mutable_executor()->set_id(kExecutorId);
      zip_call_request.mutable_function_ref()->set_id("federated_zip");
      zip_call_request.mutable_argument_ref()->set_id(
          "streamed_federated_struct");
      EXPECT_CALL(*mock_executor_service_,
                  CreateCall(_, EqualsProto(zip_call_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>(
              "zipped_federated_struct"));
    }

    // Add exepectations for the sequence of calls that result from a
    // Materialize.
    {
      std::string_view intrinsic_name =
          (test_case.placement_uri == kServerUri ? kFederatedMapAtServerUri
                                                 : kFederatedMapAtClientsUri);
      v0::CreateValueRequest create_map_comp_request;
      create_map_comp_request.mutable_executor()->set_id(kExecutorId);
      CHECK(google::protobuf::TextFormat::ParseFromString(
          absl::Substitute(
              R"pb(
                computation {
                  type {
                    function {
                      parameter {
                        federated {
                          placement { value { uri: "$0" } }
                          member {
                            struct {
                              element { value { tensor { dtype: DT_FLOAT } } }
                              element { value { tensor { dtype: DT_FLOAT } } }
                            }
                          } $1
                        }
                      }
                      result {
                        struct {
                          element {
                            value {
                              federated {
                                placement { value { uri: "$0" } }
                                member { tensor { dtype: DT_FLOAT } } $1
                              }
                            }
                          }
                          element {
                            value {
                              federated {
                                placement { value { uri: "$0" } }
                                member { tensor { dtype: DT_FLOAT } } $1
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  lambda {
                    parameter_name: "federated_struct_arg"
                    result {
                      block {
                        local {
                          name: "elem_0"
                          value {
                            call {
                              function { intrinsic { uri: "$2" } }
                              argument {
                                struct {
                                  element {
                                    value {
                                      lambda {
                                        parameter_name: "map_arg"
                                        result {
                                          selection {
                                            source {
                                              reference { name: "map_arg" }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                  element {
                                    value {
                                      reference { name: "federated_struct_arg" }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                        local {
                          name: "elem_1"
                          value {
                            call {
                              function { intrinsic { uri: "$2" } }
                              argument {
                                struct {
                                  element {
                                    value {
                                      lambda {
                                        parameter_name: "map_arg"
                                        result {
                                          selection {
                                            source {
                                              reference { name: "map_arg" }
                                            }
                                            index: 1
                                          }
                                        }
                                      }
                                    }
                                  }
                                  element {
                                    value {
                                      reference { name: "federated_struct_arg" }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                        result {
                          struct {
                            element { value { reference { name: "elem_0" } } }
                            element { value { reference { name: "elem_1" } } }
                          }
                        }
                      }
                    }
                  }
                }
              )pb",
              test_case.placement_uri,
              test_case.all_equal ? "all_equal: true" : "", intrinsic_name),
          create_map_comp_request.mutable_value()));
      EXPECT_CALL(*mock_executor_service_,
                  CreateValue(_, EqualsProto(create_map_comp_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              absl::StrCat(intrinsic_name, "_selection_comp")));

      v0::CreateCallRequest call_map_request;
      call_map_request.mutable_executor()->set_id(kExecutorId);
      call_map_request.mutable_function_ref()->set_id(
          absl::StrCat(intrinsic_name, "_selection_comp"));
      call_map_request.mutable_argument_ref()->set_id(
          "zipped_federated_struct");
      EXPECT_CALL(*mock_executor_service_,
                  CreateCall(_, EqualsProto(call_map_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>(
              "struct_of_federated_elems"));

      v0::CreateSelectionRequest create_selection_request;
      create_selection_request.mutable_executor()->set_id(kExecutorId);
      create_selection_request.mutable_source_ref()->set_id(
          "struct_of_federated_elems");
      create_selection_request.set_index(0);
      EXPECT_CALL(*mock_executor_service_,
                  CreateSelection(_, EqualsProto(create_selection_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "federated_elem_0"));
      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("federated_elem_0")), _))
          .WillOnce(
              ReturnOkWithComputeResponse(test_case.FederatedV({tensor_two})));

      create_selection_request.set_index(1);
      EXPECT_CALL(*mock_executor_service_,
                  CreateSelection(_, EqualsProto(create_selection_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "federated_elem_1"));
      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("federated_elem_1")), _))
          .WillOnce(ReturnOkWithComputeResponse(
              test_case.FederatedV({tensor_three})));
    }

    // Don't call any executor interface methods until all expectations are set
    // on the executor service above, otherwise this could trigger a TSAN
    // warning where the executor threads are reading expectations while the
    // test thread is writign new expectations.
    OwnedValueId outer_struct_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(federated_struct_value));
    materialize_status =
        test_executor_->Materialize(outer_struct_ref, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(federated_struct_value));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_P(StreamingRemoteExecutorFederatedStructsTest,
       RoundTripFederatedNestedStruct) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);

  const FederatedStructTestCase& test_case = GetParam();
  // Constructs a <<float32>>@P.
  const v0::Value tensor_two = TensorV(2.0f);
  const v0::Value inner_struct_value = StructV({tensor_two});
  const v0::Value outer_struct_value = StructV({inner_struct_value});
  const v0::Value federated_struct_value =
      test_case.FederatedV({outer_struct_value});

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    // Add expecations for the sequence of calls that result from a CreateValue.
    {
      absl::flat_hash_map<std::string_view, std::string_view>
          struct_ref_by_struct_elem_ref = {
              {"inner_federated_struct", "federated_elem"},
              {"outer_federated_struct", "zipped_inner_federated_struct"},
          };
      EXPECT_CALL(*mock_executor_service_,
                  CreateValue(_,
                              EqualsProto(CreateValueRequestForValue(
                                  test_case.FederatedV({tensor_two}))),
                              _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              "federated_elem"));

      // We expect two create struct, and two zip calls, as the executor
      // traverse the nested structure.
      for (const auto& item : struct_ref_by_struct_elem_ref) {
        std::string_view struct_ref = item.first;
        std::string_view elem_ref = item.second;
        EXPECT_CALL(
            *mock_executor_service_,
            CreateStruct(
                _, EqualsProto(CreateStructRequestForValues({elem_ref})), _))
            .WillOnce(ReturnOkWithResponseId<v0::CreateStructResponse>(
                std::string(struct_ref)));

        v0::FunctionType zip_at_placement_type_pb;
        v0::StructType* param_struct_type =
            zip_at_placement_type_pb.mutable_parameter()->mutable_struct_();
        v0::FederatedType* param_federated_type =
            param_struct_type->add_element()
                ->mutable_value()
                ->mutable_federated();
        param_federated_type->set_all_equal(test_case.all_equal);
        param_federated_type->mutable_placement()->mutable_value()->set_uri(
            std::string(test_case.placement_uri));
        v0::Type* param_value_type = param_federated_type->mutable_member();
        if (absl::EndsWith(elem_ref, "struct")) {
          // Nested struct needs another layer.
          param_value_type = param_value_type->mutable_struct_()
                                 ->add_element()
                                 ->mutable_value();
        }
        param_value_type->mutable_tensor()->set_dtype(v0::TensorType::DT_FLOAT);
        v0::FederatedType* result_federated_type =
            zip_at_placement_type_pb.mutable_result()->mutable_federated();
        result_federated_type->set_all_equal(test_case.all_equal);
        result_federated_type->mutable_placement()->mutable_value()->set_uri(
            std::string(test_case.placement_uri));
        v0::StructType* result_struct_type =
            result_federated_type->mutable_member()->mutable_struct_();
        v0::Type* result_value_type =
            result_struct_type->add_element()->mutable_value();
        if (absl::EndsWith(elem_ref, "struct")) {
          // Nested struct needs another layer.
          result_value_type = result_value_type->mutable_struct_()
                                  ->add_element()
                                  ->mutable_value();
        }
        result_value_type->mutable_tensor()->set_dtype(
            v0::TensorType::DT_FLOAT);
        EXPECT_CALL(*mock_executor_service_,
                    CreateValue(_,
                                EqualsProto(CreateValueRequestForValue(
                                    test_case.FederatedZipIntrinsicV(
                                        zip_at_placement_type_pb))),
                                _))
            .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
                absl::StrCat("federated_zip_", elem_ref)));

        v0::CreateCallRequest zip_call_request;
        zip_call_request.mutable_executor()->set_id(kExecutorId);
        zip_call_request.mutable_function_ref()->set_id(
            absl::StrCat("federated_zip_", elem_ref));
        zip_call_request.mutable_argument_ref()->set_id(
            std::string(struct_ref));
        EXPECT_CALL(*mock_executor_service_,
                    CreateCall(_, EqualsProto(zip_call_request), _))
            .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>(
                absl::StrCat("zipped_", struct_ref)));
      }
    }

    // Add exepectations for the sequence of calls that result from a
    // Materialize.
    {
      std::string_view intrinsic_name =
          (test_case.placement_uri == kServerUri ? kFederatedMapAtServerUri
                                                 : kFederatedMapAtClientsUri);
      v0::CreateValueRequest create_map_comp_request;
      create_map_comp_request.mutable_executor()->set_id(kExecutorId);
      CHECK(google::protobuf::TextFormat::ParseFromString(
          absl::Substitute(
              R"pb(
                computation {
                  type {
                    function {
                      parameter {
                        federated {
                          placement { value { uri: "$0" } }
                          member {
                            struct {
                              element {
                                value {
                                  struct {
                                    element {
                                      value { tensor { dtype: DT_FLOAT } }
                                    }
                                  }
                                }
                              }
                            }
                          } $1
                        }
                      }
                      result {
                        struct {
                          element {
                            value {
                              struct {
                                element {
                                  value {
                                    federated {
                                      placement { value { uri: "$0" } }
                                      member { tensor { dtype: DT_FLOAT } } $1
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  lambda {
                    parameter_name: "federated_struct_arg"
                    result {
                      block {
                        local {
                          name: "nested_struct_0"
                          value {
                            call {
                              function { intrinsic { uri: "$2" } }
                              argument {
                                struct {
                                  element {
                                    value {
                                      lambda {
                                        parameter_name: "map_arg"
                                        result {
                                          selection {
                                            source {
                                              reference { name: "map_arg" }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                  element {
                                    value {
                                      reference { name: "federated_struct_arg" }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                        local {
                          name: "elem_0"
                          value {
                            block {
                              local {
                                name: "elem_0"
                                value {
                                  call {
                                    function { intrinsic { uri: "$2" } }
                                    argument {
                                      struct {
                                        element {
                                          value {
                                            lambda {
                                              parameter_name: "map_arg"
                                              result {
                                                selection {
                                                  source {
                                                    reference {
                                                      name: "map_arg"
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                        element {
                                          value {
                                            reference {
                                              name: "nested_struct_0"
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                              result {
                                struct {
                                  element {
                                    value { reference { name: "elem_0" } }
                                  }
                                }
                              }
                            }
                          }
                        }
                        result {
                          struct {
                            element { value { reference { name: "elem_0" } } }
                          }
                        }
                      }
                    }
                  }
                }
              )pb",
              test_case.placement_uri,
              test_case.all_equal ? "all_equal: true" : "", intrinsic_name),
          create_map_comp_request.mutable_value()));
      EXPECT_CALL(*mock_executor_service_,
                  CreateValue(_, EqualsProto(create_map_comp_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>(
              absl::StrCat(intrinsic_name, "_selection_comp")));

      v0::CreateCallRequest call_map_request;
      call_map_request.mutable_executor()->set_id(kExecutorId);
      call_map_request.mutable_function_ref()->set_id(
          absl::StrCat(intrinsic_name, "_selection_comp"));
      call_map_request.mutable_argument_ref()->set_id(
          "zipped_outer_federated_struct");
      EXPECT_CALL(*mock_executor_service_,
                  CreateCall(_, EqualsProto(call_map_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>(
              "struct_of_struct_of_federated_elem"));

      v0::CreateSelectionRequest create_selection_request;
      create_selection_request.mutable_executor()->set_id(kExecutorId);
      create_selection_request.mutable_source_ref()->set_id(
          "struct_of_struct_of_federated_elem");
      create_selection_request.set_index(0);
      EXPECT_CALL(*mock_executor_service_,
                  CreateSelection(_, EqualsProto(create_selection_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "struct_of_federated_elem"));

      create_selection_request.mutable_source_ref()->set_id(
          "struct_of_federated_elem");
      EXPECT_CALL(*mock_executor_service_,
                  CreateSelection(_, EqualsProto(create_selection_request), _))
          .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
              "federated_elem"));

      EXPECT_CALL(
          *mock_executor_service_,
          Compute(_, EqualsProto(ComputeRequestForId("federated_elem")), _))
          .WillOnce(
              ReturnOkWithComputeResponse(test_case.FederatedV({tensor_two})));
    }

    // Don't call any executor interface methods until all expectations are set
    // on the executor service above, otherwise this could trigger a TSAN
    // warning where the executor threads are reading expectations while the
    // test thread is writign new expectations.
    OwnedValueId outer_struct_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(federated_struct_value));
    materialize_status =
        test_executor_->Materialize(outer_struct_ref, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(federated_struct_value));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_P(StreamingRemoteExecutorFederatedStructsTest,
       RoundTripFederatedEmptyStruct) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  const FederatedStructTestCase& test_case = GetParam();
  // Constructs a <>@P.
  const v0::Value federated_struct_value = test_case.FederatedV({StructV({})});

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    // Add expecations for the sequence of calls that result from a CreateValue.
    {
      EXPECT_CALL(
          *mock_executor_service_,
          CreateValue(
              _,
              EqualsProto(CreateValueRequestForValue(federated_struct_value)),
              _))
          .WillOnce(
              ReturnOkWithResponseId<v0::CreateValueResponse>("empty_struct"));
    }

    // Note: no materialization expectations, we want the executor to not send
    // unecessary RPCs to the remote machine.

    // Don't call any executor interface methods until all expectations are set
    // on the executor service above, otherwise this could trigger a TSAN
    // warning where the executor threads are reading expectations while the
    // test thread is writign new expectations.
    OwnedValueId outer_struct_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(federated_struct_value));
    materialize_status =
        test_executor_->Materialize(outer_struct_ref, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(federated_struct_value));
  WaitForDisposeExecutor(dispose_notification);
}

INSTANTIATE_TEST_SUITE_P(
    StreamingRemoteExecutorFederatedStructsTests,
    StreamingRemoteExecutorFederatedStructsTest,

    ::testing::ValuesIn<FederatedStructTestCase>({
        // Test for CLIENTS placement.
        {[](std::vector<v0::Value> values) -> v0::Value {
           return ClientsV(std::move(values));
         },
         [](v0::FunctionType type_pb) -> v0::Value {
           return testing::intrinsic::FederatedZipAtClientsV(type_pb);
         },
         kClientsUri, false},
        // Test for SERVER placement.
        {[](std::vector<v0::Value> values) -> v0::Value {
           return ServerV(values[0]);
         },
         [](v0::FunctionType type_pb) -> v0::Value {
           return testing::intrinsic::FederatedZipAtServerV(type_pb);
         },
         kServerUri, true},
    }),
    [](const ::testing::TestParamInfo<
        StreamingRemoteExecutorFederatedStructsTest::ParamType>& info) {
      return std::string(info.param.placement_uri);
    });

}  // namespace tensorflow_federated
