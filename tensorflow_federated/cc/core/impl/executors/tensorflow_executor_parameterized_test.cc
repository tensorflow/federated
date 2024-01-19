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

#include <fcntl.h>

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

ABSL_FLAG(std::string, reduce_graph_path, "",
          "Path to a serialized GraphDef containing a dataset reduce.");

namespace tensorflow_federated {
namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::CreateSerializedRangeDatasetGraphDef;
using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::SequenceV;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorV;
using ::tensorflow_federated::testing::TensorVFromIntList;
using ::tensorflow_federated::testing::intrinsic::ArgsIntoSequenceV;
using ::testing::HasSubstr;
using ::testing::Types;

template <class TfOp>
inline v0::TensorFlow::Binding TensorB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  v0::TensorFlow::Binding binding;
  *binding.mutable_tensor()->mutable_tensor_name() = node->name();
  return binding;
}

template <class TfOp>
inline v0::TensorFlow::Binding SequenceB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  v0::TensorFlow::Binding binding;
  *binding.mutable_sequence()->mutable_variant_tensor_name() = node->name();
  return binding;
}

inline v0::TensorFlow::Binding StructB(
    const absl::Span<const v0::TensorFlow::Binding> elements) {
  v0::TensorFlow::Binding binding;
  auto struct_mut = binding.mutable_struct_();
  for (const auto& element : elements) {
    *struct_mut->add_element() = element;
  }
  return binding;
}

// Placeholder Classes for parameterized testing
class TensorflowExecutor {};
class DTensorExecutor {};

template <class T>
std::shared_ptr<Executor> CreateExecutor(TFE_Context* context);

template <>
std::shared_ptr<Executor> CreateExecutor<TensorflowExecutor>(
    TFE_Context* context) {
  return CreateTensorFlowExecutor(/*max_concurrent_computation_calls=*/10);
}

template <>
std::shared_ptr<Executor> CreateExecutor<DTensorExecutor>(
    TFE_Context* context) {
  return CreateDTensorExecutor(context, std::nullopt, std::nullopt,
                               /*DTensor converter=*/nullptr,
                               /*max_concurrent_computation_calls=*/10);
}

// Enum for Parameterized typed tests.
enum ExecutorId { kDTensorExecutor, kTensorFlowExecutor };

template <class T>
ExecutorId ExecutorType() {}

template <>
ExecutorId ExecutorType<DTensorExecutor>() {
  return kDTensorExecutor;
}

template <>
ExecutorId ExecutorType<TensorflowExecutor>() {
  return kTensorFlowExecutor;
}
inline v0::Value ComputationV(
    std::optional<v0::TensorFlow::Binding> in_binding,
    v0::TensorFlow::Binding out_binding, const tensorflow::Scope& scope,
    const std::optional<const tensorflow::Operation>& init_op = std::nullopt) {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  // NOTE: we do not fill in the `type` field of `comp` because it is not needed
  // by the C++ TensorFlow executor.
  v0::TensorFlow* tensorflow_pb = comp_pb->mutable_tensorflow();
  tensorflow::GraphDef graphdef_pb;
  tensorflow::Status status = scope.ToGraphDef(&graphdef_pb);
  CHECK(status.ok()) << status;
  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  if (in_binding.has_value()) {
    *tensorflow_pb->mutable_parameter() = in_binding.value();
  }
  *tensorflow_pb->mutable_result() = out_binding;
  if (init_op.has_value()) {
    *tensorflow_pb->mutable_initialize_op() = init_op.value().node()->name();
  }
  return value_pb;
}

template <class T>
class TensorFlowBasedExecutorsTest : public ::testing::Test {
 public:
  TensorFlowBasedExecutorsTest() {
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)>
        opts(TFE_NewContextOptions(), TFE_DeleteContextOptions);
    context_ = TFE_NewContext(opts.get(), status);
    TF_DeleteStatus(status);
    test_executor_ = CreateExecutor<T>(context_);
  }

  ~TensorFlowBasedExecutorsTest() override { TFE_DeleteContext(context_); }

  std::shared_ptr<Executor> test_executor_;
  TFE_Context* context_;

  void CheckRoundTrip(const v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(input_pb)));
  }

  ExecutorId Type() { return ExecutorType<T>(); }

  template <typename... Ts>
  void CheckTensorRoundTrip(Ts... tensor_constructor_args) {
    const v0::Value input_pb = TensorV(tensor_constructor_args...);
    CheckRoundTrip(input_pb);
  }

  void CheckCallEqualsProto(const v0::Value& fn,
                            const std::optional<v0::Value>& arg,
                            const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    std::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    TFF_ASSERT_OK_AND_ASSIGN(auto result,
                             test_executor_->CreateCall(fn_id, arg_id));
    TFF_ASSERT_OK_AND_ASSIGN(auto result_proto,
                             test_executor_->Materialize(result));
    EXPECT_THAT(result_proto, EqualsProto(expected));
  }

  void CheckCallRepeatedlyEqualsProto(const v0::Value& fn,
                                      const std::optional<v0::Value>& arg,
                                      const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    std::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    for (int i = 0; i < 3; i++) {
      TFF_ASSERT_OK_AND_ASSIGN(auto result,
                               test_executor_->CreateCall(fn_id, arg_id));
      TFF_ASSERT_OK_AND_ASSIGN(auto result_proto,
                               test_executor_->Materialize(result));
      EXPECT_THAT(result_proto, EqualsProto(expected));
    }
  }

  void CheckCallParallelEqualsProto(const v0::Value& fn,
                                    const std::optional<v0::Value>& arg,
                                    const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    std::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    const int NUM_TEST_THREADS = 32;
    std::vector<std::future<absl::StatusOr<v0::Value>>> results;
    for (int i = 0; i < NUM_TEST_THREADS; i++) {
      results.emplace_back(
          std::async(std::launch::async, [&]() -> absl::StatusOr<v0::Value> {
            return test_executor_->Materialize(
                TFF_TRY(test_executor_->CreateCall(fn_id, arg_id)));
          }));
    }
    for (std::future<absl::StatusOr<v0::Value>>& result_future : results) {
      result_future.wait();
      auto result_status = result_future.get();
      TFF_EXPECT_OK(result_status);
      if (result_status.ok()) {
        EXPECT_THAT(result_status.value(), EqualsProto(expected));
      }
    }
  }
};

typedef Types<TensorflowExecutor, DTensorExecutor> Implementations;

TYPED_TEST_SUITE(TensorFlowBasedExecutorsTest, Implementations);

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateValueEmptyStruct) {
  v0::Value input_pb;
  input_pb.mutable_struct_();
  EXPECT_THAT(this->test_executor_->CreateValue(input_pb), IsOk());
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateValueSimpleTensor) {
  int8_t input_int = 9;
  v0::Value input_pb = TensorV(input_int);
  EXPECT_THAT(this->test_executor_->CreateValue(input_pb), IsOk());
}

// Returns a `GraphDef` whose serialized form is stored in a file at `path`.
tensorflow::GraphDef LoadGraph(const char* path) {
  int fd = open(path, O_RDONLY);
  CHECK(fd != -1) << "Failed to open graph at path " << path;
  google::protobuf::io::FileInputStream fs(fd);
  fs.SetCloseOnDelete(true);

  tensorflow::GraphDef graph;
  bool parsed = google::protobuf::TextFormat::Parse(&fs, &graph);
  CHECK(parsed) << "Input graph at path \"" << path
                << "\" is invalid text-format GraphDef";
  return graph;
}

// Constants necessary for using the dataset-reducing graph created by
// `make_reduce_lambda_test_graph.py`.
char const* const kReduceDatasetInputName = "input_dataset_placeholder";
char const* const kReduceResultOutputTensorName = "result_tensor";

// Returns a value containing a computation which will perform a reduce on an
// input dataset of `int64_t`s and return the sum of the elements.
v0::Value CreateDatasetReduceComputationV() {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  // NOTE: we do not fill in the `type` field of `comp` because it is not needed
  // by the C++ TensorFlow executor.
  v0::TensorFlow* tensorflow_pb = comp_pb->mutable_tensorflow();
  std::string reduce_graph_path = absl::GetFlag(FLAGS_reduce_graph_path);
  tensorflow::GraphDef graphdef_pb = LoadGraph(reduce_graph_path.c_str());
  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  *tensorflow_pb->mutable_parameter()
       ->mutable_sequence()
       ->mutable_variant_tensor_name() = kReduceDatasetInputName;
  *tensorflow_pb->mutable_result()->mutable_tensor()->mutable_tensor_name() =
      kReduceResultOutputTensorName;
  return value_pb;
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallReduceOnSequence) {
  if (this->Type() == kDTensorExecutor) {
    GTEST_SKIP() << "Sequences not supported in DTensor Executor yet";
  }
  int64_t start = 0;
  int64_t stop = 10;
  int64_t step = 2;
  int64_t expected_sum = 0 + 2 + 4 + 6 + 8;
  auto sequence = SequenceV(start, stop, step);
  auto sequence2 = SequenceV(4, 5, 1);
  this->CheckCallEqualsProto(CreateDatasetReduceComputationV(), sequence,
                             TensorV(expected_sum));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, RoundTripEmptyStruct) {
  v0::Value input_pb;
  input_pb.mutable_struct_();
  this->CheckRoundTrip(input_pb);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, RoundTripSimpleTensor) {
  int8_t an_int = 9;
  this->CheckTensorRoundTrip(an_int);
  float a_float = 1.0;
  this->CheckTensorRoundTrip(a_float);
  const char* a_string = "fooey";
  this->CheckTensorRoundTrip(a_string);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, RoundTripStructWithTensor) {
  v0::Value input_pb = StructV({TensorV(9)});
  this->CheckRoundTrip(input_pb);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, RoundTripStructOfNestedTensors) {
  v0::Value input_pb = StructV({StructV({TensorV(24)}), TensorV(88)});
  this->CheckRoundTrip(input_pb);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, RoundTripSequence) {
  if (this->Type() == kDTensorExecutor) {
    GTEST_SKIP() << "Sequences not supported in DTensor Executor yet";
  }
  tensorflow::tstring graph_def = CreateSerializedRangeDatasetGraphDef(
      /*start=*/0, /*stop=*/10, /*step=*/1);
  v0::Value value_pb;
  v0::Value::Sequence* sequence_pb = value_pb.mutable_sequence();
  *sequence_pb->mutable_serialized_graph_def() =
      std::string(graph_def.data(), graph_def.size());
  // We can't simply `this->CheckRoundTrip` because the serialized graph defs
  // don't have deterministic node orders.
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                           this->test_executor_->CreateValue(value_pb));
  v0::Value output_pb;
  EXPECT_THAT(this->test_executor_->Materialize(id, &output_pb), IsOk());
  // Compare GraphDef protos without ordering.
  tensorflow::GraphDef input_graph_def;
  ASSERT_TRUE(input_graph_def.ParseFromString(graph_def));
  tensorflow::GraphDef materialized_graph_def;
  ASSERT_TRUE(materialized_graph_def.ParseFromString(
      output_pb.sequence().serialized_graph_def()));
  EXPECT_THAT(materialized_graph_def,
              testing::proto::IgnoringRepeatedFieldOrdering(
                  EqualsProto(input_graph_def)));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateStructOneElement) {
  v0::Value input = TensorV(5);
  TFF_ASSERT_OK_AND_ASSIGN(auto value,
                           this->test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_,
                           this->test_executor_->CreateStruct({value}));
  TFF_ASSERT_OK_AND_ASSIGN(auto output,
                           this->test_executor_->Materialize(struct_));
  EXPECT_THAT(output, EqualsProto(StructV({input})));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateStructSeveralElements) {
  v0::Value t1 = TensorV(5);
  v0::Value t2 = TensorV(6);
  v0::Value t3 = TensorV(7);
  v0::Value struct_ = StructV({TensorV(5), TensorV(6), TensorV(7)});
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, this->test_executor_->CreateValue(t1));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, this->test_executor_->CreateValue(t2));
  TFF_ASSERT_OK_AND_ASSIGN(auto t3id, this->test_executor_->CreateValue(t3));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto structid, this->test_executor_->CreateStruct({t1id, t2id, t3id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto output,
                           this->test_executor_->Materialize(structid));
  EXPECT_THAT(output, EqualsProto(struct_));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateSelectionFromCreateValue) {
  v0::Value input = StructV({TensorV(1), TensorV(2)});
  TFF_ASSERT_OK_AND_ASSIGN(auto vid, this->test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id,
                           this->test_executor_->CreateSelection(vid, 0));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1, this->test_executor_->Materialize(t1id));
  EXPECT_THAT(t1, EqualsProto(TensorV(1)));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateSelectionFromCreateStruct) {
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id,
                           this->test_executor_->CreateValue(TensorV(1)));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id,
                           this->test_executor_->CreateValue(TensorV(2)));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           this->test_executor_->CreateStruct({t1id, t2id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto selectedid,
                           this->test_executor_->CreateSelection(structid, 1));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected,
                           this->test_executor_->Materialize(selectedid));
  EXPECT_THAT(selected, EqualsProto(TensorV(2)));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateSelectionNonStructImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           this->test_executor_->CreateValue(TensorV(1)));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1, this->test_executor_->Materialize(id));
  EXPECT_THAT(this->test_executor_->CreateSelection(id, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Cannot create selection on non-struct value."));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CreateSelectionOOBImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           this->test_executor_->CreateValue(StructV({})));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1, this->test_executor_->Materialize(id));
  EXPECT_THAT(this->test_executor_->CreateSelection(id, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Attempted to access index 0 of a 0-length struct."));
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallNoOutputTensors) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  // Ignore "sum".
  tensorflow::ops::AddV2 sum(root, x, y);

  auto in_binding = StructB({TensorB(x), TensorB(y)});

  // this->Check that we can make the output binding any old weird shape
  // so long as it has no tensors.
  auto out_binding = StructB({StructB({}), StructB({StructB({})})});

  v0::Value fn = ComputationV(in_binding, out_binding, root);
  v0::Value arg = StructV(
      {TensorV(static_cast<float>(1.0)), TensorV(static_cast<float>(2.0))});
  v0::Value expected = StructV({StructV({}), StructV({StructV({})})});
  this->CheckCallEqualsProto(fn, arg, expected);
  this->CheckCallRepeatedlyEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallNoArgOneOutWithInitialize) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);
  v0::Value fn = ComputationV(
      /*in_binding=*/std::nullopt,
      /*out_binding=*/TensorB(read_var), root,
      /*init_op=*/var_init);
  v0::Value expected = TensorVFromIntList({1, 2, 3});
  this->CheckCallEqualsProto(fn, std::nullopt, expected);
  // Ensure that repeatedly using the same session from the session provider
  // works correctly.
  this->CheckCallRepeatedlyEqualsProto(fn, std::nullopt, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallOneInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder in(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Identity out(root, in);
  v0::Value fn = ComputationV(TensorB(in), TensorB(out), root);
  v0::Value arg = TensorV(5.0);
  v0::Value expected = TensorV(5.0);
  this->CheckCallEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallStructSwapInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder xin(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Placeholder yin(root, tensorflow::DT_INT32);
  tensorflow::ops::Identity xout(root, xin);
  tensorflow::ops::Identity yout(root, yin);
  v0::Value fn = ComputationV(StructB({TensorB(xin), TensorB(yin)}),
                              StructB({TensorB(yout), TensorB(xout)}), root);
  v0::Value arg = StructV({TensorV(5.0), TensorV(1)});
  v0::Value expected = StructV({TensorV(1), TensorV(5.0)});
  this->CheckCallEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallNestedStructSwapInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder xin(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Placeholder yin(root, tensorflow::DT_INT32);
  tensorflow::ops::Identity xout(root, xin);
  tensorflow::ops::Identity yout(root, yin);
  v0::Value fn =
      ComputationV(StructB({StructB({TensorB(xin)}), TensorB(yin)}),
                   StructB({TensorB(yout), StructB({TensorB(xout)})}), root);
  v0::Value arg = StructV({StructV({TensorV(2.0)}), TensorV(4)});
  v0::Value expected = StructV({TensorV(4), StructV({TensorV(2.0)})});
  this->CheckCallEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallAdd) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);
  v0::Value arg = StructV({TensorV(1), TensorV(2)});
  v0::Value expected = TensorV(3);
  this->CheckCallEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, StatefulCallGetsReinitialized) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  tensorflow::ops::AssignVariableOp var_init(
      root, var, tensorflow::ops::Const(root, {0}, shape));
  tensorflow::ops::AssignAddVariableOp var_add_assign(
      root, var, tensorflow::ops::Const(root, {1}, shape));
  tensorflow::ops::ReadVariableOp read_var(
      root.WithControlDependencies({var_add_assign}), var,
      tensorflow::DT_INT32);
  v0::Value fn = ComputationV(std::nullopt, TensorB(read_var), root, var_init);
  v0::Value expected = TensorV(1);
  this->CheckCallEqualsProto(fn, std::nullopt, expected);
  this->CheckCallRepeatedlyEqualsProto(fn, std::nullopt, expected);
  this->CheckCallParallelEqualsProto(fn, std::nullopt, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest, CallWithComputationId) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);
  // Add an ID to the value.
  fn.mutable_computation()->mutable_tensorflow()->mutable_cache_key()->set_id(
      1);
  v0::Value arg = StructV({TensorV(1), TensorV(2)});
  v0::Value expected = TensorV(3);
  // First call will populate the cache.
  this->CheckCallEqualsProto(fn, arg, expected);
  // Call a second time to exercise the cache.
  this->CheckCallEqualsProto(fn, arg, expected);
}

TYPED_TEST(TensorFlowBasedExecutorsTest,
           CallArgsIntoSequenceRequiresAtLeastOneArgument) {
  if (this->Type() == kDTensorExecutor) {
    GTEST_SKIP() << "Sequences not supported in DTensor Executor yet";
  }
  OwnedValueId args_into_sequence =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(ArgsIntoSequenceV()));
  OwnedValueId structures =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(StructV({})));
  OwnedValueId result_id = TFF_ASSERT_OK(
      this->test_executor_->CreateCall(args_into_sequence, structures));
  EXPECT_THAT(this->test_executor_->Materialize(result_id),
              StatusIs(StatusCode::kInvalidArgument, HasSubstr("zero-length")));
}

TYPED_TEST(TensorFlowBasedExecutorsTest,
           CallArgsIntoSequenceStructureReturnsSequence) {
  if (this->Type() == kDTensorExecutor) {
    GTEST_SKIP() << "Sequences not supported in DTensor Executor yet";
  }
  OwnedValueId args_into_sequence =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(ArgsIntoSequenceV()));
  OwnedValueId structures =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(StructV({
          StructV({TensorV(1.0), TensorV("foo")}),
          StructV({TensorV(2.0), TensorV("bar")}),
      })));
  OwnedValueId result_id = TFF_ASSERT_OK(
      this->test_executor_->CreateCall(args_into_sequence, structures));
  v0::Value result =
      TFF_ASSERT_OK(this->test_executor_->Materialize(result_id));
  EXPECT_TRUE(result.has_sequence()) << result.ShortDebugString();
}

TYPED_TEST(TensorFlowBasedExecutorsTest, ArgsIntoSequenceReturnsReducible) {
  if (this->Type() == kDTensorExecutor) {
    GTEST_SKIP() << "Sequences not supported in DTensor Executor yet";
  }
  v0::Value elements_pb =
      StructV({TensorV(int64_t{1}), TensorV(int64_t{10}), TensorV(int64_t{100}),
               TensorV(int64_t{1000})});
  const int64_t expected_sum = 1111;
  OwnedValueId args_into_sequence =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(ArgsIntoSequenceV()));
  OwnedValueId elements =
      TFF_ASSERT_OK(this->test_executor_->CreateValue(elements_pb));
  OwnedValueId sequence = TFF_ASSERT_OK(
      this->test_executor_->CreateCall(args_into_sequence, elements));
  OwnedValueId reduce = TFF_ASSERT_OK(
      this->test_executor_->CreateValue(CreateDatasetReduceComputationV()));
  OwnedValueId result_id =
      TFF_ASSERT_OK(this->test_executor_->CreateCall(reduce, sequence));
  v0::Value result =
      TFF_ASSERT_OK(this->test_executor_->Materialize(result_id));
  EXPECT_THAT(result, EqualsProto(TensorV(expected_sum)));
}

class TensorFlowExecutorTest : public ::testing::Test {
 public:
  TensorFlowExecutorTest() { test_executor_ = CreateTensorFlowExecutor(10); }
  std::shared_ptr<Executor> test_executor_;

  void CheckMaterializeEqual(ValueId id, v0::Value expected_result) {
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(expected_result)));
  }
};

TEST_F(TensorFlowExecutorTest, CreateValueComputationLiteralReturnsResult) {
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

}  // namespace
}  // namespace tensorflow_federated
