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

#include "tensorflow_federated/cc/core/impl/executors/dataset_from_tensor_structures.h"

#include <fcntl.h>

#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"

ABSL_FLAG(std::string, reduce_graph_path, "",
          "Path to a serialized GraphDef containing a dataset reduce.");

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::SequenceValueToList;
using ::tensorflow_federated::testing::TensorsProtoEqual;
using ::testing::HasSubstr;
using ::testing::Pointwise;

namespace tf = ::tensorflow;

TEST(DatasetfromTensorStructuresTest, SingleElementDatasetReturnsTensor) {
  tf::Tensor serialized_dataset = TFF_ASSERT_OK(DatasetFromTensorStructures({
      {tf::Tensor(5), tf::Tensor("foo")},
  }));
  EXPECT_EQ(serialized_dataset.dtype(), tf::DT_STRING);
}

TEST(DatasetFromTensorStructuresTest, TwoElementDatasetReturnsTensor) {
  tf::Tensor serialized_dataset = TFF_ASSERT_OK(DatasetFromTensorStructures({
      {tf::Tensor(5), tf::Tensor("foo")},
      {tf::Tensor(6), tf::Tensor("bar")},
  }));
  EXPECT_EQ(serialized_dataset.dtype(), tf::DT_STRING);
}

// Returns a `GraphDef` whose serialized form is stored in a file at `path`.
tf::GraphDef LoadGraph(const char* path) {
  int fd = open(path, O_RDONLY);
  CHECK(fd != -1) << "Failed to open graph at path " << path;
  google::protobuf::io::FileInputStream fs(fd);
  fs.SetCloseOnDelete(true);

  tf::GraphDef graph;
  bool parsed = google::protobuf::TextFormat::Parse(&fs, &graph);
  CHECK(parsed) << "Input graph at path \"" << path
                << "\" is invalid text-format GraphDef";
  return graph;
}

template <typename T>
std::vector<std::vector<tf::Tensor>> ValueStructuresToTensorStructures(
    std::vector<std::vector<T>> inputs) {
  std::vector<std::vector<tf::Tensor>> tensor_structures;
  tensor_structures.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    std::vector<tf::Tensor> structure;
    structure.reserve(inputs[i].size());
    for (const T& input : inputs[i]) {
      structure.push_back(tf::Tensor(input));
    }
    tensor_structures.push_back(std::move(structure));
  }
  return tensor_structures;
}

TEST(DatasetFromTensorStructuresTest, ReturnsDatasetWithElements) {
  std::vector<std::vector<int64_t>> input_values({
      {1, 2, 3},
      {10, 20, 30},
      {100, 200, 300},
  });
  std::vector<std::vector<tf::Tensor>> input_tensors =
      ValueStructuresToTensorStructures(input_values);
  tf::Tensor serialized_dataset =
      TFF_ASSERT_OK(DatasetFromTensorStructures(input_tensors));
  v0::Value::Sequence sequence;
  *sequence.mutable_serialized_graph_def() =
      serialized_dataset.scalar<tf::tstring>()();
  std::vector<std::vector<tf::Tensor>> output_tensors =
      TFF_ASSERT_OK(SequenceValueToList(sequence));
  ASSERT_EQ(input_tensors.size(), output_tensors.size());
  for (size_t i = 0; i < input_tensors.size(); i++) {
    EXPECT_THAT(output_tensors[i],
                Pointwise(TensorsProtoEqual(), input_tensors[i]));
  }
}

TEST(DatasetFromTensorStructuresTest, ReturnsReducibleDataset) {
  std::vector<std::vector<int64_t>> input_values({
      {1, 2, 3},
      {10, 20, 30},
      {100, 200, 300},
  });
  std::vector<std::vector<tf::Tensor>> input_tensors =
      ValueStructuresToTensorStructures(input_values);
  tf::Tensor serialized_dataset =
      TFF_ASSERT_OK(DatasetFromTensorStructures(input_tensors));
  tensorflow::GraphDef reduce_graph_def =
      LoadGraph(FLAGS_reduce_graph_path.CurrentValue().c_str());

  // Attempt to reduce the dataset using a pre-made graph.
  SessionProvider session_provider(std::move(reduce_graph_def));
  auto session = TFF_ASSERT_OK(session_provider.BorrowSession());
  std::vector<tf::Tensor> outputs;
  std::string reduce_dataset_input_name = "serialized_dataset_input";
  std::vector<std::string> output_tensor_names({
      "output_tensor_0",
      "output_tensor_1",
      "output_tensor_2",
  });
  tf::Status status = session->Run(
      {
          {reduce_dataset_input_name, serialized_dataset},
      },
      output_tensor_names,
      /*target_tensor_names=*/{}, &outputs);
  EXPECT_TRUE(status.ok()) << status;
  ASSERT_EQ(outputs.size(), 3);
  for (const tf::Tensor& output : outputs) {
    ASSERT_EQ(output.dtype(), tf::DT_INT64);
  }
  EXPECT_EQ(outputs[0].scalar<int64_t>()(), 111);
  EXPECT_EQ(outputs[1].scalar<int64_t>()(), 222);
  EXPECT_EQ(outputs[2].scalar<int64_t>()(), 333);
}

TEST(DatasetFromTensorStructuresTest, FailsOnMismatchedDimensions) {
  EXPECT_THAT(
      DatasetFromTensorStructures({
          {
              tf::Tensor(tf::DT_INT32, {1}),
          },
          {
              tf::Tensor(tf::DT_INT32, {4}),
          },
      }),
      StatusIs(StatusCode::kUnimplemented, HasSubstr("different dimensions")));
}

TEST(DatasetFromTensorStructuresTest, FailsOnNoStructures) {
  EXPECT_THAT(DatasetFromTensorStructures({}),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("structure list of length zero")));
}

TEST(DatasetFromTensorStructuresTest, FailsOnDifferingLengthStructures) {
  EXPECT_THAT(DatasetFromTensorStructures({
                  {
                      tf::Tensor(5),
                  },
                  {tf::Tensor(5), tf::Tensor(6)},
              }),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("structures of different sizes")));
}

TEST(DatasetFromTensorStructuresTest, FailsOnMismatchedDataTypes) {
  EXPECT_THAT(DatasetFromTensorStructures({
                  {
                      tf::Tensor(5),
                  },
                  {
                      tf::Tensor("foo"),
                  },
              }),
              StatusIs(StatusCode::kInvalidArgument, HasSubstr("dtype")));
}

TEST(DatasetFromTensorStructuresTest, FailsOnMismatchedRanks) {
  EXPECT_THAT(DatasetFromTensorStructures({
                  {
                      tf::Tensor(tf::DataType::DT_INT64, {1, 1, 1}),
                  },
                  {
                      tf::Tensor(tf::DataType::DT_INT64, {1, 1, 1, 1}),
                  },
              }),
              StatusIs(StatusCode::kInvalidArgument, HasSubstr("rank")));
}

}  // namespace

}  // namespace tensorflow_federated
