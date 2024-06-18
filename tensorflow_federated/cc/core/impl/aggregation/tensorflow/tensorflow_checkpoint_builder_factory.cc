/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/file_system.h"
#include "tsl/platform/env.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/checkpoint_writer.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

using tsl::Env;

// A CheckpointBuilder implementation that builds TensorFlow checkpoints using a
// CheckpointWriter.
class TensorflowCheckpointBuilder : public CheckpointBuilder {
 public:
  explicit TensorflowCheckpointBuilder(std::string filename)
      : filename_(std::move(filename)) {}

  ~TensorflowCheckpointBuilder() override {
    Env::Default()->DeleteFile(filename_).IgnoreError();
  }

  absl::Status Add(const std::string& name, const Tensor& tensor) override {
    return writer_.Add(name, tensor);
  }

  absl::StatusOr<absl::Cord> Build() override {
    TFF_RETURN_IF_ERROR(writer_.Finish());

    // Read the checkpoints contents from the file.
    std::unique_ptr<::tensorflow::RandomAccessFile> file;
    TFF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(filename_, &file));

    absl::Cord output;
    for (;;) {
      char scratch[4096];
      absl::string_view read_result;
      absl::Status status =
          file->Read(output.size(), sizeof(scratch), &read_result, scratch);
      output.Append(read_result);
      if (status.code() == absl::StatusCode::kOutOfRange) {
        return output;
      } else if (!status.ok()) {
        return status;
      }
    }
  }

 private:
  std::string filename_;
  CheckpointWriter writer_{filename_};
};

}  // namespace

std::unique_ptr<CheckpointBuilder> TensorflowCheckpointBuilderFactory::Create()
    const {
  // Create a (likely) unique filename in Tensorflow's RamFileSystem. This
  // results in a second in-memory copy of the data but avoids disk I/O.
  std::string filename =
      absl::StrCat("ram://",
                   absl::Hex(absl::Uniform(
                       absl::BitGen(), 0, std::numeric_limits<int64_t>::max())),
                   ".ckpt");

  return std::make_unique<TensorflowCheckpointBuilder>(std::move(filename));
}

}  // namespace tensorflow_federated::aggregation::tensorflow
