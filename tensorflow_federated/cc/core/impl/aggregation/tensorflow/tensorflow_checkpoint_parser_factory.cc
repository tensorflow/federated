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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/file_system.h"
#include "tsl/platform/env.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/checkpoint_reader.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

using ::tensorflow::Env;

// A CheckpointParser implementation that reads TensorFlow checkpoints using a
// CheckpointReader.
class TensorflowCheckpointParser : public CheckpointParser {
 public:
  TensorflowCheckpointParser(std::string filename,
                             std::unique_ptr<CheckpointReader> reader)
      : filename_(std::move(filename)), reader_(std::move(reader)) {}

  ~TensorflowCheckpointParser() override {
    Env::Default()->DeleteFile(filename_).IgnoreError();
  }

  absl::StatusOr<Tensor> GetTensor(const std::string& name) override {
    return reader_->GetTensor(name);
  }

 private:
  std::string filename_;
  std::unique_ptr<CheckpointReader> reader_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<CheckpointParser>>
TensorflowCheckpointParserFactory::Create(
    const absl::Cord& serialized_checkpoint) const {
  // Create a (likely) unique filename in Tensorflow's RamFileSystem. This
  // results in a second in-memory copy of the data but avoids disk I/O.
  std::string filename =
      absl::StrCat("ram://",
                   absl::Hex(absl::Uniform(
                       absl::BitGen(), 0, std::numeric_limits<int64_t>::max())),
                   ".ckpt");

  // Write the checkpoint to the temporary file.
  std::unique_ptr<::tensorflow::WritableFile> file;
  TFF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(filename, &file));
  absl::Cleanup cleanup = [&] {
    Env::Default()->DeleteFile(filename).IgnoreError();
  };
  for (absl::string_view chunk : serialized_checkpoint.Chunks()) {
    TFF_RETURN_IF_ERROR(file->Append(chunk));
  }
  TFF_RETURN_IF_ERROR(file->Close());

  // Return a TensorflowCheckpointParser that will read from the file.
  TFF_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointReader> reader,
                       CheckpointReader::Create(filename));
  std::move(cleanup).Cancel();
  return std::make_unique<TensorflowCheckpointParser>(std::move(filename),
                                                      std::move(reader));
}

}  // namespace tensorflow_federated::aggregation::tensorflow
