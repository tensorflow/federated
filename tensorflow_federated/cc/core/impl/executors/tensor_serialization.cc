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

#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

absl::Status SerializeTensorValue(const tensorflow::Tensor tensor,
                                  v0::Value* value_pb) {
  v0::Array array_pb;
  if (tensor.dtype() == tensorflow::DataType::DT_STRING) {
    array_pb = TFF_TRY(ArrayFromTensor(tensor));
  } else {
    array_pb = TFF_TRY(ArrayContentFromTensor(tensor));
  }
  value_pb->mutable_array()->Swap(&array_pb);
  return absl::OkStatus();
}

absl::StatusOr<tensorflow::Tensor> DeserializeTensorValue(
    const v0::Value& value_pb) {
  if (!value_pb.has_array()) {
    LOG(ERROR) << "Attempted to deserialize non-Array value to a Tensor";
    LOG(ERROR) << "Value proto: " << value_pb.ShortDebugString();
    return absl::InvalidArgumentError(
        "value_pb must have an `array` oneof field to be deserializable to a "
        "Tensor");
  }

  // (value_pb.array().shape().dim().size() == 0 &&
  //      !value_pb.array().shape().unknown_rank()) ||

  if (value_pb.array().dtype() == v0::DataType::DT_STRING) {
    return TensorFromArray(value_pb.array());
  } else {
    return TensorFromArrayContent(value_pb.array());
  }
}

}  // namespace tensorflow_federated
