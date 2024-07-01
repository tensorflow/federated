/* Copyright 2023, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow_federated/proto/v0/array.pb.h"

namespace tensorflow_federated {

// Creates a tensorflow::TensorShape from a v0::ArrayShape.
absl::StatusOr<tensorflow::TensorShape> TensorShapeFromArrayShape(
    const v0::ArrayShape& shape_pb);

// Creates a tensorflow::PartialTensorShape from a v0::ArrayShape.
tensorflow::PartialTensorShape PartialTensorShapeFromArrayShape(
    const v0::ArrayShape& shape_pb);

// Creates an v0::Array from a tensorflow::Tensor.
absl::StatusOr<v0::Array> ArrayFromTensor(const tensorflow::Tensor& tensor);
absl::StatusOr<v0::Array> ArrayContentFromTensor(
    const tensorflow::Tensor& tensor);

// Creates a tensorflow::Tensor from an v0::Array.
absl::StatusOr<tensorflow::Tensor> TensorFromArray(const v0::Array& array_pb);
absl::StatusOr<tensorflow::Tensor> TensorFromArrayContent(
    const v0::Array& array_pb);

std::string GetNodeName(absl::string_view tensor_name);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_UTILS_H_
