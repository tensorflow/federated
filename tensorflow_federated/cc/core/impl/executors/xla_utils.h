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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_

#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"

namespace tensorflow_federated {

// Creates a xla::PrimitiveType from a ::tensorflow_federated::v0::DataType.
absl::StatusOr<xla::PrimitiveType> PrimitiveTypeFromDataType(
    v0::DataType data_type);

// Creates a xla::Shape from a v0::TensorType.
absl::StatusOr<xla::Shape> ShapeFromTensorType(
    const v0::TensorType& tensor_type_pb);

// Creates a xla::Shape from a v0::ArrayShape.
absl::StatusOr<xla::Shape> ShapeFromArrayShape(v0::DataType data_type,
                                               const v0::ArrayShape& shape_pb);

// Creates a xla::Literal from a v0::Array.
absl::StatusOr<xla::Literal> LiteralFromArray(const v0::Array& array_pb);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_
