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

#include "tensorflow_federated/cc/core/impl/executors/type_utils.h"

#include <cstdint>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

absl::StatusOr<v0::Type> InferTypeFromValue(const v0::Value& value_pb) {
  v0::Type value_type_pb;
  switch (value_pb.value_case()) {
    case v0::Value::kArray: {
      v0::TensorType* tensor_type_pb = value_type_pb.mutable_tensor();

      tensor_type_pb->set_dtype(value_pb.array().dtype());
      for (int64_t dim : value_pb.array().shape().dim()) {
        tensor_type_pb->add_dims(dim);
      }
      break;
    }
    case v0::Value::kStruct: {
      v0::StructType* struct_type = value_type_pb.mutable_struct_();
      for (const v0::Value::Struct::Element& element_pb :
           value_pb.struct_().element()) {
        *struct_type->add_element()->mutable_value() =
            TFF_TRY(InferTypeFromValue(element_pb.value()));
      }
      break;
    }
    case v0::Value::kFederated: {
      *value_type_pb.mutable_federated() = value_pb.federated().type();
      break;
    }
    case v0::Value::kComputation: {
      value_type_pb = value_pb.computation().type();
      break;
    }
    case v0::Value::kSequence: {
      v0::SequenceType* sequence_type = value_type_pb.mutable_sequence();
      *sequence_type->mutable_element() = value_pb.sequence().element_type();
      break;
    }
    default:
      // In all other cases, an error will be returned below.
      return absl::UnimplementedError(absl::StrCat(
          "Cannot infer type for value of case [", value_pb.value_case(), "]"));
  }
  return value_type_pb;
}
}  // namespace tensorflow_federated
