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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_SHAPE_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_SHAPE_TEST_UTILS_H_

#include <cstdint>
#include <initializer_list>

#include "tensorflow_federated/proto/v0/array.pb.h"

namespace tensorflow_federated {
namespace testing {

inline v0::ArrayShape CreateArrayShape(std::initializer_list<int64_t> dims,
                                       bool unknown_rank) {
  v0::ArrayShape shape_pb;
  shape_pb.mutable_dim()->Assign(dims.begin(), dims.end());
  shape_pb.set_unknown_rank(unknown_rank);
  return shape_pb;
}

inline v0::ArrayShape CreateArrayShape(std::initializer_list<int64_t> dims) {
  return CreateArrayShape(dims, false);
}

}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_SHAPE_TEST_UTILS_H_
