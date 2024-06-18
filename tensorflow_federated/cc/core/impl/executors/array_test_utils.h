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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_TEST_UTILS_H_

#include <complex>
#include <cstdint>
#include <initializer_list>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"

namespace tensorflow_federated {
namespace testing {

template <typename T>
inline absl::StatusOr<v0::Array> CreateArray(v0::DataType dtype,
                                             v0::ArrayShape shape_pb,
                                             std::initializer_list<T> values) {
  v0::Array array_pb;
  array_pb.set_dtype(dtype);
  array_pb.mutable_shape()->Swap(&shape_pb);
  switch (dtype) {
    case v0::DataType::DT_BOOL: {
      array_pb.mutable_bool_list()->mutable_value()->Assign(values.begin(),
                                                            values.end());
      break;
    }
    case v0::DataType::DT_INT8: {
      array_pb.mutable_int8_list()->mutable_value()->Assign(values.begin(),
                                                            values.end());
      break;
    }
    case v0::DataType::DT_INT16: {
      array_pb.mutable_int16_list()->mutable_value()->Assign(values.begin(),
                                                             values.end());
      break;
    }
    case v0::DataType::DT_INT32: {
      array_pb.mutable_int32_list()->mutable_value()->Assign(values.begin(),
                                                             values.end());
      break;
    }
    case v0::DataType::DT_INT64: {
      array_pb.mutable_int64_list()->mutable_value()->Assign(values.begin(),
                                                             values.end());
      break;
    }
    case v0::DataType::DT_UINT8: {
      array_pb.mutable_uint8_list()->mutable_value()->Assign(values.begin(),
                                                             values.end());
      break;
    }
    case v0::DataType::DT_UINT16: {
      array_pb.mutable_uint16_list()->mutable_value()->Assign(values.begin(),
                                                              values.end());
      break;
    }
    case v0::DataType::DT_UINT32: {
      array_pb.mutable_uint32_list()->mutable_value()->Assign(values.begin(),
                                                              values.end());
      break;
    }
    case v0::DataType::DT_UINT64: {
      array_pb.mutable_uint64_list()->mutable_value()->Assign(values.begin(),
                                                              values.end());
      break;
    }
    case v0::DataType::DT_FLOAT: {
      array_pb.mutable_float32_list()->mutable_value()->Assign(values.begin(),
                                                               values.end());
      break;
    }
    case v0::DataType::DT_DOUBLE: {
      array_pb.mutable_float64_list()->mutable_value()->Assign(values.begin(),
                                                               values.end());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", dtype));
  }
  return array_pb;
}

// Overload for Eigen::half.
inline absl::StatusOr<v0::Array> CreateArray(
    v0::DataType dtype, v0::ArrayShape shape_pb,
    std::initializer_list<const Eigen::half> values) {
  v0::Array array_pb;
  array_pb.set_dtype(dtype);
  array_pb.mutable_shape()->Swap(&shape_pb);
  switch (dtype) {
    case v0::DataType::DT_HALF: {
      auto size = values.size();
      array_pb.mutable_float16_list()->mutable_value()->Reserve(size);
      for (auto element : values) {
        array_pb.mutable_float16_list()->mutable_value()->AddAlreadyReserved(
            Eigen::numext::bit_cast<uint16_t>(element));
      }
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", dtype));
  }
  return array_pb;
}

// Overload for complex.
template <typename T>
inline absl::StatusOr<v0::Array> CreateArray(
    v0::DataType dtype, v0::ArrayShape shape_pb,
    std::initializer_list<std::complex<T>> values) {
  v0::Array array_pb;
  array_pb.set_dtype(dtype);
  array_pb.mutable_shape()->Swap(&shape_pb);
  const T* begin = reinterpret_cast<const T*>(values.begin());
  switch (dtype) {
    case v0::DataType::DT_COMPLEX64: {
      array_pb.mutable_complex64_list()->mutable_value()->Assign(
          begin, begin + values.size() * 2);
      break;
    }
    case v0::DataType::DT_COMPLEX128: {
      array_pb.mutable_complex128_list()->mutable_value()->Assign(
          begin, begin + values.size() * 2);
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", dtype));
  }
  return array_pb;
}

// Overload for string.
inline absl::StatusOr<v0::Array> CreateArray(
    v0::DataType dtype, v0::ArrayShape shape_pb,
    std::initializer_list<const char*> values) {
  v0::Array array_pb;
  array_pb.set_dtype(dtype);
  array_pb.mutable_shape()->Swap(&shape_pb);
  switch (dtype) {
    case v0::DT_STRING: {
      array_pb.mutable_string_list()->mutable_value()->Assign(values.begin(),
                                                              values.end());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", dtype));
  }
  return array_pb;
}

}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_ARRAY_TEST_UTILS_H_
