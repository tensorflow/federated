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

#include "tensorflow_federated/cc/core/impl/executors/xla_utils.h"

#include <algorithm>
#include <complex>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/array.pb.h"

namespace tensorflow_federated {
absl::StatusOr<xla::Shape> ShapeFromArrayShape(xla::PrimitiveType element_type,
                                               const v0::ArrayShape& shape_pb) {
  if (shape_pb.unknown_rank()) {
    return absl::InvalidArgumentError(
        "Shapes of unknown rank are not supported in the XLA executor.");
  }
  return xla::ShapeUtil::MakeValidatedShape(element_type, shape_pb.dim());
}

template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  T* dest) {
  std::copy(src.begin(), src.end(), dest);
}

// Overload for different SrcType and DestType.
template <typename SrcType, typename DestType>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<SrcType>& src,
                                  DestType* dest) {
  std::transform(
      src.begin(), src.end(), dest,
      [](const SrcType& x) -> DestType { return static_cast<DestType>(x); });
}

// Overload for Eigen::half.
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<int32_t>& src,
                                  Eigen::half* dest) {
  // Values of dtype np.float16 are packed to and unpacked from a protobuf
  // field of type int32 using the following logic in order to maintain
  // compatibility with how other external environments (e.g. TensorFlow, Jax)
  // represent values of np.float16.
  std::transform(src.begin(), src.end(), dest, [](int x) -> Eigen::half {
    return Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(x));
  });
}

// Overload for complex.
template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  std::complex<T>* dest) {
  std::copy(src.begin(), src.end(), reinterpret_cast<T*>(dest));
}

absl::StatusOr<xla::Literal> LiteralFromArray(const v0::Array& array_pb) {
  switch (array_pb.kind_case()) {
    case v0::Array::kBoolList: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<bool>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.bool_list().value(),
                            literal.data<bool>().begin());
      return literal;
    }
    case v0::Array::kInt8List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<int8_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.int8_list().value(),
                            literal.data<int8_t>().begin());
      return literal;
    }
    case v0::Array::kInt16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<int16_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.int16_list().value(),
                            literal.data<int16_t>().begin());
      return literal;
    }
    case v0::Array::kInt32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<int32_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.int32_list().value(),
                            literal.data<int32_t>().begin());
      return literal;
    }
    case v0::Array::kInt64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<int64_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.int64_list().value(),
                            literal.data<int64_t>().begin());
      return literal;
    }
    case v0::Array::kUint8List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<uint8_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint8_list().value(),
                            literal.data<uint8_t>().begin());
      return literal;
    }
    case v0::Array::kUint16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<uint16_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint16_list().value(),
                            literal.data<uint16_t>().begin());
      return literal;
    }
    case v0::Array::kUint32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<uint32_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint32_list().value(),
                            literal.data<uint32_t>().begin());
      return literal;
    }
    case v0::Array::kUint64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<uint64_t>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint64_list().value(),
                            literal.data<uint64_t>().begin());
      return literal;
    }
    case v0::Array::kFloat16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<xla::half>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.float16_list().value(),
                            literal.data<xla::half>().begin());
      return literal;
    }
    case v0::Array::kFloat32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<float>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.float32_list().value(),
                            literal.data<float>().begin());
      return literal;
    }
    case v0::Array::kFloat64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<double>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.float64_list().value(),
                            literal.data<double>().begin());
      return literal;
    }
    case v0::Array::kComplex64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<xla::complex64>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex64_list().value(),
                            literal.data<xla::complex64>().begin());
      return literal;
    }
    case v0::Array::kComplex128List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          xla::primitive_util::NativeToPrimitiveType<xla::complex128>(),
          array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex128_list().value(),
                            literal.data<xla::complex128>().begin());
      return literal;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }
}

}  // namespace tensorflow_federated
