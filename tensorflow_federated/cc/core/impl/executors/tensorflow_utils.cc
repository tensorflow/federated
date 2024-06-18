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

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/array.pb.h"

namespace tensorflow_federated {

absl::StatusOr<tensorflow::TensorShape> TensorShapeFromArrayShape(
    const v0::ArrayShape& shape_pb) {
  if (shape_pb.unknown_rank()) {
    return absl::InvalidArgumentError(
        "Expected v0::ArrayShape to have a known rank, try constructing "
        "a tensorflow::PartialTensorShape using "
        "tensorflow_federated::PartialTensorShapeFromArrayShape instead.");
  }

  tensorflow::TensorShape shape;
  TFF_TRY(tensorflow::TensorShape::BuildTensorShape(shape_pb.dim(), &shape));
  return shape;
}

tensorflow::PartialTensorShape PartialTensorShapeFromArrayShape(
    const v0::ArrayShape& shape_pb) {
  if (!shape_pb.unknown_rank()) {
    return tensorflow::PartialTensorShape(shape_pb.dim());
  } else {
    return tensorflow::PartialTensorShape();
  }
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

// Overload for string.
static void CopyFromRepeatedField(
    const google::protobuf::RepeatedPtrField<std::string>& src,
    tensorflow::tstring* dest) {
  std::copy(src.begin(), src.end(), dest);
}

absl::StatusOr<tensorflow::Tensor> TensorFromArray(const v0::Array& array_pb) {
  switch (array_pb.kind_case()) {
    case v0::Array::kBoolList: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<bool>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.bool_list().value(),
                            tensor.flat<bool>().data());
      return tensor;
    }
    case v0::Array::kInt8List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int8_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int8_list().value(),
                            tensor.flat<int8_t>().data());
      return tensor;
    }
    case v0::Array::kInt16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int16_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int16_list().value(),
                            tensor.flat<int16_t>().data());
      return tensor;
    }
    case v0::Array::kInt32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int32_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int32_list().value(),
                            tensor.flat<int32_t>().data());
      return tensor;
    }
    case v0::Array::kInt64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int64_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int64_list().value(),
                            tensor.flat<int64_t>().data());
      return tensor;
    }
    case v0::Array::kUint8List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint8_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint8_list().value(),
                            tensor.flat<uint8_t>().data());
      return tensor;
    }
    case v0::Array::kUint16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint16_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint16_list().value(),
                            tensor.flat<uint16_t>().data());
      return tensor;
    }
    case v0::Array::kUint32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint32_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint32_list().value(),
                            tensor.flat<uint32_t>().data());
      return tensor;
    }
    case v0::Array::kUint64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint64_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint64_list().value(),
                            tensor.flat<uint64_t>().data());
      return tensor;
    }
    case v0::Array::kFloat16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<Eigen::half>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float16_list().value(),
                            tensor.flat<Eigen::half>().data());
      return tensor;
    }
    case v0::Array::kFloat32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<float>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float32_list().value(),
                            tensor.flat<float>().data());
      return tensor;
    }
    case v0::Array::kFloat64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<double>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float64_list().value(),
                            tensor.flat<double>().data());
      return tensor;
    }
    case v0::Array::kComplex64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::complex64>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex64_list().value(),
                            tensor.flat<tensorflow::complex64>().data());
      return tensor;
    }
    case v0::Array::kComplex128List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::complex128>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex128_list().value(),
                            tensor.flat<tensorflow::complex128>().data());
      return tensor;
    }
    case v0::Array::kStringList: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::tstring>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.string_list().value(),
                            tensor.flat<tensorflow::tstring>().data());
      return tensor;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }
}

std::string GetNodeName(absl::string_view tensor_name) {
  absl::string_view::size_type pos = tensor_name.find(':');
  if (pos == absl::string_view::npos) {
    return std::string(tensor_name);
  } else {
    return std::string(tensor_name.substr(0, pos));
  }
}

}  // namespace tensorflow_federated
