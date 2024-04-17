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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/data_type.pb.h"

namespace tensorflow_federated {

absl::StatusOr<v0::DataType> DataTypeFromTensorFlowDataType(
    tensorflow::DataType data_type_pb) {
  switch (data_type_pb) {
    case tensorflow::DataType::DT_BOOL:
      return v0::DataType::DT_BOOL;
    case tensorflow::DataType::DT_INT8:
      return v0::DataType::DT_INT8;
    case tensorflow::DataType::DT_INT16:
      return v0::DataType::DT_INT16;
    case tensorflow::DataType::DT_INT32:
      return v0::DataType::DT_INT32;
    case tensorflow::DataType::DT_INT64:
      return v0::DataType::DT_INT64;
    case tensorflow::DataType::DT_UINT8:
      return v0::DataType::DT_UINT8;
    case tensorflow::DataType::DT_UINT16:
      return v0::DataType::DT_UINT16;
    case tensorflow::DataType::DT_UINT32:
      return v0::DataType::DT_UINT32;
    case tensorflow::DataType::DT_UINT64:
      return v0::DataType::DT_UINT64;
    case tensorflow::DataType::DT_HALF:
      return v0::DataType::DT_HALF;
    case tensorflow::DataType::DT_FLOAT:
      return v0::DataType::DT_FLOAT;
    case tensorflow::DataType::DT_DOUBLE:
      return v0::DataType::DT_DOUBLE;
    case tensorflow::DataType::DT_COMPLEX64:
      return v0::DataType::DT_COMPLEX64;
    case tensorflow::DataType::DT_COMPLEX128:
      return v0::DataType::DT_COMPLEX128;
    case tensorflow::DataType::DT_STRING:
      return v0::DataType::DT_STRING;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", data_type_pb));
  }
}

absl::StatusOr<tensorflow::DataType> TensorFlowDataTypeFromDataType(
    v0::DataType data_type_pb) {
  switch (data_type_pb) {
    case v0::DataType::DT_BOOL:
      return tensorflow::DataType::DT_BOOL;
    case v0::DataType::DT_INT8:
      return tensorflow::DataType::DT_INT8;
    case v0::DataType::DT_INT16:
      return tensorflow::DataType::DT_INT16;
    case v0::DataType::DT_INT32:
      return tensorflow::DataType::DT_INT32;
    case v0::DataType::DT_INT64:
      return tensorflow::DataType::DT_INT64;
    case v0::DataType::DT_UINT8:
      return tensorflow::DataType::DT_UINT8;
    case v0::DataType::DT_UINT16:
      return tensorflow::DataType::DT_UINT16;
    case v0::DataType::DT_UINT32:
      return tensorflow::DataType::DT_UINT32;
    case v0::DataType::DT_UINT64:
      return tensorflow::DataType::DT_UINT64;
    case v0::DataType::DT_HALF:
      return tensorflow::DataType::DT_HALF;
    case v0::DataType::DT_FLOAT:
      return tensorflow::DataType::DT_FLOAT;
    case v0::DataType::DT_DOUBLE:
      return tensorflow::DataType::DT_DOUBLE;
    case v0::DataType::DT_COMPLEX64:
      return tensorflow::DataType::DT_COMPLEX64;
    case v0::DataType::DT_COMPLEX128:
      return tensorflow::DataType::DT_COMPLEX128;
    case v0::DataType::DT_STRING:
      return tensorflow::DataType::DT_STRING;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", data_type_pb));
  }
}

absl::StatusOr<v0::ArrayShape> ArrayShapeFromTensorShape(
    const tensorflow::TensorShape& tensor_shape) {
  v0::ArrayShape shape_pb;
  for (int i = 0; i < tensor_shape.dims(); i++) {
    shape_pb.mutable_dim()->Add(tensor_shape.dim_size(i));
  }
  shape_pb.set_unknown_rank(tensor_shape.unknown_rank());
  return shape_pb;
}

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

absl::StatusOr<v0::Array> ArrayFromTensor(const tensorflow::Tensor& tensor) {
  v0::Array array_pb;
  v0::DataType data_type =
      TFF_TRY(DataTypeFromTensorFlowDataType(tensor.dtype()));
  array_pb.set_dtype(data_type);
  v0::ArrayShape shape_pb = TFF_TRY(ArrayShapeFromTensorShape(tensor.shape()));
  array_pb.mutable_shape()->Swap(&shape_pb);

  tensorflow::TensorProto tensor_pb;
  tensor.AsProtoField(&tensor_pb);

  switch (tensor_pb.dtype()) {
    case tensorflow::DataType::DT_BOOL: {
      array_pb.mutable_bool_list()->mutable_value()->Assign(
          tensor_pb.bool_val().begin(), tensor_pb.bool_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT8: {
      array_pb.mutable_int8_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT16: {
      array_pb.mutable_int16_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT32: {
      array_pb.mutable_int32_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT64: {
      array_pb.mutable_int64_list()->mutable_value()->Assign(
          tensor_pb.int64_val().begin(), tensor_pb.int64_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT8: {
      array_pb.mutable_uint8_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT16: {
      array_pb.mutable_uint16_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT32: {
      array_pb.mutable_uint32_list()->mutable_value()->Assign(
          tensor_pb.uint32_val().begin(), tensor_pb.uint32_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT64: {
      array_pb.mutable_uint64_list()->mutable_value()->Assign(
          tensor_pb.uint64_val().begin(), tensor_pb.uint64_val().end());
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      array_pb.mutable_float16_list()->mutable_value()->Assign(
          tensor_pb.half_val().begin(), tensor_pb.half_val().end());
      break;
    }
    case tensorflow::DataType::DT_FLOAT: {
      array_pb.mutable_float32_list()->mutable_value()->Assign(
          tensor_pb.float_val().begin(), tensor_pb.float_val().end());
      break;
    }
    case tensorflow::DataType::DT_DOUBLE: {
      array_pb.mutable_float64_list()->mutable_value()->Assign(
          tensor_pb.double_val().begin(), tensor_pb.double_val().end());
      break;
    }
    case tensorflow::DataType::DT_COMPLEX64: {
      array_pb.mutable_complex64_list()->mutable_value()->Assign(
          tensor_pb.scomplex_val().begin(), tensor_pb.scomplex_val().end());
      break;
    }
    case tensorflow::DataType::DT_COMPLEX128: {
      array_pb.mutable_complex128_list()->mutable_value()->Assign(
          tensor_pb.dcomplex_val().begin(), tensor_pb.dcomplex_val().end());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }

  return array_pb;
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

absl::StatusOr<v0::Array> ArrayContentFromTensor(
    const tensorflow::Tensor& tensor) {
  v0::Array array_pb;
  v0::DataType data_type =
      TFF_TRY(DataTypeFromTensorFlowDataType(tensor.dtype()));
  array_pb.set_dtype(data_type);
  v0::ArrayShape shape_pb = TFF_TRY(ArrayShapeFromTensorShape(tensor.shape()));
  array_pb.mutable_shape()->Swap(&shape_pb);
  tensorflow::TensorProto tensor_pb;
  tensor.AsProtoTensorContent(&tensor_pb);

  LOG(ERROR) << "--- tensor_pb";
  LOG(ERROR) << tensor_pb;

  absl::CopyCordToString(tensor_pb.tensor_content(),
                         array_pb.mutable_content());

  return array_pb;
}

absl::StatusOr<tensorflow::Tensor> TensorFromArrayContent(
    const v0::Array& array_pb) {
  if (!array_pb.has_content()) {
    return absl::UnimplementedError(
        absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }

  tensorflow::TensorProto tensor_pb;
  tensorflow::DataType data_type =
      TFF_TRY(TensorFlowDataTypeFromDataType(array_pb.dtype()));
  tensor_pb.set_dtype(data_type);
  tensorflow::TensorShapeProto shape_pb =
      TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())).AsProto();
  tensor_pb.mutable_tensor_shape()->Swap(&shape_pb);
  *tensor_pb.mutable_tensor_content() = array_pb.content();

  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tensor_pb)) {
    return absl::InvalidArgumentError(
        "Seriailzed tensor proto could not be parsed into Tensor.");
  }
  return tensor;
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
