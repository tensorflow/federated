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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/refcount.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

namespace tf = ::tensorflow;

using ::tensorflow_federated::aggregation::TensorData;

// A primitive TensorData implementation that wraps the original
// tf::Tensor data.
// NumericTensorDataAdapter gets the ownership of the wrapped tensor, which
// keeps the underlying data alive.
class NumericTensorDataAdapter : public TensorData {
 public:
  explicit NumericTensorDataAdapter(std::unique_ptr<tf::Tensor> tensor)
      : tensor_(std::move(tensor)) {}

  // The source tf::Tensor has the data as one continuous blob.
  size_t byte_size() const override { return tensor_->tensor_data().size(); }
  const void* data() const override { return tensor_->tensor_data().data(); }

 private:
  std::unique_ptr<tf::Tensor> tensor_;
};

// Similar to  NumericTensorDataAdapter but performs additional conversion
// of the original tensor tstring values to string_view while keeping the
// the tstring values owned by the original tensor.
class StringTensorDataAdapter : public TensorData {
 public:
  explicit StringTensorDataAdapter(std::unique_ptr<tf::Tensor> tensor)
      : tensor_(std::move(tensor)), string_views_(tensor_->NumElements()) {
    auto string_values = tensor_->flat<tf::tstring>();
    for (size_t i = 0; i < string_values.size(); ++i) {
      string_views_[i] = string_values(i);
    }
  }

  size_t byte_size() const override {
    return string_views_.size() * sizeof(absl::string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  std::unique_ptr<tf::Tensor> tensor_;
  std::vector<absl::string_view> string_views_;
};

// A TensorBuffer that wraps a string Aggregation Tensor and does not delete it.
// This TensorBuffer also owns the array of tstrings that are views of the
// data in the wrapped Tensor.
class WrappedStringAggregationTensorBuffer : public tf::TensorBuffer {
 public:
  explicit WrappedStringAggregationTensorBuffer(
      Tensor tensor, std::unique_ptr<tf::tstring[]> owning_tstring_arr,
      size_t data_size)
      : tf::TensorBuffer(owning_tstring_arr.get()),
        size_(data_size),
        tensor_(std::move(tensor)),
        owning_arr_(std::move(owning_tstring_arr)) {}

  size_t size() const override { return size_; }

  tf::TensorBuffer* root_buffer() override { return this; }
  bool OwnsMemory() const override { return true; }

  void FillAllocationDescription(
      tf::AllocationDescription* proto) const override {
    proto->set_requested_bytes(size_);
  }

 private:
  size_t size_;
  Tensor tensor_;
  std::unique_ptr<tf::tstring[]> owning_arr_;
};

// A TensorBuffer that wraps a numeric Aggregation Tensor and does not delete
// it.
class WrappedNumericAggregationTensorBuffer : public tf::TensorBuffer {
 public:
  // We must apply a const_cast as the tf::Tensor requires the data to be
  // non-const. In the places we intend this code to be used, we should not be
  // mutating the original Tensor data; however, it would not be safe in all
  // cases. We prefer this to a copy for efficiency purposes.
  explicit WrappedNumericAggregationTensorBuffer(Tensor tensor)
      : tf::TensorBuffer(const_cast<void*>(tensor.data().data())),
        size_(tensor.data().byte_size()),
        tensor_(std::move(tensor)) {}

  size_t size() const override { return size_; }

  tf::TensorBuffer* root_buffer() override { return this; }
  bool OwnsMemory() const override { return true; }

  void FillAllocationDescription(
      tf::AllocationDescription* proto) const override {
    proto->set_requested_bytes(size_);
  }

 private:
  size_t size_;
  Tensor tensor_;
};

}  // namespace

StatusOr<DataType> ToAggDataType(tf::DataType dtype) {
  switch (dtype) {
    case tf::DT_FLOAT:
      return DT_FLOAT;
    case tf::DT_DOUBLE:
      return DT_DOUBLE;
    case tf::DT_INT32:
      return DT_INT32;
    case tf::DT_INT64:
      return DT_INT64;
    case tf::DT_STRING:
      return DT_STRING;
    default:
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Unsupported tf::DataType: " << dtype;
  }
}

TensorShape ToAggShape(const tf::TensorShape& shape) {
  TFF_CHECK(shape.IsFullyDefined());
  std::vector<size_t> dim_sizes;
  for (auto dim_size : shape.dim_sizes()) {
    TFF_CHECK(dim_size >= 0);
    dim_sizes.push_back(dim_size);
  }
  return TensorShape(dim_sizes.begin(), dim_sizes.end());
}

TensorShape ToAggShape(const tf::PartialTensorShape& shape) {
  std::vector<size_t> dim_sizes;
  for (auto dim_size : shape.dim_sizes()) {
    // Unknown dimension is supported for PartialTensorShape.
    TFF_CHECK(dim_size >= -1);
    dim_sizes.push_back(dim_size);
  }
  return TensorShape(dim_sizes.begin(), dim_sizes.end());
}

StatusOr<TensorSpec> ToAggTensorSpec(
    const ::tensorflow::TensorSpecProto& spec) {
  TFF_ASSIGN_OR_RETURN(DataType dtype, ToAggDataType(spec.dtype()));
  tf::PartialTensorShape tf_shape;
  if (!tf::PartialTensorShape::BuildPartialTensorShape(spec.shape(), &tf_shape)
           .ok()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "Unsupported tf::PartialTensorShape: "
           << spec.shape().DebugString();
  }
  return TensorSpec(spec.name(), dtype, ToAggShape(tf_shape));
}

// Conversion of tensor data for numeric data types, which can be
// done by simply wrapping the original tensorflow tensor data.
template <typename t>
std::unique_ptr<TensorData> ToAggTensorData(
    std::unique_ptr<tf::Tensor> tensor) {
  return std::make_unique<NumericTensorDataAdapter>(std::move(tensor));
}

// Specialization of ToAggTensorData for the DT_STRING data type.
template <>
std::unique_ptr<TensorData> ToAggTensorData<string_view>(
    std::unique_ptr<tf::Tensor> tensor) {
  return std::make_unique<StringTensorDataAdapter>(std::move(tensor));
}

StatusOr<Tensor> ToAggTensor(const ::tensorflow::TensorProto& tensor_proto) {
  tf::Tensor tf_tensor;
  if (!tf_tensor.FromProto(tensor_proto)) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "Failed to parse TensorProto into tf::Tensor: ";
  }
  return ToAggTensor(std::make_unique<tf::Tensor>(std::move(tf_tensor)));
}

StatusOr<Tensor> ToAggTensor(std::unique_ptr<tf::Tensor> tensor) {
  TFF_ASSIGN_OR_RETURN(DataType dtype, ToAggDataType(tensor->dtype()));
  TensorShape shape = ToAggShape(tensor->shape());
  std::unique_ptr<TensorData> data;
  DTYPE_CASES(dtype, T, data = ToAggTensorData<T>(std::move(tensor)));
  return Tensor::Create(dtype, std::move(shape), std::move(data));
}

absl::StatusOr<tf::DataType> ToTfDataType(DataType dtype) {
  switch (dtype) {
    case DT_FLOAT:
      return tf::DT_FLOAT;
    case DT_DOUBLE:
      return tf::DT_DOUBLE;
    case DT_INT32:
      return tf::DT_INT32;
    case DT_INT64:
      return tf::DT_INT64;
    case DT_STRING:
      return tf::DT_STRING;
    case DT_UINT64:
      return tf::DT_UINT64;
    default:
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Unsupported Aggregation DataType: " << dtype;
  }
}

absl::StatusOr<tf::TensorShape> ToTfShape(
    const ::tensorflow_federated::aggregation::TensorShape& shape) {
  tf::TensorShape tf_tensor_shape;
  TFF_RETURN_IF_ERROR(
      tf::TensorShape::BuildTensorShape(shape.dim_sizes(), &tf_tensor_shape));
  return tf_tensor_shape;
}

absl::StatusOr<tf::Tensor> ToTfTensor(Tensor tensor) {
  if (!tensor.is_dense()) {
    return absl::InvalidArgumentError(
        "ConvertTensor only currently supports dense tensors.");
  }
  TFF_ASSIGN_OR_RETURN(tf::DataType dtype, ToTfDataType(tensor.dtype()));
  TFF_ASSIGN_OR_RETURN(tf::TensorShape shape, ToTfShape(tensor.shape()));

  tsl::core::RefCountPtr<tf::TensorBuffer> tensor_buffer;
  if (tensor.dtype() == DT_STRING) {
    // Views the string_views in the TFF Tensor to an array of TensorFlow
    // tstrings for the data of the TF Tensor.
    auto string_data = tensor.AsSpan<string_view>();
    size_t num_strings = string_data.size();
    auto tstring_arr = std::make_unique<tf::tstring[]>(num_strings);
    for (int i = 0; i < num_strings; ++i) {
      tstring_arr[i] =
          tf::tstring::view(string_data[i].data(), string_data[i].size());
    }
    tensor_buffer.reset(new WrappedStringAggregationTensorBuffer(
        std::move(tensor), std::move(tstring_arr),
        num_strings * sizeof(tf::tstring)));
  } else {
    tensor_buffer.reset(
        new WrappedNumericAggregationTensorBuffer(std::move(tensor)));
  }
  return tf::Tensor(dtype, std::move(shape), std::move(tensor_buffer));
}

}  // namespace tensorflow_federated::aggregation::tensorflow
