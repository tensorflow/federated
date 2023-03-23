/* Copyright 2022, The TensorFlow Federated Authors.

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
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"

#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/dtensor/cc/dtensor_device.h"
#include "tensorflow/dtensor/cc/dtensor_device_util.h"
#include "tensorflow/dtensor/cc/mesh_type.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

extern "C" {

void* TFE_DTENSOR_RegisterDTensorDevice(TFE_Context* context,
                                        tensorflow::TF_Mesh* mesh,
                                        const char* dtensor_device_name,
                                        TF_Status* status) {
  TFE_CustomDevice device;
  void* device_info;
  tensorflow::dtensor::AllocateDTensorDevice(
      /*device_name=*/dtensor_device_name, &device, &device_info,
      /*is_async=*/false, /*in_flight_nodes_limit=*/0, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  std::string mesh_string = tensorflow::unwrap(mesh)->ToString();
  TFE_RegisterCustomDevice(context, device, dtensor_device_name, device_info,
                           status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  tensorflow::dtensor::AddMesh(mesh_string, device_info,
                               /*is_host_mesh=*/false, status);

  if (TF_GetCode(status) != TF_OK) return nullptr;

  return device_info;
}

bool TFE_DTENSOR_IsTensorHandleOnDevice(TFE_Context* context,
                                        TFE_TensorHandle* tensor_handle,
                                        const char* device_name,
                                        TF_Status* status) {
  const char* tensor_device = TFE_TensorHandleDeviceName(tensor_handle, status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (strcmp(tensor_device, device_name) == 0) return true;
  return false;
}

TFE_TensorHandle* TFE_DTENSOR_TensorToDTensor(
    TFE_Context* context, TFE_TensorHandle* handle,
    const tensorflow::TF_Layout* layout, const char* device_name,
    TF_Status* status) {
  const tensorflow::dtensor::Layout* layout_object = tensorflow::unwrap(layout);

  if (layout_object->IsFullyReplicated()) {
    TFE_TensorHandle* replicated_result =
        TFE_DTENSOR_CopyToMesh(context, handle, layout, device_name, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    return replicated_result;
  }

  // Perform copy to mesh followed by relayout to get result
  auto replicated_layout = tensorflow::dtensor::Layout::ReplicatedOnMesh(
      layout_object->mesh(), layout_object->rank());
  TFE_TensorHandle* replicated_result = TFE_DTENSOR_CopyToMesh(
      context, handle, tensorflow::wrap(&replicated_layout), device_name,
      status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_TensorHandle* result = TFE_DTENSOR_Relayout(context, replicated_result,
                                                  layout, device_name, status);
  // Delete intermediate result handle from copying to mesh.
  TFE_DeleteTensorHandle(replicated_result);
  return result;
}

TFE_TensorHandle* TFE_DTENSOR_DTensorToTensor(TFE_Context* context,
                                              TFE_TensorHandle* dtensor_handle,
                                              const char* device_name,
                                              TF_Status* status) {
  tensorflow::dtensor::TensorWithLayout* t =
      reinterpret_cast<tensorflow::dtensor::TensorWithLayout*>(
          TFE_TensorHandleDevicePointer(dtensor_handle, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (t->layout().IsFullyReplicated()) {
    // Get the tensor value
    return TFE_TensorHandleCopySharingTensor(t->get_tensor(0), status);
  }

  auto replicated_layout = tensorflow::dtensor::Layout::ReplicatedOnMesh(
      t->layout().mesh(), t->layout().rank());

  TFE_TensorHandle* result = TFE_DTENSOR_Relayout(
      context, dtensor_handle, tensorflow::wrap(&replicated_layout),
      device_name, status);

  tensorflow::dtensor::TensorWithLayout* t_replicated =
      reinterpret_cast<tensorflow::dtensor::TensorWithLayout*>(
          TFE_TensorHandleDevicePointer(result, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;

  auto tensor =
      TFE_TensorHandleCopySharingTensor(t_replicated->get_tensor(0), status);

  TFE_DeleteTensorHandle(result);
  return tensor;
}

TFE_TensorHandle* TFE_DTENSOR_CopyToMesh(TFE_Context* context,
                                         TFE_TensorHandle* tensor_handle,
                                         const tensorflow::TF_Layout* layout,
                                         const char* device_name,
                                         TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "CopyToMesh", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_OpSetDevice(op.get(), device_name, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  std::string serialized_layout = tensorflow::unwrap(layout)->ToString();
  TFE_OpSetAttrString(op.get(), "layout", serialized_layout.data(),
                      serialized_layout.length());
  TFE_OpAddInput(op.get(), tensor_handle, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  int num_results = 1;
  TFE_TensorHandle* replicated_result;
  TFE_Execute(op.get(), &replicated_result, &num_results, status);

  if (TF_GetCode(status) != TF_OK) return nullptr;

  return replicated_result;
}

TFE_TensorHandle* TFE_DTENSOR_Relayout(TFE_Context* context,
                                       TFE_TensorHandle* handle,
                                       const tensorflow::TF_Layout* layout,
                                       const char* device_name,
                                       TF_Status* status) {
  bool is_dtensor =
      TFE_DTENSOR_IsTensorHandleOnDevice(context, handle, device_name, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (!is_dtensor) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        absl::StrCat("Input to Relayout should be a DTensor on device ",
                     device_name)
            .c_str());
    return nullptr;
  }
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> relayout(
      TFE_NewOp(context, "Relayout", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(relayout.get(), device_name, status);

  if (TF_GetCode(status) != TF_OK) return nullptr;

  std::string serialized_layout = tensorflow::unwrap(layout)->ToString();
  TFE_OpSetAttrString(relayout.get(), "layout", serialized_layout.data(),
                      serialized_layout.length());
  TFE_OpAddInput(relayout.get(), handle, status);

  if (TF_GetCode(status) != TF_OK) return nullptr;

  int num_results = 1;
  TFE_TensorHandle* result;
  TFE_Execute(relayout.get(), &result, &num_results, status);

  if (TF_GetCode(status) != TF_OK) return nullptr;
  return result;
}
}
