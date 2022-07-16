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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_EXECUTOR_H_
#include <memory>
#include <string>

#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

// Returns an executor that can resolve XLA computations and structures
// of `v0::Values` of Tensor type. The platform name parameter will be used to
// create an XLA client against which we can execute XLA calls; the specified
// platform is assumed to be registered in TensorFlow's MultiPlatformManager,
// e.g. by including appropriate build dependencies. This string is
// case-insensitive. The default value of "Host" is guaranteed to be valid.
std::shared_ptr<Executor> CreateXLAExecutor(std::string platform_name = "Host");

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_EXECUTOR_H_
