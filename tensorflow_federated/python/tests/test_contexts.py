# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contexts and constructors for integration testing."""

import os

import portpicker
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import remote_runtime_test_utils

WORKER_PORTS = [portpicker.pick_unused_port() for _ in range(2)]
AGGREGATOR_PORTS = [portpicker.pick_unused_port() for _ in range(2)]

# Hack to point Python to the Bazel-built binary.
file_dir = os.path.dirname(os.path.realpath(__file__))
split_dir = file_dir.split(os.path.sep)
# At least, I think its 3
workspace_root = os.path.sep.join(split_dir[:-3])
_TFF_SERVICE_BINARY_PATH = os.path.join(
    workspace_root, 'bazel-bin',
    'tensorflow_federated/cc/simulation/worker_binary')


def _create_local_python_mergeable_comp_context():
  async_context = tff.backends.native.create_local_async_python_execution_context(
  )
  return tff.backends.native.create_mergeable_comp_execution_context(
      [async_context])


def create_sequence_op_supporting_context():
  executor_factory = tff.framework.local_executor_factory(
      support_sequence_ops=True)
  return tff.framework.ExecutionContext(
      executor_fn=executor_factory,
      compiler_fn=tff.backends.native.compiler.transform_to_native_form)  # pytype: disable=wrong-arg-types


def create_localhost_cpp_context():
  return tff.backends.native.create_localhost_cpp_execution_context(
      _TFF_SERVICE_BINARY_PATH)


def _get_all_contexts():
  """Returns a list containing a (name, context_fn) tuple for each context."""
  # pylint: disable=unnecessary-lambda
  # pyformat: disable
  return [
      ('native_mergeable_python',
       lambda: _create_local_python_mergeable_comp_context()),
      ('native_local_python',
       tff.backends.native.create_local_python_execution_context),
      ('native_localhost_cpp',
       create_localhost_cpp_context),
      ('native_remote',
       lambda: remote_runtime_test_utils.create_localhost_remote_context(WORKER_PORTS),
       lambda: remote_runtime_test_utils.create_inprocess_worker_contexts(WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       lambda: remote_runtime_test_utils.create_localhost_remote_context(AGGREGATOR_PORTS),
       lambda: remote_runtime_test_utils.create_inprocess_aggregator_contexts(WORKER_PORTS, AGGREGATOR_PORTS)),
      ('native_sizing',
       tff.backends.native.create_sizing_execution_context),
      ('native_thread_debug',
       tff.backends.native.create_thread_debugging_execution_context),
      ('test_python',
       tff.backends.test.create_test_python_execution_context),
  ]
  # pyformat: enable
  # pylint: enable=unnecessary-lambda
