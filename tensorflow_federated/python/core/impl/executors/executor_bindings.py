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
"""Python interface to C++ Executor implementations."""

from collections.abc import Mapping

from tensorflow_federated.cc.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import data_conversions
from tensorflow_federated.python.core.impl.types import placements

# Serialization methods.
serialize_tensor_value = executor_bindings.serialize_tensor_value
deserialize_tensor_value = executor_bindings.deserialize_tensor_value

# Import classes.
OwnedValueId = executor_bindings.OwnedValueId
Executor = executor_bindings.Executor

# Import executor constructors.
create_reference_resolving_executor = (
    executor_bindings.create_reference_resolving_executor
)
create_composing_executor = executor_bindings.create_composing_executor
create_sequence_executor = executor_bindings.create_sequence_executor

# Import executor constructor helpers.
create_insecure_grpc_channel = executor_bindings.create_insecure_grpc_channel
GRPCChannel = executor_bindings.GRPCChannelInterface


# Wrap any construction requiring cardinalities arguments to convert placement
# literals to strings.
def create_federating_executor(
    inner_server_executor: executor_bindings.Executor,
    inner_client_executor: executor_bindings.Executor,
    cardinalities: Mapping[placements.PlacementLiteral, int],
) -> executor_bindings.Executor:
  """Constructs a FederatingExecutor with a specified placement."""
  uri_cardinalities = (
      data_conversions.convert_cardinalities_dict_to_string_keyed(cardinalities)
  )
  return executor_bindings.create_federating_executor(
      inner_server_executor, inner_client_executor, uri_cardinalities
  )


def create_remote_executor(
    channel: GRPCChannel,
    cardinalities: Mapping[placements.PlacementLiteral, int],
) -> executor_bindings.Executor:
  """Constructs a RemoteExecutor proxying service on `channel`."""
  uri_cardinalities = (
      data_conversions.convert_cardinalities_dict_to_string_keyed(cardinalities)
  )
  return executor_bindings.create_remote_executor(channel, uri_cardinalities)


def create_streaming_remote_executor(
    channel: GRPCChannel,
    cardinalities: Mapping[placements.PlacementLiteral, int],
) -> executor_bindings.Executor:
  """Constructs a StreamingRemoteExecutor proxying service on `channel`."""
  uri_cardinalities = (
      data_conversions.convert_cardinalities_dict_to_string_keyed(cardinalities)
  )
  return executor_bindings.create_streaming_remote_executor(
      channel, uri_cardinalities
  )


def create_composing_child(
    executor: executor_bindings.Executor,
    cardinalities: Mapping[placements.PlacementLiteral, int],
) -> executor_bindings.Executor:
  """Constructs a ComposingChild with specified cardinalities."""
  uri_cardinalities = (
      data_conversions.convert_cardinalities_dict_to_string_keyed(cardinalities)
  )
  return executor_bindings.create_composing_child(executor, uri_cardinalities)
