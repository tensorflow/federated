# Copyright 2018, The TensorFlow Federated Authors.
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
"""Libraries for extending the TensorFlow Federated core library."""

from tensorflow_federated.python.core.impl.compiler.building_blocks import ComputationBuildingBlock
from tensorflow_federated.python.core.impl.compiler.intrinsic_reductions import replace_intrinsics_with_bodies
from tensorflow_federated.python.core.impl.compiler.tree_transformations import check_disallowed_ops
from tensorflow_federated.python.core.impl.computation.computation_impl import ConcreteComputation
from tensorflow_federated.python.core.impl.computation.computation_serialization import deserialize_computation
from tensorflow_federated.python.core.impl.computation.computation_serialization import serialize_computation
from tensorflow_federated.python.core.impl.context_stack.context_base import Context
from tensorflow_federated.python.core.impl.context_stack.context_stack_base import ContextStack
from tensorflow_federated.python.core.impl.context_stack.get_context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack.set_default_context import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts.sync_execution_context import ExecutionContext
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors.cardinalities_utils import merge_cardinalities
from tensorflow_federated.python.core.impl.executors.cardinality_carrying_base import CardinalityCarrying
from tensorflow_federated.python.core.impl.executors.data_backend_base import DataBackend
from tensorflow_federated.python.core.impl.executors.data_descriptor import CardinalityFreeDataDescriptor
from tensorflow_federated.python.core.impl.executors.data_descriptor import DataDescriptor
from tensorflow_federated.python.core.impl.executors.data_executor import DataExecutor
from tensorflow_federated.python.core.impl.executors.eager_tf_executor import EagerTFExecutor
from tensorflow_federated.python.core.impl.executors.executor_base import Executor
from tensorflow_federated.python.core.impl.executors.executor_factory import CardinalitiesType
from tensorflow_federated.python.core.impl.executors.executor_factory import ExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_serialization import deserialize_value
from tensorflow_federated.python.core.impl.executors.executor_serialization import serialize_value
from tensorflow_federated.python.core.impl.executors.executor_service import ExecutorService
from tensorflow_federated.python.core.impl.executors.executor_stacks import ComposingExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_stacks import local_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import ReconstructOnChangeExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_stacks import remote_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import remote_executor_factory_from_stubs
from tensorflow_federated.python.core.impl.executors.executor_stacks import ResourceManagingExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_stacks import SizeInfo
from tensorflow_federated.python.core.impl.executors.executor_stacks import sizing_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import SizingExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_stacks import thread_debugging_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import UnplacedExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_value_base import ExecutorValue
from tensorflow_federated.python.core.impl.executors.federated_composing_strategy import FederatedComposingStrategy
from tensorflow_federated.python.core.impl.executors.federated_resolving_strategy import FederatedResolvingStrategy
from tensorflow_federated.python.core.impl.executors.federating_executor import FederatingExecutor
from tensorflow_federated.python.core.impl.executors.federating_executor import FederatingStrategy
from tensorflow_federated.python.core.impl.executors.ingestable_base import Ingestable
from tensorflow_federated.python.core.impl.executors.reference_resolving_executor import ReferenceResolvingExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor_stub import RemoteExecutorStub
from tensorflow_federated.python.core.impl.executors.thread_delegating_executor import ThreadDelegatingExecutor
from tensorflow_federated.python.core.impl.executors.transforming_executor import TransformingExecutor
from tensorflow_federated.python.core.impl.tensorflow_context.tensorflow_computation_context import get_input_sidechannel_filename as experimental_get_input_sidechannel_filename
from tensorflow_federated.python.core.impl.tensorflow_context.tensorflow_computation_context import get_output_sidechannel_filename as experimental_get_output_sidechannel_filename
from tensorflow_federated.python.core.impl.tensorflow_context.tensorflow_computation_context import get_session_token
from tensorflow_federated.python.core.impl.types.placements import PlacementLiteral
from tensorflow_federated.python.core.impl.types.type_conversions import type_from_tensors
from tensorflow_federated.python.core.impl.types.type_conversions import type_to_tf_tensor_specs
from tensorflow_federated.python.core.impl.types.type_serialization import deserialize_type
from tensorflow_federated.python.core.impl.types.type_serialization import serialize_type
from tensorflow_federated.python.core.impl.wrappers.computation_wrapper_instances import building_block_to_computation
