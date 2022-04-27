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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Utilities for serializing TensorFlow computations."""

from typing import Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import variable_utils


def tf_computation_serializer(parameter_type: Optional[computation_types.Type],
                              context_stack):
  """Serializes a TF computation with a given parameter type.

  Args:
    parameter_type: The parameter type specification if the target accepts a
      parameter, or `None` if the target doesn't declare any parameters. Either
      an instance of `computation_types.Type`.
    context_stack: The context stack to use.

  Yields:
    The first yielded value will be a Python object (such as a dataset,
    a placeholder, or a `structure.Struct`) to be passed to the function to
    serialize. The result of the function should then be passed to the
    following `send` call.
    The next yielded value will be
    a tuple of (`pb.Computation`, `tff.Type`), where the computation contains
    the instance with the `pb.TensorFlow` variant set, and the type is an
    instance of `tff.Type`, potentially including Python container annotations,
    for use by TensorFlow computation wrappers.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the signature of the target is not compatible with the given
      parameter type.
  """
  # TODO(b/113112108): Support a greater variety of target type signatures,
  # with keyword args or multiple args corresponding to elements of a tuple.
  # Document all accepted forms with examples in the API, and point to there
  # from here.

  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if parameter_type is not None:
    py_typecheck.check_type(parameter_type, computation_types.Type)

  with tf.Graph().as_default() as graph:
    session_token_tensor = tf.compat.v1.placeholder(
        tf.string, shape=(), name='session_token_tensor')
    input_sidechannel_filename_tensor = tf.compat.v1.placeholder(
        tf.string, shape=(), name='input_sidechannel_filename_tensor')
    output_sidechannel_filename_tensor = tf.compat.v1.placeholder(
        tf.string, shape=(), name='output_sidechannel_filename_tensor')
    if parameter_type is not None:
      parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
          'arg', parameter_type, graph)
    else:
      parameter_value = None
      parameter_binding = None
    context = tensorflow_computation_context.TensorFlowComputationContext(
        graph, session_token_tensor, input_sidechannel_filename_tensor,
        output_sidechannel_filename_tensor)
    with context_stack.install(context):
      with variable_utils.record_variable_creation_scope() as all_variables:
        result = yield parameter_value
      initializer_ops = []
      if all_variables:
        # Use a readable but not-too-long name for the init_op.
        name = 'init_op_for_' + '_'.join(
            [v.name.replace(':0', '') for v in all_variables])
        if len(name) > 50:
          name = 'init_op_for_{}_variables'.format(len(all_variables))
        initializer_ops.append(
            tf.compat.v1.initializers.variables(all_variables, name=name))
      initializer_ops.extend(
          tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TABLE_INITIALIZERS))
      if initializer_ops:
        # Before running the main new init op, run any initializers for sub-
        # computations from context.init_ops. Variables from import_graph_def
        # will not make it into the global collections, and so will not be
        # initialized without this code path.
        with tf.compat.v1.control_dependencies(context.init_ops):
          init_op_name = tf.group(
              *initializer_ops, name='grouped_initializers').name
      elif context.init_ops:
        init_op_name = tf.group(
            *context.init_ops, name='subcomputation_init_ops').name
      else:
        init_op_name = None

    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)

  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
      session_token_tensor_name=session_token_tensor.name,
      initialize_op=init_op_name,
      experimental=pb.TensorFlow.Experimental(
          input_sidechannel_filename_tensor_name=input_sidechannel_filename_tensor
          .name,
          output_sidechannel_filename_tensor_name=output_sidechannel_filename_tensor
          .name,
          appends_to_output_sidechannel=context.appends_to_output_sidechannel))
  yield pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow), type_signature
