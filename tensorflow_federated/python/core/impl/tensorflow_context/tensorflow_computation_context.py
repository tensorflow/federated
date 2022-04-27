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
"""The implementation of a context to use in building TF computations."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class NotInTensorFlowComputationContextError(context_base.ContextError):

  def __init__(self, attempted_to_retrieve: str,
               actual_context: context_base.Context):
    super().__init__(
        f'{attempted_to_retrieve} can only be retrieved from within the '
        '`TensorFlowComputationContext (in a `@tff.tf_computation`). '
        f'Instead, the context {actual_context} of type {type(actual_context)} '
        'was found.')


def expect_tfc_context(
    attempted_to_retrieve: str) -> 'TensorFlowComputationContext':
  context = context_stack_impl.context_stack.current
  if not isinstance(context, TensorFlowComputationContext):
    raise NotInTensorFlowComputationContextError(attempted_to_retrieve, context)
  return context


def get_session_token() -> tf.Tensor:
  """Returns a string tensor identifying the current session."""
  return expect_tfc_context('Session tokens').session_token


def get_input_sidechannel_filename() -> tf.Tensor:
  """Returns a string tensor containing a filename for sidechannel input."""
  return expect_tfc_context(
      'Input sidechannel filename').input_sidechannel_filename


def get_output_sidechannel_filename() -> tf.Tensor:
  """Returns a string tensor containing a filename for sidechannel output."""
  return expect_tfc_context(
      'Output sidechannel filename').output_sidechannel_filename


def set_appends_to_output_sidechannel():
  """Declare that the current context appends to the sidechannel output."""
  expect_tfc_context(
      '`appends_to_output_sidechannel`').set_appends_to_output_sidechannel()


class TensorFlowComputationContext(context_base.Context):
  """The context for building TensorFlow computations."""

  def __init__(self, graph, session_token, input_sidechannel_filename,
               output_sidechannel_filename):
    py_typecheck.check_type(graph, tf.Graph)
    self._graph = graph
    self._init_ops = []
    self._shared_name_index = 0
    self._session_token = session_token
    self._input_sidechannel_filename = input_sidechannel_filename
    self._output_sidechannel_filename = output_sidechannel_filename
    self._appends_to_output_sidechannel = False

  @property
  def init_ops(self):
    """Returns a list of init ops for computations invoked in this context."""
    return list(self._init_ops)

  @property
  def session_token(self):
    """Returns a string tensor which uniquely identifies the current session."""
    return self._session_token

  @property
  def input_sidechannel_filename(self):
    """Returns a string tensor containing a filename for sidechannel input."""
    return self._input_sidechannel_filename

  @property
  def output_sidechannel_filename(self):
    """Returns a string tensor containing a filename for sidechannel output."""
    return self._output_sidechannel_filename

  def set_appends_to_output_sidechannel(self):
    """Declare that the current context appends to the output sidechannel."""
    self._appends_to_output_sidechannel = True

  @property
  def appends_to_output_sidechannel(self):
    """Returns whether the output sidechannel has been appended to."""
    return self._appends_to_output_sidechannel

  def invoke(self, comp: computation_impl.ConcreteComputation, arg):
    if arg is not None:
      type_analysis.check_type(arg, comp.type_signature.parameter)
    # We are invoking a tff.tf_computation inside of another
    # tf_computation.
    py_typecheck.check_type(comp, computation_impl.ConcreteComputation)
    computation_proto = computation_impl.ConcreteComputation.get_proto(comp)
    computation_oneof = computation_proto.WhichOneof('computation')
    if computation_oneof != 'tensorflow':
      raise ValueError(
          'Can only invoke TensorFlow in the body of a TensorFlow '
          'computation; got computation of type {}'.format(computation_oneof))
    shared_name_suffix = f'_tffshared_{self._shared_name_index}'
    self._shared_name_index += 1
    call_result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto, arg, self._graph, shared_name_suffix,
        self.session_token, self.input_sidechannel_filename,
        self.output_sidechannel_filename)
    if call_result.appended_to_output_sidechannel:
      self.set_appends_to_output_sidechannel()
    if call_result.init_op:
      self._init_ops.append(call_result.init_op)
    return type_conversions.type_to_py_container(call_result.result,
                                                 comp.type_signature.result)
