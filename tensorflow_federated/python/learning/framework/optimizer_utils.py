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
"""Common building blocks for federated optimization algorithms."""

import abc
import collections
from typing import Callable, List, Optional, Tuple, Union

from absl import logging
import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.tensorflow_libs import tensor_utils

# Type aliases.
_ModelConstructor = Callable[[], model_lib.Model]
_OptimizerConstructor = Union[optimizer_base.Optimizer,
                              Callable[[], tf.keras.optimizers.Optimizer]]


class ProcessTypeError(Exception):
  """Error raised when a `MeasuredProcess` does not have the correct type signature."""
  pass


# TODO(b/202027089): Revise this docstring once all models do not implement
# `report_local_outputs` and `federated_output_computation`.
@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    weights_delta: A dictionary of updates to the model's trainable variables.
    weights_delta_weight: Weight to use in a weighted mean when aggregating
      `weights_delta`.
    model_output: A structure matching `tff.learning.Model.report_local_outputs`
      (or `tff.learning.Model.report_local_unfinalized_metrics` if
      `report_local_outputs` is not implemented), reflecting the results of
      training on the input dataset.
    optimizer_output: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib(default=None)


class ClientDeltaFn(object, metaclass=abc.ABCMeta):
  """Represents a client computation that produces an update to a model."""

  @abc.abstractproperty
  def variables(self):
    """Returns all the variables of this object.

    Note this only includes variables that are part of the state of this object,
    and not the model variables themselves.

    Returns:
      An iterable of `tf.Variable` objects.
    """
    pass

  @abc.abstractmethod
  def __call__(self, dataset, initial_weights):
    """Defines the complete client computation.

    Typically implementations should be decorated with `tf.function`.

    Args:
      dataset: A `tf.data.Dataset` producing batches than can be fed to
        `tff.learning.Model.forward_pass`.
      initial_weights: A dictionary of initial values for all trainable and
        non-trainable model variables, keyed by name. This will be supplied by
        the server in Federated Averaging.

    Returns:
      An `optimizer_utils.ClientOutput` namedtuple.
    """
    pass


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Represents the state of the server carried between rounds.

  Attributes:
    model: A `ModelWeights` structure, containing Tensors or Variables.
    optimizer_state: A list of Tensors or Variables, in the order returned by
      `optimizer.variables()`
    delta_aggregate_state: State (possibly empty) of the delta_aggregate_fn.
    model_broadcast_state: State (possibly empty) of the model_broadcast_fn.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  delta_aggregate_state = attr.ib()
  model_broadcast_state = attr.ib()


def state_with_new_model_weights(
    server_state: ServerState,
    trainable_weights: List[np.ndarray],
    non_trainable_weights: List[np.ndarray],
) -> ServerState:
  """Returns a `ServerState` with updated model weights.

  Args:
    server_state: A server state object returned by an iterative training
      process like `tff.learning.build_federated_averaging_process`.
    trainable_weights: A list of `numpy` values in the order of the original
      model's `trainable_variables`.
    non_trainable_weights: A list of `numpy` values in the order of the original
      model's `non_trainable_variables`.

  Returns:
    A new server `ServerState` object which can be passed to the `next` method
    of the iterative process.
  """
  py_typecheck.check_type(server_state, ServerState)
  leaf_types = (int, float, np.ndarray, tf.Tensor, np.number)

  def assert_weight_lists_match(old_value, new_value):
    """Assert two flat lists of ndarrays or tensors match."""
    if isinstance(new_value, leaf_types) and isinstance(old_value, leaf_types):
      if (old_value.dtype != new_value.dtype or
          old_value.shape != new_value.shape):
        raise TypeError('Element is not the same tensor type. old '
                        f'({old_value.dtype}, {old_value.shape}) != '
                        f'new ({new_value.dtype}, {new_value.shape})')
    elif (isinstance(new_value, collections.abc.Sequence) and
          isinstance(old_value, collections.abc.Sequence)):
      if len(old_value) != len(new_value):
        raise TypeError('Model weights have different lengths: '
                        f'(old) {len(old_value)} != (new) {len(new_value)})\n'
                        f'Old values: {old_value}\nNew values: {new_value}')
      for old, new in zip(old_value, new_value):
        assert_weight_lists_match(old, new)
    else:
      raise TypeError('Model weights structures contains types that cannot be '
                      'handled.\nOld weights structure: {old}\n'
                      'New weights structure: {new}\n'
                      'Must be one of (int, float, np.ndarray, tf.Tensor, '
                      'collections.abc.Sequence)'.format(
                          old=tf.nest.map_structure(type, old_value),
                          new=tf.nest.map_structure(type, new_value)))

  assert_weight_lists_match(server_state.model.trainable, trainable_weights)
  assert_weight_lists_match(server_state.model.non_trainable,
                            non_trainable_weights)
  new_server_state = ServerState(
      model=model_utils.ModelWeights(
          trainable=trainable_weights, non_trainable=non_trainable_weights),
      optimizer_state=server_state.optimizer_state,
      delta_aggregate_state=server_state.delta_aggregate_state,
      model_broadcast_state=server_state.model_broadcast_state)
  return new_server_state


def _apply_delta(
    *,
    optimizer: tf.keras.optimizers.Optimizer,
    model_variables: model_utils.ModelWeights,
    delta,
) -> None:
  """Applies `delta` to `model` using `optimizer`."""
  tf.nest.assert_same_structure(delta, model_variables.trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(model_variables.trainable))
  # Note: this may create variables inside `optimizer`, for example if this is
  # the first usage of Adam or momentum optmizers.
  optimizer.apply_gradients(grads_and_vars)


# ==============================================================================
# Federated Computations
#
# These constructors setup the system level orchestration logic.
# ==============================================================================


def _build_initialize_computation(
    *,
    model_and_optimizer_init_fn: computation_base.Computation,
    broadcast_process: measured_process.MeasuredProcess,
    aggregation_process: measured_process.MeasuredProcess,
) -> computation_base.Computation:
  """Builds the `initialize` computation for a model delta averaging process.

  Args:
    model_and_optimizer_init_fn: A no-arg non-federated `tff.Computation` that
      returns initial model weights and initial state of the global optimizer.
    broadcast_process: A `tff.templates.MeasuredProcess` to broadcast the global
      model to the clients.
    aggregation_process: A `tff.templates.MeasuredProcess` to aggregate client
      model deltas.

  Returns:
    A `tff.Computation` that initializes the process. The computation takes no
    arguments and returns a `tuple` of global model weights and server state
    with `tff.SERVER` placement.
  """

  @computations.federated_computation()
  def initialize_computation():
    """Orchestration logic for server model initialization."""
    initial_global_model, initial_global_optimizer_state = intrinsics.federated_eval(
        model_and_optimizer_init_fn, placements.SERVER)
    return intrinsics.federated_zip(
        ServerState(
            model=initial_global_model,
            optimizer_state=initial_global_optimizer_state,
            delta_aggregate_state=aggregation_process.initialize(),
            model_broadcast_state=broadcast_process.initialize()))

  return initialize_computation


# TODO(b/202027089): Remove the note on `metrics_aggregator` once all models do
# not implement `report_local_outputs` and `federated_output_computation`.
def _build_one_round_computation(
    *,
    model_fn: _ModelConstructor,
    server_optimizer_fn: _OptimizerConstructor,
    model_to_client_delta_fn: Callable[[Callable[[], model_lib.Model]],
                                       ClientDeltaFn],
    broadcast_process: measured_process.MeasuredProcess,
    aggregation_process: measured_process.MeasuredProcess,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
) -> computation_base.Computation:
  """Builds the `next` computation for a model delta averaging process.

  Args:
    model_fn: A no-argument callable that constructs and returns a
      `tff.learning.Model`. *Must* construct and return a new model when called.
      Returning captured models from other scopes will raise errors.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-argument
      callable that constructs and returns a `tf.keras.optimizers.Optimizer`.
      The callable *must* construct and return a new Keras optimizer when
      called. Returning captured optimizers from other scopes will raise errors.
    model_to_client_delta_fn: A callable that takes a single no-arg callable
      that returns `tff.learning.Model` as an argument and returns a
      `ClientDeltaFn` which performs the local training loop and model delta
      computation.
    broadcast_process: A `tff.templates.MeasuredProcess` to broadcast the global
      model to the clients.
    aggregation_process: A `tff.templates.MeasuredProcess` to aggregate client
      model deltas.
    metrics_aggregator: An optional function that takes in the metric finalizers
      (i.e., `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a federated TFF computation of the following type signature
      `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`. If set,
      use the provided `metrics_aggregator`. If `None` and `model_fn` implements
      `federated_output_computation` and `report_local_outputs` (these two
      methods are deprecated and will be removed in 2022Q1), then
      `federated_output_computation` is used to aggregate the metrics. If `None`
      and `model_fn` does not implement `federated_output_computation` and
      `report_local_outputs`, uses `tff.learning.metrics.sum_then_finalize`,
      which returns a federated TFF computation that sums the unfinalized
      metrics from `CLIENTS`, and then applies the corresponding metric
      finalizers at `SERVER`.

  Returns:
    A `tff.Computation` that initializes the process. The computation takes
    a tuple of `(ServerState@SERVER, tf.data.Dataset@CLIENTS)` argument, and
    returns a tuple of `(ServerState@SERVER, metrics@SERVER)`.
  """
  # TODO(b/124477628): would be nice not to have the construct a throwaway model
  # here just to get the types. After fully moving to TF2.0 and eager-mode, we
  # should re-evaluate what happens here.
  # TODO(b/144382142): Keras name uniquification is probably the main reason we
  # still need this.
  with tf.Graph().as_default():
    whimsy_model_for_metadata = model_fn()
    model_weights = model_utils.ModelWeights.from_model(
        whimsy_model_for_metadata)
    model_weights_type = type_conversions.type_from_tensors(model_weights)

    optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        server_optimizer_fn,
        model_weights.trainable,
        disjoint_init_and_next=True)
    trainable_tensor_specs = tf.nest.map_structure(
        lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights.trainable)
    optimizer_state_type = type_conversions.type_from_tensors(
        optimizer.initialize(trainable_tensor_specs))
    # TODO(b/202027089): Remove this try/except logic once all models do not
    # implement `report_local_outputs` and `federated_output_computation`.
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        whimsy_model_for_metadata.report_local_unfinalized_metrics())

    if metrics_aggregator is not None:
      metrics_aggregation_computation = metrics_aggregator(
          whimsy_model_for_metadata.metric_finalizers(),
          unfinalized_metrics_type)
    else:
      try:
        metrics_aggregation_computation = (
            whimsy_model_for_metadata.federated_output_computation)
        logging.warning(
            'DeprecationWarning: `report_local_outputs` and '
            '`federated_output_computation` are deprecated and will be removed '
            'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
            '`metric_finalizers` instead. The cross-client metrics aggregation '
            'should be specified as the `metrics_aggregator` argument when you '
            'build a training process or evaluation computation.')
      except NotImplementedError:
        metrics_aggregation_computation = aggregator.sum_then_finalize(
            whimsy_model_for_metadata.metric_finalizers(),
            unfinalized_metrics_type)

  @computations.tf_computation(model_weights_type, model_weights_type.trainable,
                               optimizer_state_type)
  @tf.function
  def server_update(global_model, mean_model_delta, optimizer_state):
    """Updates the global model with the mean model update from clients."""
    with tf.init_scope():
      # Create a structure of variables that the server optimizer can update.
      model_variables = tf.nest.map_structure(
          lambda t: tf.Variable(initial_value=tf.zeros(t.shape, t.dtype)),
          global_model)
      optimizer = keras_optimizer.build_or_verify_tff_optimizer(
          server_optimizer_fn,
          model_variables.trainable,
          disjoint_init_and_next=True)

    # Set the variables to the current global model, the optimizer will
    # update these variables.
    tf.nest.map_structure(lambda a, b: a.assign(b), model_variables,
                          global_model)
    # We might have a NaN value e.g. if all of the clients processed had no
    # data, so the denominator in the federated_mean is zero. If we see any
    # NaNs, zero out the whole update.
    # TODO(b/124538167): We should increment a server counter to
    # track the fact a non-finite weights_delta was encountered.
    finite_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
        mean_model_delta)
    # Update the global model variables with the delta as a pseudo-gradient.
    negative_weights_delta = tf.nest.map_structure(lambda w: -1.0 * w,
                                                   finite_weights_delta)
    optimizer_state, updated_weights = optimizer.next(optimizer_state,
                                                      model_variables.trainable,
                                                      negative_weights_delta)
    # Keras optimizers mutate model variables in with the `next` step above, so
    # we skip calling the assignment for those optimizers.
    if not isinstance(optimizer, keras_optimizer.KerasOptimizer):
      tf.nest.map_structure(lambda a, b: a.assign(b), model_variables.trainable,
                            updated_weights)
    return model_variables, optimizer_state

  dataset_type = computation_types.SequenceType(
      whimsy_model_for_metadata.input_spec)

  @computations.tf_computation(dataset_type, model_weights_type)
  @tf.function
  def _compute_local_training_and_client_delta(dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      dataset: A `tf.data.Dataset` that provides training examples.
      initial_model_weights: A `model_utils.ModelWeights` containing the
        starting weights.

    Returns:
      A `ClientOutput` structure.
    """
    with tf.init_scope():
      client_delta_fn = model_to_client_delta_fn(model_fn)
    client_output = client_delta_fn(dataset, initial_model_weights)
    return client_output

  broadcast_state = broadcast_process.initialize.type_signature.result.member
  aggregation_state = aggregation_process.initialize.type_signature.result.member

  server_state_type = ServerState(
      model=model_weights_type,
      optimizer_state=optimizer_state_type,
      delta_aggregate_state=aggregation_state,
      model_broadcast_state=broadcast_state)

  @computations.federated_computation(
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(dataset_type, placements.CLIENTS))
  def one_round_computation(server_state, federated_dataset):
    """Orchestration logic for one round of optimization.

    Args:
      server_state: A `tff.learning.framework.ServerState` named tuple.
      federated_dataset: A federated `tf.Dataset` with placement tff.CLIENTS.

    Returns:
      A tuple of updated `tff.learning.framework.ServerState` and the result of
      `tff.learning.Model.federated_output_computation`, both having
      `tff.SERVER` placement.
    """
    broadcast_output = broadcast_process.next(
        server_state.model_broadcast_state, server_state.model)
    client_outputs = intrinsics.federated_map(
        _compute_local_training_and_client_delta,
        (federated_dataset, broadcast_output.result))
    if aggregation_process.is_weighted:
      aggregation_output = aggregation_process.next(
          server_state.delta_aggregate_state, client_outputs.weights_delta,
          client_outputs.weights_delta_weight)
    else:
      aggregation_output = aggregation_process.next(
          server_state.delta_aggregate_state, client_outputs.weights_delta)
    new_global_model, new_optimizer_state = intrinsics.federated_map(
        server_update, (server_state.model, aggregation_output.result,
                        server_state.optimizer_state))
    new_server_state = intrinsics.federated_zip(
        ServerState(new_global_model, new_optimizer_state,
                    aggregation_output.state, broadcast_output.state))
    aggregated_outputs = metrics_aggregation_computation(
        client_outputs.model_output)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            broadcast=broadcast_output.measurements,
            aggregation=aggregation_output.measurements,
            train=aggregated_outputs))
    return new_server_state, measurements

  return one_round_computation


def _is_valid_stateful_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates whether a `MeasuredProcess` is valid for model delta processes.

  Valid processes must have `state` and `measurements` placed on the server.
  This method is intended to be used with additional validation on the non-state
  parameters, inputs and result.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff `process` is a valid stateful process, `False` otherwise.
  """
  init_type = process.initialize.type_signature
  next_type = process.next.type_signature
  return (init_type.result.placement is placements.SERVER and
          next_type.parameter[0].placement is placements.SERVER and
          next_type.result.state.placement is placements.SERVER and
          next_type.result.measurements.placement is placements.SERVER)


def is_valid_broadcast_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates a `MeasuredProcess` adheres to the broadcast signature.

  A valid broadcast process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff the process is a validate broadcast process, otherwise `False`.
  """
  next_type = process.next.type_signature
  return (isinstance(process, measured_process.MeasuredProcess) and
          _is_valid_stateful_process(process) and
          next_type.parameter[1].placement is placements.SERVER and
          next_type.result.result.placement is placements.CLIENTS)


# ============================================================================


@computations.federated_computation()
def _empty_server_initialization():
  return intrinsics.federated_value((), placements.SERVER)


def build_stateless_broadcaster(
    *, model_weights_type: Union[computation_types.StructType,
                                 computation_types.TensorType]
) -> measured_process.MeasuredProcess:
  """Builds a `MeasuredProcess` that wraps `tff.federated_broadcast`."""

  @computations.federated_computation(
      computation_types.FederatedType((), placements.SERVER),
      computation_types.FederatedType(model_weights_type, placements.SERVER),
  )
  def stateless_broadcast(state, value):
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        state=state,
        result=intrinsics.federated_broadcast(value),
        measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=_empty_server_initialization, next_fn=stateless_broadcast)


# TODO(b/202027089): Remove the note on `metrics_aggregator` once all models do
# not implement `report_local_outputs` and `federated_output_computation`.
def build_model_delta_optimizer_process(
    model_fn: _ModelConstructor,
    model_to_client_delta_fn: Callable[[Callable[[], model_lib.Model]],
                                       ClientDeltaFn],
    server_optimizer_fn: _OptimizerConstructor,
    *,
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[
        factory.AggregationFactory] = None,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
) -> iterative_process.IterativeProcess:
  """Constructs `tff.templates.IterativeProcess` for Federated Averaging or SGD.

  This provides the TFF orchestration logic connecting the common server logic
  which applies aggregated model deltas to the server model with a
  `ClientDeltaFn` that specifies how `weight_deltas` are computed on device.

  Note: We pass in functions rather than constructed objects so we can ensure
  any variables or ops created in constructors are placed in the correct graph.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    model_to_client_delta_fn: A function from a `model_fn` to a `ClientDeltaFn`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer` or a no-arg
      function that returns a `tf.keras.optimizers.Optimizer`.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`. If set to default None,
      the server model is broadcast to the clients using the default
      tff.federated_broadcast.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` that contstructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses `tff.aggregators.MeanFactory`.
    metrics_aggregator: An optional function that takes in the metric finalizers
      (i.e., `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a federated TFF computation of the following type signature
      `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`. If set,
      use the provided `metrics_aggregator`. If `None` and `model_fn` implements
      `federated_output_computation` and `report_local_outputs` (these two
      methods are deprecated and will be removed in 2022Q1), then
      `federated_output_computation` is used to aggregate the metrics. If `None`
      and `model_fn` does not implement `federated_output_computation` and
      `report_local_outputs`, uses `tff.learning.metrics.sum_then_finalize`,
      which returns a federated TFF computation that sums the unfinalized
      metrics from `CLIENTS`, and then applies the corresponding metric
      finalizers at `SERVER`.

  Returns:
    A `tff.templates.IterativeProcess`.

  Raises:
    ProcessTypeError: If `broadcast_process` does not conform to the signature
      of broadcast (SERVER->CLIENTS).
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(model_to_client_delta_fn)

  @computations.tf_computation
  def model_and_optimizer_init_fn(
  ) -> Tuple[model_utils.ModelWeights, List[tf.Variable]]:
    """Returns initial model weights and state of the global optimizer."""
    model_variables = model_utils.ModelWeights.from_model(model_fn())
    optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        server_optimizer_fn,
        model_variables.trainable,
        disjoint_init_and_next=True)
    trainable_tensor_specs = tf.nest.map_structure(
        lambda v: tf.TensorSpec(v.shape, v.dtype), model_variables.trainable)
    optimizer_state = optimizer.initialize(trainable_tensor_specs)
    return model_variables, optimizer_state

  model_weights_type = model_and_optimizer_init_fn.type_signature.result[0]

  if broadcast_process is None:
    broadcast_process = build_stateless_broadcaster(
        model_weights_type=model_weights_type)
  if not is_valid_broadcast_process(broadcast_process):
    raise ProcessTypeError(
        'broadcast_process type signature does not conform to expected '
        'signature (<state@S, input@S> -> <state@S, result@C, measurements@S>).'
        ' Got: {t}'.format(t=broadcast_process.next.type_signature))

  if model_update_aggregation_factory is None:
    model_update_aggregation_factory = mean.MeanFactory()
  py_typecheck.check_type(model_update_aggregation_factory,
                          factory.AggregationFactory.__args__)
  if isinstance(model_update_aggregation_factory,
                factory.WeightedAggregationFactory):
    aggregation_process = model_update_aggregation_factory.create(
        model_weights_type.trainable, computation_types.TensorType(tf.float32))
  else:
    aggregation_process = model_update_aggregation_factory.create(
        model_weights_type.trainable)
  process_signature = aggregation_process.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`model_update_aggregation_factory` does not produce a '
                    'compatible `AggregationProcess`. The processes must '
                    'retain the type structure of the inputs on the '
                    f'server, but got {input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  initialize_computation = _build_initialize_computation(
      model_and_optimizer_init_fn=model_and_optimizer_init_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process)

  run_one_round_computation = _build_one_round_computation(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      model_to_client_delta_fn=model_to_client_delta_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process,
      metrics_aggregator=metrics_aggregator)

  return iterative_process.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round_computation)
