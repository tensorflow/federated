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
"""An implementation of the Federated Averaging algorithm.

The original Federated Averaging algorithm is proposed by the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

This file implements a generalized version of the Federated Averaging algorithm:

Adaptive Federated Optimization
    Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
    Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan. ICLR 2021.
    https://arxiv.org/abs/2003.00295
"""

from typing import Callable, Optional, Union

from absl import logging
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.tensorflow_libs import tensor_utils


# TODO(b/202027089): Revise the following docstring once all models do not
# implement `report_local_outputs` and `federated_output_computation`.
class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(
      self,
      model: model_lib.Model,
      optimizer: Union[optimizer_base.Optimizer,
                       Callable[[], tf.keras.optimizers.Optimizer]],
      client_weighting: client_weight_lib.ClientWeightType = client_weight_lib
      .ClientWeighting.NUM_EXAMPLES,
      use_experimental_simulation_loop: bool = False,
      user_provide_metrics_aggregator: bool = False):
    """Creates the client computation for Federated Averaging.

    Note: All variable creation required for the client computation (e.g. model
    variable creation) must occur in during construction, and not during
    `__call__`.

    Args:
      model: A `tff.learning.Model` instance.
      optimizer: A `optimizer_base.Optimizer` instance, or a no-arg callable
        that returns a `tf.keras.Optimizer` instance..
      client_weighting: A value of `tff.learning.ClientWeighting` that specifies
        a built-in weighting method, or a callable that takes the model output
        and returns a tensor that provides the weight in the federated average
        of model deltas.
      use_experimental_simulation_loop: Controls the reduce loop function for
        input dataset. An experimental reduce loop is used for simulation.
      user_provide_metrics_aggregator: If True, the model output is given by
        `model.report_local_unfinalized_metrics()`. If False, and if
        `model.report_local_outputs` is implemented (this method is deprecated
        and will be removed in 2022Q1), the model output is given by
        `model.report_local_outputs()`; otherwise, the model output is given by
        `model.report_local_unfinalized_metrics()`.
    """
    py_typecheck.check_type(model, model_lib.Model)
    self._model = model
    self._optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        optimizer,
        model_utils.ModelWeights.from_model(self._model).trainable,
        disjoint_init_and_next=False)
    client_weight_lib.check_is_client_weighting_or_callable(client_weighting)
    self._client_weighting = client_weighting
    self._dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    self._user_provide_metrics_aggregator = user_provide_metrics_aggregator

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    model = self._model
    optimizer = self._optimizer
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          initial_weights)

    def reduce_fn(state, batch):
      """Train `tff.learning.Model` on local client batch."""
      num_examples_sum, optimizer_state = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      optimizer_state, updated_weights = optimizer.next(optimizer_state,
                                                        model_weights.trainable,
                                                        gradients)
      if not isinstance(optimizer, keras_optimizer.KerasOptimizer):
        # Keras optimizer mutates model variables within the `next` step.
        tf.nest.map_structure(lambda a, b: a.assign(b), model_weights.trainable,
                              updated_weights)

      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum, optimizer_state

    def initial_state_for_reduce_fn():
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights.trainable)
      return tf.zeros(
          shape=[],
          dtype=tf.int64), optimizer.initialize(trainable_tensor_specs)

    num_examples_sum, _ = self._dataset_reduce_fn(
        reduce_fn, dataset, initial_state_fn=initial_state_for_reduce_fn)

    weights_delta = tf.nest.map_structure(tf.subtract, model_weights.trainable,
                                          initial_weights.trainable)
    # TODO(b/202027089): Remove this try/except logic once all models do not
    # implement `report_local_outputs` and `federated_output_computation`.
    if self._user_provide_metrics_aggregator:
      model_output = model.report_local_unfinalized_metrics()
    else:
      try:
        model_output = model.report_local_outputs()
        logging.warning(
            'DeprecationWarning: `report_local_outputs` and '
            '`federated_output_computation` are deprecated and will be removed '
            'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
            '`metric_finalizers` instead. The cross-client metrics aggregation '
            'should be specified as the `metrics_aggregator` argument when you '
            'build a training process or evaluation computation.')
      except NotImplementedError:
        model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      # TODO(b/176171842): Zeroing has no effect with unweighted aggregation.
      weights_delta_weight = tf.constant(0.0)
    elif self._client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    elif self._client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      weights_delta_weight = tf.constant(1.0)
    else:
      weights_delta_weight = self._client_weighting(model_output)
    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model_output, optimizer_output=())


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


# TODO(b/202027089): Remove the note on `metrics_aggregator` once all models do
# not implement `report_local_outputs` and `federated_output_computation`.
def build_federated_averaging_process(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Union[optimizer_base.Optimizer,
                               Callable[[], tf.keras.optimizers.Optimizer]],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    *,  # Require named (non-positional) parameters for the following kwargs:
    client_weighting: Optional[client_weight_lib.ClientWeightType] = None,
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[
        factory.AggregationFactory] = None,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
    use_experimental_simulation_loop: bool = False
) -> iterative_process.IterativeProcess:
  """Builds an iterative process that performs federated averaging.

  This function creates a `tff.templates.IterativeProcess` that performs
  federated averaging on client models. The iterative process has the following
  methods inherited from `tff.templates.IterativeProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `tff.learning.framework.ServerState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `tff.learning.framework.ServerState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `tff.learning.framework.ServerState` representing the updated server state
      and aggregated metrics at the server, including client training metrics
      and any other metrics from broadcast and aggregation processes.

  The iterative process also has the following method not inherited from
  `tff.templates.IterativeProcess`:

  *   `get_model_weights`: A `tff.Computation` that takes as input the
      a `tff.learning.framework.ServerState`, and returns a
      `tff.learning.ModelWeights` containing the state's model weights.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, local training on one
  pass of the pre-processed client dataset (multiple epochs are possible if the
  dataset is pre-processed with `repeat` operation) is performed via the
  `tf.keras.optimizers.Optimizer.apply_gradients` method of the client
  optimizer. Each client computes the difference between the client model after
  training and the initial broadcast model. These model deltas are then
  aggregated at the server using some aggregation function. The aggregate model
  delta is applied at the server by using the
  `tf.keras.optimizers.Optimizer.apply_gradients` method of the server
  optimizer.

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedAvg algorithm in
  [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers (this generalized FedAvg algorithm is described in
  [Reddi et al., 2021](https://arxiv.org/abs/2003.00295)).

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If None, defaults to weighting
      by number of examples.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`. If set to default None,
      the server model is broadcast to the clients using the default
      tff.federated_broadcast.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` that constructs
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
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  if isinstance(model_update_aggregation_factory,
                factory.UnweightedAggregationFactory):
    if client_weighting is None:
      client_weighting = client_weight_lib.ClientWeighting.UNIFORM
    elif client_weighting is not client_weight_lib.ClientWeighting.UNIFORM:
      raise ValueError('Cannot use non-uniform client weighting with '
                       'unweighted aggregation.')
  elif client_weighting is None:
    client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES

  user_provide_metrics_aggregator = metrics_aggregator is not None

  def client_fed_avg(model_fn: Callable[[], model_lib.Model]) -> ClientFedAvg:
    return ClientFedAvg(model_fn(), client_optimizer_fn, client_weighting,
                        use_experimental_simulation_loop,
                        user_provide_metrics_aggregator)

  iter_proc = optimizer_utils.build_model_delta_optimizer_process(
      model_fn,
      model_to_client_delta_fn=client_fed_avg,
      server_optimizer_fn=server_optimizer_fn,
      broadcast_process=broadcast_process,
      model_update_aggregation_factory=model_update_aggregation_factory,
      metrics_aggregator=metrics_aggregator)

  server_state_type = iter_proc.state_type.member

  @computations.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iter_proc.get_model_weights = get_model_weights
  return iter_proc
