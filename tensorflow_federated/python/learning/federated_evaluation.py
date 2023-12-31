# Copyright 2019, The TensorFlow Federated Authors.
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
"""A simple implementation of federated evaluation."""

import collections
from collections.abc import Callable, Mapping
from typing import Optional, Union

import tensorflow as tf
import typing_extensions

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable

# Convenience aliases.
_SequenceType = computation_types.SequenceType
_MetricsAggregatorFirstArgType = Union[
    types.MetricFinalizersType, types.FunctionalMetricFinalizersType
]
_MetricsAggregator = Callable[
    [_MetricsAggregatorFirstArgType, computation_types.StructWithPythonType],
    computation_base.Computation,
]


def build_local_evaluation(
    model_fn: Callable[[], variable.VariableModel],
    model_weights_type: computation_types.StructType,
    batch_type: computation_types.Type,
    use_experimental_simulation_loop: bool = False,
) -> computation_base.Computation:
  """Builds the local TFF computation for evaluation of the given model.

  This produces an unplaced function that evaluates a
  `tff.learning.models.VariableModel`
  on a `tf.data.Dataset`. This function can be mapped to placed data, i.e.
  is mapped to client placed data in `build_federated_evaluation`.

  The TFF type notation for the returned computation is:

  ```
  (<M, D*> → <local_outputs=N, num_examples=tf.int64>)
  ```

  Where `M` is the model weights type structure, `D` is the type structure of a
  single data point, and `N` is the type structure of the local metrics.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`.
    model_weights_type: The `tff.Type` of the model parameters that will be used
      to initialize the model during evaluation.
    batch_type: The type of one entry in the dataset.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and sequential data, and returns the evaluation metrics.
  """

  @tensorflow_computation.tf_computation(
      model_weights_type, _SequenceType(batch_type)
  )
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluating `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda v, t: v.assign(t), model_weights, incoming_model_weights
    )

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(batch, training=False)
      if model_output.num_examples is None:
        # Compute shape from the size of the predictions if model didn't use the
        # batch size.
        return (
            num_examples
            + tf.shape(model_output.predictions, out_type=tf.int64)[0]
        )
      else:
        return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop
    )
    num_examples = dataset_reduce_fn(
        reduce_fn, dataset, lambda: tf.zeros([], dtype=tf.int64)
    )
    model_output = model.report_local_unfinalized_metrics()
    return collections.OrderedDict(
        local_outputs=model_output, num_examples=num_examples
    )

  return client_eval


def build_functional_local_evaluation(
    model: functional.FunctionalModel,
    model_weights_type: computation_types.StructType,
    batch_type: Union[
        computation_types.StructType, computation_types.TensorType
    ],
) -> computation_base.Computation:
  """Creates client evaluation logic for a functional model.

  This produces an unplaced function that evaluates a
  `tff.learning.models.FunctionalModel` on a `tf.data.Dataset`. This function
  can be mapped to placed data.

  The TFF type notation for the returned computation is:

  ```
  (<M, D*> → <local_outputs=N>)
  ```

  Where `M` is the model weights type structure, `D` is the type structure of a
  single data point, and `N` is the type structure of the local metrics.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    model_weights_type: The `tff.Type` of the model parameters that will be used
      in the forward pass.
    batch_type: The type of one entry in the dataset.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and sequential data, and returns the evaluation metrics.
  """

  @tensorflow_computation.tf_computation(
      model_weights_type, _SequenceType(batch_type)
  )
  @tf.function
  def local_eval(weights, dataset):
    metrics_state = model.initialize_metrics_state()

    for batch in iter(dataset):
      if isinstance(batch, Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch

      batch_output = model.predict_on_batch(weights, x, training=False)
      batch_loss = model.loss(output=batch_output, label=y)
      predictions = tf.nest.flatten(batch_output)[0]
      batch_num_examples = tf.shape(predictions)[0]

      # TODO: b/272099796 - Update `update_metrics_state` of FunctionalModel
      metrics_state = model.update_metrics_state(
          metrics_state,
          batch_output=variable.BatchOutput(
              loss=batch_loss,
              predictions=batch_output,
              num_examples=batch_num_examples,
          ),
          labels=y,
      )

    unfinalized_metrics = metrics_state
    return unfinalized_metrics

  return local_eval


@typing_extensions.deprecated(
    '`tff.learning.build_federated_evaluation` is deprecated, use '
    '`tff.learning.algorithms.build_fed_eval` instead.'
)
def build_federated_evaluation(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    metrics_aggregator: Optional[_MetricsAggregator] = None,
    use_experimental_simulation_loop: bool = False,
) -> computation_base.Computation:
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENTS)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default tff.federated_broadcast.
    metrics_aggregator: An optional function that takes in the metric finalizers
      (i.e., `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a federated TFF computation of the following type signature
      `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`. If
      `None`, uses `tff.learning.metrics.sum_then_finalize`, which returns a
      federated TFF computation that sums the unfinalized metrics from
      `CLIENTS`, and then applies the corresponding metric finalizers at
      `SERVER`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics.
  """
  if not callable(model_fn):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance '
          f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
      )

  if broadcast_process is not None:
    if not isinstance(broadcast_process, measured_process.MeasuredProcess):
      raise ValueError(
          '`broadcast_process` must be a `MeasuredProcess`, got '
          f'{type(broadcast_process)}.'
      )
    if iterative_process.is_stateful(broadcast_process):
      raise ValueError(
          'Cannot create a federated evaluation with a stateful '
          'broadcast process, must be stateless (have empty state), has state: '
          f'{broadcast_process.initialize.type_signature.result!r}'
      )

  if metrics_aggregator is None:
    metrics_aggregator = aggregator.sum_then_finalize

  if not callable(model_fn):
    return _build_functional_federated_evaluation(
        model=model_fn,
        broadcast_process=broadcast_process,
        metrics_aggregator=metrics_aggregator,
    )
  else:
    return _build_federated_evaluation(
        model_fn=model_fn,
        broadcast_process=broadcast_process,
        metrics_aggregator=metrics_aggregator,
        use_experimental_simulation_loop=use_experimental_simulation_loop,
    )


def _build_federated_evaluation(
    *,
    model_fn: Callable[[], variable.VariableModel],
    broadcast_process: Optional[measured_process.MeasuredProcess],
    metrics_aggregator: _MetricsAggregator,
    use_experimental_simulation_loop: bool,
) -> computation_base.Computation:
  """Builds a federated evaluation computation for a `tff.learning.models.VariableModel`."""
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO: b/124477628 - Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_weights_lib.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)

    unfinalized_metrics_type = type_conversions.infer_type(
        model.report_local_unfinalized_metrics()
    )
    metrics_aggregation_computation = metrics_aggregator(
        model.metric_finalizers(),
        unfinalized_metrics_type,  # pytype: disable=wrong-arg-types
    )

  local_eval = build_local_evaluation(
      model_fn=model_fn,
      model_weights_type=model_weights_type,
      batch_type=batch_type,
      use_experimental_simulation_loop=use_experimental_simulation_loop,
  )

  @federated_computation.federated_computation(
      computation_types.at_server(model_weights_type),
      computation_types.at_clients(_SequenceType(batch_type)),
  )
  def server_eval(server_model_weights, federated_dataset):
    if broadcast_process is not None:
      # TODO: b/179091838 - Zip the measurements from the broadcast_process with
      # the result of `model_metrics` below to avoid dropping these metrics.
      broadcast_output = broadcast_process.next(
          broadcast_process.initialize(), server_model_weights
      )
      client_outputs = intrinsics.federated_map(
          local_eval, (broadcast_output.result, federated_dataset)
      )
    else:
      client_outputs = intrinsics.federated_map(
          local_eval,
          [
              intrinsics.federated_broadcast(server_model_weights),
              federated_dataset,
          ],
      )
    model_metrics = metrics_aggregation_computation(
        client_outputs.local_outputs
    )
    return intrinsics.federated_zip(collections.OrderedDict(eval=model_metrics))

  return server_eval


def _build_functional_federated_evaluation(
    *,
    model: functional.FunctionalModel,
    broadcast_process: Optional[measured_process.MeasuredProcess],
    metrics_aggregator: _MetricsAggregator,
) -> computation_base.Computation:
  """Builds a federated evaluation computation for a functional model."""

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(
        shape=ndarray.shape, dtype=tf.dtypes.as_dtype(ndarray.dtype)
    )

  weights_type = tf.nest.map_structure(
      ndarray_to_tensorspec, model.initial_weights
  )
  batch_type = computation_types.to_type(model.input_spec)
  local_eval = build_functional_local_evaluation(
      model,
      weights_type,
      batch_type,  # pytype: disable=wrong-arg-types
  )

  @federated_computation.federated_computation(
      computation_types.at_server(weights_type),
      computation_types.at_clients(_SequenceType(batch_type)),
  )
  def federated_eval(server_weights, client_data):
    if broadcast_process is not None:
      # TODO: b/179091838 - Zip the measurements from the broadcast_process with
      # the result of `model_metrics` below to avoid dropping these metrics.
      broadcast_output = broadcast_process.next(
          broadcast_process.initialize(), server_weights
      )
      client_weights = broadcast_output.result
    else:
      client_weights = intrinsics.federated_broadcast(server_weights)
    unfinalized_metrics = intrinsics.federated_map(
        local_eval, (client_weights, client_data)
    )
    metrics_aggregation_fn = metrics_aggregator(
        model.finalize_metrics,
        unfinalized_metrics.type_signature.member,  # pytype: disable=attribute-error
    )
    finalized_metrics = metrics_aggregation_fn(unfinalized_metrics)
    return intrinsics.federated_zip(
        collections.OrderedDict(eval=finalized_metrics)
    )

  return federated_eval
