# Copyright 2020, The TensorFlow Federated Authors.
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
"""Utilities for constructing reconstruction models from Keras models."""

import collections
from typing import Iterable, List

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import keras_utils as base_utils
from tensorflow_federated.python.learning.reconstruction import model as model_lib


def from_keras_model(
    keras_model: tf.keras.Model,
    *,  # Caller passes below args by name.
    global_layers: Iterable[tf.keras.layers.Layer],
    local_layers: Iterable[tf.keras.layers.Layer],
    input_spec,
) -> model_lib.Model:
  """Builds a `tff.learning.reconstruction.Model` from a `tf.keras.Model`.

  The `tff.learning.reconstruction.Model` returned by this function uses
  `keras_model` for its forward pass and autodifferentiation steps. During
  reconstruction, variables in `local_layers` are initialized and trained.
  Post-reconstruction, variables in `global_layers` are trained and aggregated
  on the server. All variables must be partitioned between global and local
  layers, without overlap.

  Args:
    keras_model: A `tf.keras.Model` object that is not compiled.
    global_layers: Iterable of global layers to be aggregated across users. All
      trainable and non-trainable model variables that can be aggregated on the
      server should be included in these layers.
    local_layers: Iterable of local layers not shared with the server. All
      trainable and non-trainable model variables that should not be aggregated
      on the server should be included in these layers.
    input_spec: A structure of `tf.TensorSpec`s specifying the type of arguments
      the model expects. Notice this must be a compound structure of two
      elements, specifying both the data fed into the model to generate
      predictions, as its first element, as well as the expected type of the
      ground truth as its second.

  Returns:
    A `tff.learning.reconstruction.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
    ValueError: If `keras_model` was compiled.
  """
  if not isinstance(keras_model, tf.keras.Model):
    raise TypeError('Expected `int`, found {}.'.format(type(keras_model)))
  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model.')

  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  return _KerasModel(
      inner_model=keras_model,
      global_layers=global_layers,
      local_layers=local_layers,
      input_spec=input_spec)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for `tf.keras.Model` objects.

  Wraps uncompiled Keras models as `tff.learning.reconstruction.Model`s.
  Tracks global and local layers of the model. Parameters contained in global
  layers are sent to the server and aggregated across users normally, and
  parameters contained in local layers are reconstructed at the beginning of
  each round and not sent to the server. The loss function and metrics are
  passed to a `tff.templates.IterativeProcess` wrapping this model and computed
  there for both training and evaluation.
  """

  def __init__(self, inner_model: tf.keras.Model,
               global_layers: Iterable[tf.keras.layers.Layer],
               local_layers: Iterable[tf.keras.layers.Layer],
               input_spec: computation_types.Type):
    self._keras_model = inner_model
    self._global_layers = list(global_layers)
    self._local_layers = list(local_layers)
    self._input_spec = input_spec

    # Ensure global_layers and local_layers include exactly the Keras model's
    # trainable and non-trainable variables. Use hashable refs to uniquely
    # compare variables, and track variable names for informative error
    # messages.
    global_and_local_variables = set()
    for layer in self._global_layers + self._local_layers:
      global_and_local_variables.update(
          (var.ref(), var.name)
          for var in layer.trainable_variables + layer.non_trainable_variables)

    keras_variables = set((var.ref(), var.name)
                          for var in inner_model.trainable_variables +
                          inner_model.non_trainable_variables)

    if global_and_local_variables != keras_variables:
      # Use a symmetric set difference to compare the variables, since either
      # set may include variables not present in the other.
      variables_difference = global_and_local_variables ^ keras_variables
      raise ValueError('Global and local layers must include all trainable '
                       'and non-trainable variables in the Keras model. '
                       'Difference: {d}, Global and local layers vars: {v}, '
                       'Keras vars: {k}'.format(
                           d=variables_difference,
                           v=global_and_local_variables,
                           k=keras_variables))

  @property
  def global_trainable_variables(self):
    variables = []
    for layer in self._global_layers:
      variables.extend(layer.trainable_variables)
    return variables

  @property
  def global_non_trainable_variables(self):
    variables = []
    for layer in self._global_layers:
      variables.extend(layer.non_trainable_variables)
    return variables

  @property
  def local_trainable_variables(self):
    variables = []
    for layer in self._local_layers:
      variables.extend(layer.trainable_variables)
    return variables

  @property
  def local_non_trainable_variables(self):
    variables = []
    for layer in self._local_layers:
      variables.extend(layer.non_trainable_variables)
    return variables

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def forward_pass(self, batch_input, training=True):
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()
    if isinstance(batch_input, collections.abc.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     'Instead have keys {}'.format(list(batch_input.keys())))
    predictions = self._keras_model(inputs, training=training)

    if isinstance(batch_input, collections.abc.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]

    return model_lib.BatchOutput(
        predictions=predictions,
        labels=y_true,
        num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0])


class MeanLossMetric(tf.keras.metrics.Mean):
  """A `tf.keras.metrics.Metric` wrapper for a loss function.

  The loss function can be a `tf.keras.losses.Loss`, or it can be any callable
  with the signature loss(y_true, y_pred).

  Note that the dependence on a passed-in loss function may cause issues with
  serialization of this metric.
  """

  def __init__(self, loss_fn, name='loss', dtype=tf.float32):
    super().__init__(name, dtype)
    self._loss_fn = loss_fn

  def update_state(self, y_true, y_pred, sample_weight=None):
    batch_size = tf.cast(tf.shape(y_pred)[0], self._dtype)
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)
    batch_loss = self._loss_fn(y_true, y_pred)

    return super().update_state(batch_loss, batch_size)

  def get_config(self):
    """Used to recreate an instance of this class during aggregation."""
    config = {'loss_fn': self._loss_fn}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def read_metric_variables(
    metrics: List[tf.keras.metrics.Metric]) -> collections.OrderedDict:
  """Reads values from Keras metric variables."""
  metric_variables = collections.OrderedDict()
  for metric in metrics:
    metric_variables[metric.name] = [v.read_value() for v in metric.variables]
  return metric_variables


def federated_output_computation_from_metrics(
    metrics: List[tf.keras.metrics.Metric]
) -> computations.federated_computation:
  """Produces a federated computation for aggregating Keras metrics.

  This can be used to evaluate both Keras and non-Keras models using Keras
  metrics. Aggregates metrics across clients by summing their internal
  variables, producing new metrics with summed internal variables, and calling
  metric.result() on each. See `tff.learning.federated_aggregate_keras_metric`
  for details.

  Args:
    metrics: A List of `tf.keras.metrics.Metric` to aggregate.

  Returns:
    A `tff.federated_computation` aggregating metrics across clients by summing
    their internal variables, producing new metrics with summed internal
    variables, and calling metric.result() on each.
  """
  # Get a sample of metric variables to use to determine its type.
  sample_metric_variables = read_metric_variables(metrics)

  metric_variable_type_dict = tf.nest.map_structure(tf.TensorSpec.from_tensor,
                                                    sample_metric_variables)
  federated_local_outputs_type = computation_types.at_clients(
      metric_variable_type_dict)

  def federated_output(local_outputs):
    return base_utils.federated_aggregate_keras_metric(metrics, local_outputs)

  federated_output_computation = computations.federated_computation(
      federated_output, federated_local_outputs_type)
  return federated_output_computation
