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

import collections
from unittest import mock
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name='num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_pred)[0], sample_weight)


class FederatedAveragingClientWithModelTest(test_case.TestCase,
                                            parameterized.TestCase):
  """Tests of ClientFedAvg that use a common model and data."""

  def create_dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.LinearRegression.make_batch(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def create_model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[tf.zeros((2, 1)), tf.constant(0.0)],
        non_trainable=[0.0],
    )

  @parameterized.named_parameters(
      ('non-simulation_noclip', True, False, {}, 0.1),
      ('unweighted_non-simulation_noclip', False, False, {}, 0.1),
      ('simulation_noclip', True, True, {}, 0.1),
      ('non-simulation_clipnorm', True, False, {
          'clipnorm': 0.2
      }, 0.05),
      ('non-simulation_clipvalue', True, False, {
          'clipvalue': 0.1
      }, 0.02),
  )
  @test_utils.skip_test_for_multi_gpu
  def test_client_tf(self, weighted, simulation, optimizer_kwargs,
                     expected_norm):
    model = self.create_model()
    dataset = self.create_dataset()
    if weighted:
      client_weighting = federated_averaging.ClientWeighting.NUM_EXAMPLES
    else:
      client_weighting = federated_averaging.ClientWeighting.UNIFORM
    client_tf = federated_averaging.ClientFedAvg(
        model,
        tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs),
        client_weighting=client_weighting,
        use_experimental_simulation_loop=simulation)
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_outputs.weights_delta, axis=-1), expected_norm)
    if weighted:
      self.assertEqual(client_outputs.weights_delta_weight, 8.0)
    self.assertEqual(client_outputs.optimizer_output['num_examples'], 8)
    self.assertDictContainsSubset(
        {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
        }, client_outputs.model_output)
    self.assertBetween(client_outputs.model_output['loss'],
                       np.finfo(np.float32).eps, 10.0)

  def test_client_tf_custom_delta_weight(self):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model,
        tf.keras.optimizers.SGD(learning_rate=0.1),
        client_weighting=lambda _: tf.constant(1.5))
    client_outputs = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model, tf.keras.optimizers.SGD(learning_rate=0.1))
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(dataset, init_weights)
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.weights_delta), [[[0.0], [0.0]], 0.0])

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model,
        tf.keras.optimizers.SGD(learning_rate=0.1),
        use_experimental_simulation_loop=simulation)
    client_tf(dataset, self.initial_weights())
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()


class FederatedAveragingModelTffTest(test_case.TestCase,
                                     parameterized.TestCase):

  def _run_test(self, process, *, datasets, expected_num_examples):
    state = process.initialize()
    prev_loss = np.inf
    aggregation_metrics = collections.OrderedDict(mean_value=(), mean_weight=())
    for _ in range(3):
      state, metric_outputs = process.next(state, datasets)
      self.assertEqual(
          list(metric_outputs.keys()),
          ['broadcast', 'aggregation', 'train', 'stat'])
      self.assertEmpty(metric_outputs['broadcast'])
      self.assertEqual(aggregation_metrics, metric_outputs['aggregation'])
      train_metrics = metric_outputs['train']
      self.assertEqual(train_metrics['num_examples'], expected_num_examples)
      self.assertLess(train_metrics['loss'], prev_loss)
      prev_loss = train_metrics['loss']

  @parameterized.named_parameters([
      ('unweighted', federated_averaging.ClientWeighting.UNIFORM),
      ('example_weighted', federated_averaging.ClientWeighting.NUM_EXAMPLES),
      ('custom_weighted', lambda _: tf.constant(1.5)),
  ])
  @test_utils.skip_test_for_multi_gpu
  def test_basic_orchestration_execute(self, client_weighting):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        client_weighting=client_weighting)

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=2 * num_clients)

  @parameterized.named_parameters([
      ('functional_model',
       model_examples.build_linear_regression_keras_functional_model),
      ('sequential_model',
       model_examples.build_linear_regression_keras_sequential_model),
  ])
  @test_utils.skip_test_for_multi_gpu
  def test_orchestration_execute_from_keras(self, build_keras_model_fn):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    def model_fn():
      keras_model = build_keras_model_fn(feature_dims=2)
      return keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[NumExamplesCounter()])

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01))

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=2 * num_clients)

  @test_utils.skip_test_for_multi_gpu
  def test_orchestration_execute_from_keras_with_lookup(self):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[['R'], ['G'], ['B']], y=[[1.0], [2.0], [3.0]])).batch(2)

    def model_fn():
      keras_model = model_examples.build_lookup_table_keras_model()
      return keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[NumExamplesCounter()])

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=3 * num_clients)

  @test_utils.skip_test_for_multi_gpu
  def test_execute_empty_data(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0]],
            y=[[5.0]],
        )).batch(
            5, drop_remainder=True)

    server_state = iterative_process.initialize()

    first_state, metric_outputs = iterative_process.next(server_state, [ds] * 2)
    self.assertAllClose(
        list(first_state.model.trainable), [[[0.0], [0.0]], 0.0])
    self.assertEqual(metric_outputs['train']['num_examples'], 0)
    self.assertTrue(tf.math.is_nan(metric_outputs['train']['loss']))

  @test_utils.skip_test_for_multi_gpu
  def test_get_model_weights(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    num_clients = 3
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    datasets = [ds] * num_clients

    state = iterative_process.initialize()
    self.assertIsInstance(
        iterative_process.get_model_weights(state), model_utils.ModelWeights)
    self.assertAllClose(state.model.trainable,
                        iterative_process.get_model_weights(state).trainable)

    for _ in range(3):
      state, _ = iterative_process.next(state, datasets)
      self.assertIsInstance(
          iterative_process.get_model_weights(state), model_utils.ModelWeights)
      self.assertAllClose(state.model.trainable,
                          iterative_process.get_model_weights(state).trainable)

  @parameterized.named_parameters([
      ('robust', model_update_aggregator.robust_aggregator),
      ('dp', lambda: model_update_aggregator.dp_aggregator(1e-3, 3)),
      ('compression', model_update_aggregator.compression_aggregator),
      ('secure', model_update_aggregator.secure_aggregator),
  ])
  @test_utils.skip_test_for_multi_gpu
  def test_recommended_aggregations_execute(self, default_aggregation):
    process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        model_update_aggregation_factory=default_aggregation())

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    num_clients = 3
    state = process.initialize()
    state, metrics = process.next(state, [ds] * num_clients)
    self.assertNotEmpty(metrics['aggregation'])


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
