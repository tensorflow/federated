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
"""Tests for local client training implemented in ClientFedAvg.

Integration tests that include server averaging and alternative tff.aggregator
factories are in found in
tensorflow_federated/python/tests/federated_averaging_integration_test.py.
"""

import collections
import functools
import itertools
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.optimizers import sgdm


class FederatedAveragingClientTest(test_case.TestCase, parameterized.TestCase):
  """Tests of ClientFedAvg that use a common model and data."""

  def create_dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def create_model(self, use_metrics_aggregator=False):
    return model_examples.LinearRegression(
        feature_dim=2, use_metrics_aggregator=use_metrics_aggregator)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[tf.zeros((2, 1)), tf.constant(0.0)],
        non_trainable=[0.0],
    )

  @parameterized.named_parameters((  # pylint: disable=g-complex-comprehension
      named_params[0][0] + '_' + named_params[1][0], *named_params[0][1:],
      named_params[1][1]) for named_params in itertools.product(
          [('non-simulation_noclip', True, False, {},
            0.1), ('unweighted_non-simulation_noclip', False, False, {},
                   0.1), ('simulation_noclip', True, True, {}, 0.1),
           ('non-simulation_clipnorm', True, False, {
               'clipnorm': 0.2
           }, 0.05
           ), ('non-simulation_clipvalue', True, False, {
               'clipvalue': 0.1
           }, 0.02)], [('use_metrics_aggregator',
                        True), ('not_use_metrics_aggregator', False)]))
  def test_client_tf(self, weighted, simulation, optimizer_kwargs,
                     expected_norm, use_metrics_aggregator):
    model = self.create_model(use_metrics_aggregator=use_metrics_aggregator)
    dataset = self.create_dataset()
    if weighted:
      client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
    else:
      client_weighting = client_weight_lib.ClientWeighting.UNIFORM
    client_tf = federated_averaging.ClientFedAvg(
        model,
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs),
        client_weighting=client_weighting,
        use_experimental_simulation_loop=simulation)
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_outputs.weights_delta, axis=-1), expected_norm)
    if weighted:
      self.assertEqual(client_outputs.weights_delta_weight, 8.0)
    if use_metrics_aggregator:
      self.assertDictContainsSubset({'num_examples': 8},
                                    client_outputs.model_output)
      self.assertBetween(client_outputs.model_output['loss'][0],
                         np.finfo(np.float32).eps, 10.0)
      self.assertEqual(client_outputs.model_output['loss'][1], 8.0)
    else:
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
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        client_weighting=lambda _: tf.constant(1.5))
    client_outputs = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model, lambda: tf.keras.optimizers.SGD(learning_rate=0.1))
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
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model,
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        use_experimental_simulation_loop=simulation)
    client_tf(dataset, self.initial_weights())
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()


class FederatedAveragingTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of FedAvg training process."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters((
      '_'.join(name for name, _ in named_params),
      *(param for _, param in named_params),
  ) for named_params in itertools.product([
      ('keras_optimizer', tf.keras.optimizers.SGD),
      ('tff_optimizer', sgdm.build_sgdm(learning_rate=0.1)),
  ], [
      ('robust_aggregator', model_update_aggregator.robust_aggregator),
      ('dp_aggregator', lambda: model_update_aggregator.dp_aggregator(1e-3, 3)),
      ('compression_aggregator',
       model_update_aggregator.compression_aggregator),
      ('secure_aggreagtor', model_update_aggregator.secure_aggregator),
  ]))
  # pylint: enable=g-complex-comprehension
  def test_construction_calls_model_fn(self, optimizer_fn, aggregation_factory):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    federated_averaging.build_federated_averaging_process(
        model_fn=mock_model_fn,
        client_optimizer_fn=optimizer_fn,
        model_update_aggregation_factory=aggregation_factory())
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)

  # TODO(b/202027089): Remove this test once the try/except logic is gone.
  @test_utils.skip_test_for_multi_gpu
  def test_custom_metrics_aggregator_does_not_get_ignored(self):

    def aggregate_nothing(metric_finaliers, local_unfinalized_metrics_type):
      del metric_finaliers  # Unused.

      @computations.federated_computation(
          computation_types.at_clients(local_unfinalized_metrics_type))
      def aggregator_computation(client_local_unfinalized_metrics):
        del client_local_unfinalized_metrics  # Unused.
        return collections.OrderedDict()

      return aggregator_computation

    # Constructs a model that implements the two old attributes.
    model_fn = lambda: model_examples.LinearRegression(  # pylint: disable=g-long-lambda
        use_metrics_aggregator=False)
    fed_avg = federated_averaging.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        metrics_aggregator=aggregate_nothing)
    self.assertEqual(fed_avg.next.type_signature.result[1].member.train,
                     computation_types.to_type(collections.OrderedDict()))

  def test_no_unsecure_aggregation_with_secure_aggregator(self):
    model_fn = functools.partial(
        model_examples.LinearRegression, use_metrics_aggregator=True)
    fed_avg = federated_averaging.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        model_update_aggregation_factory=model_update_aggregator
        .secure_aggregator(),
        metrics_aggregator=aggregator.secure_sum_then_finalize)
    static_assert.assert_not_contains_unsecure_aggregation(fed_avg.next)

if __name__ == '__main__':
  test_case.main()
