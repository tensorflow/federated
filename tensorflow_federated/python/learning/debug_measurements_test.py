# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import debug_measurements

TensorType = computation_types.TensorType
FloatType = TensorType(tf.float32)
FloatAtServer = computation_types.at_server(FloatType)
FloatAtClients = computation_types.at_clients(FloatType)

SERVER_MEASUREMENTS_OUTPUT_TYPE = computation_types.at_server(
    collections.OrderedDict([
        ('server_update_max', FloatType),
        ('server_update_norm', FloatType),
        ('server_update_min', FloatType),
    ]))

CLIENT_MEASUREMENTS_OUTPUT_TYPE = computation_types.at_server(
    collections.OrderedDict([
        ('average_client_norm', FloatType),
        ('std_dev_client_norm', FloatType),
    ]))


class DebugMeasurementsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_type', FloatType),
      ('vector_type', TensorType(tf.float32, [3])),
      ('struct_type', [FloatType, FloatType]),
      ('nested_struct_type', [
          [TensorType(tf.float32, [3])],
          [FloatType, FloatType],
      ]),
  )
  def test_server_measurement_fn_traceable_by_federated_computation(
      self, value_type):
    _, server_measurement_fn = (
        debug_measurements.build_aggregator_measurement_fns())
    input_type = computation_types.at_server(value_type)

    @computations.federated_computation(input_type)
    def get_server_measurements(server_update):
      return server_measurement_fn(server_update)

    type_signature = get_server_measurements.type_signature
    type_signature.parameter.check_assignable_from(input_type)
    type_signature.result.check_assignable_from(SERVER_MEASUREMENTS_OUTPUT_TYPE)

  @parameterized.named_parameters(
      ('scalar_type', FloatType),
      ('vector_type', TensorType(tf.float32, [3])),
      ('struct_type', [FloatType, FloatType]),
      ('nested_struct_type', [
          [TensorType(tf.float32, [3])],
          [FloatType, FloatType],
      ]),
  )
  def test_unweighted_client_measurement_fn_traceable_by_federated_computation(
      self, value_type):
    client_measurement_fn, _ = debug_measurements.build_aggregator_measurement_fns(
        weighted_aggregator=False)
    input_type = computation_types.at_clients(value_type)

    @computations.federated_computation(input_type)
    def get_client_measurements(client_update):
      return client_measurement_fn(client_update)

    type_signature = get_client_measurements.type_signature
    type_signature.parameter.check_assignable_from(input_type)
    type_signature.result.check_assignable_from(CLIENT_MEASUREMENTS_OUTPUT_TYPE)

  @parameterized.named_parameters(
      ('scalar_type', FloatType),
      ('vector_type', TensorType(tf.float32, [3])),
      ('struct_type', [FloatType, FloatType]),
      ('nested_struct_type', [
          [TensorType(tf.float32, [3])],
          [FloatType, FloatType],
      ]),
  )
  def test_weighted_client_measurement_fn_traceable_by_federated_computation(
      self, value_type):
    client_measurement_fn, _ = debug_measurements.build_aggregator_measurement_fns(
        weighted_aggregator=True)
    input_type = computation_types.at_clients(value_type)
    weights_type = computation_types.at_clients(tf.float32)

    @computations.federated_computation(input_type, weights_type)
    def get_client_measurements(client_update, client_weights):
      return client_measurement_fn(client_update, client_weights)

    type_signature = get_client_measurements.type_signature
    type_signature.parameter[0].check_assignable_from(input_type)
    type_signature.parameter[1].check_assignable_from(weights_type)
    type_signature.result.check_assignable_from(CLIENT_MEASUREMENTS_OUTPUT_TYPE)

  @parameterized.named_parameters(
      ('server_update1', [-3.0, 4.0, 0.0], 4.0, 5.0, -3.0),
      ('server_update2', [0.0], 0.0, 0.0, 0.0),
      ('server_update3',
       collections.OrderedDict(
           a=tf.constant([1.0, -1.0]),
           b=tf.constant(2.0),
       ), 2.0, tf.math.sqrt(6.0), -1.0),
  )
  def test_correctness_of_server_update_statistics(self, server_update,
                                                   expected_max, expected_norm,
                                                   expected_min):
    actual_server_statistics = debug_measurements.calculate_server_update_statistics(
        server_update)
    expected_server_statistics = collections.OrderedDict(
        server_update_max=expected_max,
        server_update_norm=expected_norm,
        server_update_min=expected_min)
    self.assertAllClose(actual_server_statistics, expected_server_statistics)

  @parameterized.named_parameters(
      ('distribution1', [1.0, 3.0, 0.0]),
      ('distribution2', [-1.0]),
      ('distribution3', [2.0, 2.0, 2.0]),
  )
  def test_correctness_unbiased_std_dev_unweighted(self, distribution):
    n = tf.cast(len(distribution), dtype=tf.float32)
    expected_value = tf.math.reduce_mean(distribution)
    expected_value_squared = tf.math.reduce_mean(tf.constant(distribution)**2)
    unbiased_std_dev = debug_measurements.calculate_unbiased_std_dev(
        expected_value, expected_value_squared, n, n)
    biased_std_dev = tf.math.reduce_std(distribution)
    correct_unbiased_std_dev = tf.math.sqrt(tf.math.divide_no_nan(
        n, n - 1)) * biased_std_dev
    self.assertNear(unbiased_std_dev, correct_unbiased_std_dev, 1e-6)

  @parameterized.named_parameters(
      ('client_updates1', [1.0, -2.0, 5.0]),
      ('client_updates2', [7.0]),
      ('client_updates3', [2.0, 2.0, 2.0]),
  )
  def test_correctness_of_unweighted_client_update_statistics(
      self, client_updates):
    client_weights = [1.0 for _ in client_updates]

    @computations.federated_computation(
        computation_types.at_clients(tf.float32),
        computation_types.at_clients(tf.float32))
    def compute_client_statistics(client_updates, client_weights):
      return debug_measurements.calculate_client_update_statistics(
          client_updates, client_weights)

    actual_client_statistics = compute_client_statistics(
        client_updates, client_weights)

    client_norms = [tf.math.abs(a) for a in client_updates]
    expected_average_norm = tf.math.reduce_mean(client_norms)
    num_clients = tf.cast(len(client_updates), tf.float32)
    expected_std_dev = tf.math.reduce_std(client_norms) * tf.math.sqrt(
        tf.math.divide_no_nan(num_clients, num_clients - 1))
    expected_client_statistics = collections.OrderedDict(
        average_client_norm=expected_average_norm,
        std_dev_client_norm=expected_std_dev)
    self.assertAllClose(actual_client_statistics, expected_client_statistics)

  @parameterized.named_parameters(
      ('distribution1', [1.0, 3.0, 0.0], [2.0, 3.0, 1.0]),
      ('distribution2', [-1.0], [5.0]),
      ('distribution3', [2.0, -2.0, 2.0], [6.0, 7.0, 4.0]),
      ('distribution4', [1.0, 2.0, -3.0], [1.0, 1.0, 0.0]),
  )
  def test_correctness_of_weighted_client_update_statistics(
      self, client_updates, client_weights):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32),
        computation_types.at_clients(tf.float32))
    def compute_client_statistics(client_updates, client_weights):
      return debug_measurements.calculate_client_update_statistics(
          client_updates, client_weights)

    actual_client_statistics = compute_client_statistics(
        client_updates, client_weights)

    client_updates = tf.constant(client_updates)
    client_weights = tf.constant(client_weights)
    weights_sum = tf.math.reduce_sum(client_weights)
    weights_squared_sum = tf.math.reduce_sum(client_weights**2)
    expected_norm = tf.math.divide_no_nan(
        tf.math.reduce_sum(tf.math.abs(client_updates) * client_weights),
        weights_sum)
    expected_norm_squared = tf.math.divide_no_nan(
        tf.math.reduce_sum(client_updates**2 * client_weights), weights_sum)

    biased_variance = expected_norm_squared - expected_norm**2
    unbiased_variance = tf.math.divide_no_nan(
        weights_sum**2, weights_sum**2 - weights_squared_sum) * biased_variance
    unbiased_std_dev = tf.math.sqrt(unbiased_variance)

    expected_client_statistics = collections.OrderedDict(
        average_client_norm=expected_norm, std_dev_client_norm=unbiased_std_dev)
    self.assertAllClose(actual_client_statistics, expected_client_statistics)

  def test_add_measurements_to_weighted_aggregation_factory_types(self):
    mean_factory = mean.MeanFactory()
    debug_mean_factory = debug_measurements.add_debug_measurements(mean_factory)
    value_type = computation_types.TensorType(tf.float32)
    mean_aggregator = mean_factory.create(value_type, value_type)
    debug_aggregator = debug_mean_factory.create(value_type, value_type)
    self.assertTrue(debug_aggregator.is_weighted)
    self.assertEqual(mean_aggregator.initialize.type_signature,
                     debug_aggregator.initialize.type_signature)
    self.assertEqual(mean_aggregator.next.type_signature.parameter,
                     debug_aggregator.next.type_signature.parameter)
    self.assertEqual(mean_aggregator.next.type_signature.result.state,
                     debug_aggregator.next.type_signature.result.state)
    self.assertEqual(mean_aggregator.next.type_signature.result.result,
                     debug_aggregator.next.type_signature.result.result)

  def test_add_measurements_to_weighted_aggregation_factory_output(self):
    mean_factory = mean.MeanFactory()
    debug_mean_factory = debug_measurements.add_debug_measurements(mean_factory)
    value_type = computation_types.TensorType(tf.float32)
    mean_aggregator = mean_factory.create(value_type, value_type)
    debug_aggregator = debug_mean_factory.create(value_type, value_type)

    state = mean_aggregator.initialize()
    mean_output = mean_aggregator.next(state, [2.0, 4.0], [1.0, 1.0])
    debug_output = debug_aggregator.next(state, [2.0, 4.0], [1.0, 1.0])
    self.assertEqual(mean_output.state, debug_output.state)
    self.assertNear(mean_output.result, debug_output.result, err=1e-6)

    mean_measurements = mean_output.measurements
    expected_debugging_measurements = {
        'average_client_norm': 3.0,
        'std_dev_client_norm': tf.math.sqrt(2.0),
        'server_update_max': 3.0,
        'server_update_norm': 3.0,
        'server_update_min': 3.0,
    }
    debugging_measurements = debug_output.measurements
    self.assertCountEqual(
        list(debugging_measurements.keys()),
        list(mean_measurements.keys()) +
        list(expected_debugging_measurements.keys()))
    for k in mean_output.measurements:
      self.assertEqual(mean_measurements[k], debugging_measurements[k])
    for k in expected_debugging_measurements:
      self.assertNear(
          debugging_measurements[k],
          expected_debugging_measurements[k],
          err=1e-6)

  def test_add_measurements_to_unweighted_aggregation_factory_types(self):
    mean_factory = mean.UnweightedMeanFactory()
    debug_mean_factory = debug_measurements.add_debug_measurements(mean_factory)
    value_type = computation_types.TensorType(tf.float32)
    mean_aggregator = mean_factory.create(value_type)
    debug_aggregator = debug_mean_factory.create(value_type)
    self.assertFalse(debug_aggregator.is_weighted)
    self.assertEqual(mean_aggregator.initialize.type_signature,
                     debug_aggregator.initialize.type_signature)
    self.assertEqual(mean_aggregator.next.type_signature.parameter,
                     debug_aggregator.next.type_signature.parameter)
    self.assertEqual(mean_aggregator.next.type_signature.result.state,
                     debug_aggregator.next.type_signature.result.state)
    self.assertEqual(mean_aggregator.next.type_signature.result.result,
                     debug_aggregator.next.type_signature.result.result)

  def test_add_measurements_to_unweighted_aggregation_factory_output(self):
    mean_factory = mean.UnweightedMeanFactory()
    debug_mean_factory = debug_measurements.add_debug_measurements(mean_factory)
    value_type = computation_types.TensorType(tf.float32)
    mean_aggregator = mean_factory.create(value_type)
    debug_aggregator = debug_mean_factory.create(value_type)

    state = mean_aggregator.initialize()
    mean_output = mean_aggregator.next(state, [2.0, 4.0])
    debug_output = debug_aggregator.next(state, [2.0, 4.0])
    self.assertEqual(mean_output.state, debug_output.state)
    self.assertNear(mean_output.result, debug_output.result, err=1e-6)

    mean_measurements = mean_output.measurements
    expected_debugging_measurements = {
        'average_client_norm': 3.0,
        'std_dev_client_norm': tf.math.sqrt(2.0),
        'server_update_max': 3.0,
        'server_update_norm': 3.0,
        'server_update_min': 3.0,
    }
    debugging_measurements = debug_output.measurements
    self.assertCountEqual(
        list(debugging_measurements.keys()),
        list(mean_measurements.keys()) +
        list(expected_debugging_measurements.keys()))
    for k in mean_output.measurements:
      self.assertEqual(mean_measurements[k], debugging_measurements[k])
    for k in expected_debugging_measurements:
      self.assertNear(
          debugging_measurements[k],
          expected_debugging_measurements[k],
          err=1e-6)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
