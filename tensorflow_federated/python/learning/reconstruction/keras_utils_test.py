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
"""Tests for keras_utils.py."""

import collections

import tensorflow as tf

from tensorflow_federated.python.learning.reconstruction import keras_utils
from tensorflow_federated.python.learning.reconstruction import model as model_lib


def _create_input_spec():
  return collections.namedtuple('Batch', ['x', 'y'])(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32))


def _create_keras_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Reshape(target_shape=[784], input_shape=(28 * 28,)),
      tf.keras.layers.Dense(10),
  ])
  return model


class KerasUtilsTest(tf.test.TestCase):

  def test_from_keras_model_succeeds(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers,
        local_layers=[],
        input_spec=input_spec)

  def test_from_keras_model_fails_bad_input_spec(self):
    keras_model = _create_keras_model()
    input_spec = collections.namedtuple('Batch', ['x'])(
        x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32))

    with self.assertRaisesRegex(ValueError, 'input_spec'):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          global_layers=keras_model.layers,
          local_layers=[],
          input_spec=input_spec)

  def test_from_keras_model_fails_compiled(self):
    keras_model = _create_keras_model()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
    input_spec = _create_input_spec()

    with self.assertRaisesRegex(ValueError, 'compiled'):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          global_layers=keras_model.layers,
          local_layers=[],
          input_spec=input_spec)

  def test_from_keras_model_fails_missing_variables(self):
    """Ensures failure if global/local layers are missing variables."""
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    with self.assertRaisesRegex(ValueError, 'variables'):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          global_layers=keras_model.layers[:-1],
          local_layers=[],
          input_spec=input_spec)

  def test_from_keras_model_succeeds_from_set(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=set(keras_model.layers),
        local_layers=set(),
        input_spec=input_spec)

  def test_from_keras_model_properties(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    recon_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers,
        local_layers=[],
        input_spec=input_spec)

    # Global trainable/non_trainable should include all the variables, and
    # local should be empty.
    self.assertEqual(recon_model.global_trainable_variables,
                     keras_model.trainable_variables)
    self.assertEqual(recon_model.global_non_trainable_variables,
                     keras_model.non_trainable_variables)
    self.assertEmpty(recon_model.local_trainable_variables)
    self.assertEmpty(recon_model.local_non_trainable_variables)
    self.assertEqual(input_spec, recon_model.input_spec)

  def test_from_keras_model_local_layers_properties(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    recon_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],  # Last Dense layer is local.
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)

    # Expect last two variables, the weights and bias for the final Dense layer,
    # to be local trainable, and the rest global.
    self.assertEqual(recon_model.global_trainable_variables,
                     keras_model.trainable_variables[:-2])
    self.assertEqual(recon_model.global_non_trainable_variables,
                     keras_model.non_trainable_variables)
    self.assertEqual(recon_model.local_trainable_variables,
                     keras_model.trainable_variables[-2:])
    self.assertEmpty(recon_model.local_non_trainable_variables)
    self.assertEqual(input_spec, recon_model.input_spec)

  def test_from_keras_model_forward_pass(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    recon_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)

    batch_input = collections.namedtuple('Batch', ['x', 'y'])(
        x=tf.ones(shape=[10, 784], dtype=tf.float32),
        y=tf.zeros(shape=[10, 1], dtype=tf.int32))

    batch_output = recon_model.forward_pass(batch_input)

    self.assertIsInstance(batch_output, model_lib.BatchOutput)
    self.assertEqual(batch_output.num_examples, 10)
    self.assertAllEqual(batch_output.labels,
                        tf.zeros(shape=[10, 1], dtype=tf.int32))

    # Change num_examples and labels.
    batch_input = collections.namedtuple('Batch', ['x', 'y'])(
        x=tf.zeros(shape=[5, 784], dtype=tf.float32),
        y=tf.ones(shape=[5, 1], dtype=tf.int32))

    batch_output = recon_model.forward_pass(batch_input)

    self.assertIsInstance(batch_output, model_lib.BatchOutput)
    self.assertEqual(batch_output.num_examples, 5)
    self.assertAllEqual(batch_output.labels,
                        tf.ones(shape=[5, 1], dtype=tf.int32))

  def test_from_keras_model_forward_pass_list_input(self):
    """Forward pass still works with a 2-element list batch input."""
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    recon_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)

    batch_input = [
        tf.ones(shape=[10, 784], dtype=tf.float32),
        tf.zeros(shape=[10, 1], dtype=tf.int32)
    ]

    batch_output = recon_model.forward_pass(batch_input)

    self.assertIsInstance(batch_output, model_lib.BatchOutput)
    self.assertEqual(batch_output.num_examples, 10)
    self.assertAllEqual(batch_output.labels,
                        tf.zeros(shape=[10, 1], dtype=tf.int32))

  def test_from_keras_model_forward_pass_fails_bad_input_keys(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()

    recon_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers,
        local_layers=[],
        input_spec=input_spec)

    batch_input = collections.namedtuple('Batch', ['a', 'b'])(
        a=tf.ones(shape=[10, 784], dtype=tf.float32),
        b=tf.zeros(shape=[10, 1], dtype=tf.int32))

    with self.assertRaisesRegex(KeyError, 'keys'):
      recon_model.forward_pass(batch_input)

  def test_mean_loss_metric_from_keras_loss(self):
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse_metric = keras_utils.MeanLossMetric(mse_loss)

    y_true = tf.ones([10, 1], dtype=tf.float32)
    y_pred = tf.ones([10, 1], dtype=tf.float32) * 0.5

    mse_metric.update_state(y_true, y_pred)
    self.assertEqual(mse_loss(y_true, y_pred), mse_metric.result())

  def test_mean_loss_metric_multiple_weighted_batches(self):
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse_metric = keras_utils.MeanLossMetric(mse_loss)

    y_true = tf.ones([10, 1], dtype=tf.float32)
    y_pred = tf.ones([10, 1], dtype=tf.float32) * 0.5
    mse_metric.update_state(y_true, y_pred)

    y_true = tf.ones([40, 1], dtype=tf.float32)
    y_pred = tf.ones([40, 1], dtype=tf.float32)
    mse_metric.update_state(y_true, y_pred)

    # Final weighted loss is (10 * 0.5^2 + 40 * 0.0) / 50
    self.assertEqual(mse_metric.result(), 0.05)

  def test_mean_loss_metric_from_fn(self):
    """Ensures the mean loss metric also works with a callable."""

    def mse_loss(y_true, y_pred):
      return tf.reduce_mean(tf.square(y_true - y_pred))

    mse_metric = keras_utils.MeanLossMetric(mse_loss)

    y_true = tf.ones([10, 1], dtype=tf.float32)
    y_pred = tf.ones([10, 1], dtype=tf.float32) * 0.5

    mse_metric.update_state(y_true, y_pred)
    self.assertEqual(mse_loss(y_true, y_pred), mse_metric.result())

  def test_recreate_mean_loss_from_keras_loss(self):
    """Ensures we can create a metric from config, as is done in aggregation."""
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse_metric = keras_utils.MeanLossMetric(mse_loss)
    recreated_mse_metric = type(mse_metric).from_config(mse_metric.get_config())

    y_true = tf.ones([10, 1], dtype=tf.float32)
    y_pred = tf.ones([10, 1], dtype=tf.float32) * 0.5

    mse_metric.update_state(y_true, y_pred)
    recreated_mse_metric.update_state(y_true, y_pred)

    self.assertEqual(recreated_mse_metric.result(), mse_metric.result())

  def test_recreate_mean_loss_from_fn(self):

    def mse_loss(y_true, y_pred):
      return tf.reduce_mean(tf.square(y_true - y_pred))

    mse_metric = keras_utils.MeanLossMetric(mse_loss)
    recreated_mse_metric = type(mse_metric).from_config(mse_metric.get_config())

    y_true = tf.ones([10, 1], dtype=tf.float32)
    y_pred = tf.ones([10, 1], dtype=tf.float32) * 0.5

    mse_metric.update_state(y_true, y_pred)
    recreated_mse_metric.update_state(y_true, y_pred)

    self.assertEqual(recreated_mse_metric.result(), mse_metric.result())


if __name__ == '__main__':
  tf.test.main()
