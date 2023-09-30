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
"""Tests for reconstruction_utils.py."""

import collections

import tensorflow as tf

from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.reconstruction import keras_utils
from tensorflow_federated.python.learning.reconstruction import reconstruction_utils


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32))


def _create_keras_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Reshape(target_shape=[784], input_shape=(28 * 28,)),
      tf.keras.layers.Dense(10),
  ])
  return model


class ReconstructionUtilsTest(tf.test.TestCase):

  def test_simple_dataset_split_fn(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    recon_dataset, post_recon_dataset = reconstruction_utils.simple_dataset_split_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=2, post_recon_epochs=1)
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_recon_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=2, recon_steps_max=4)
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

    # Adding more steps than the number of actual steps has no effect.
    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=2, recon_steps_max=7)
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_recon_epochs_max_steps_zero_post_epochs(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=1,
        recon_steps_max=4,
        post_recon_epochs=0)

    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_post_recon_multiple_epochs_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        post_recon_epochs=2, post_recon_steps_max=4)
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])

  def test_build_dataset_split_fn_split_dataset_odd_batches(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

  def test_build_dataset_split_fn_split_dataset_even_batches(self):
    # 4 batches.
    client_dataset = tf.data.Dataset.range(8).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3], [6, 7]])

  def test_build_dataset_split_fn_split_dataset_zero_batches(self):
    """Ensures clients without any data don't fail."""
    # 0 batches.
    client_dataset = tf.data.Dataset.range(0).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_split_dataset_one_batch(self):
    """Ensures clients without any data don't fail."""
    # 1 batch. Batch size can be larger than number of examples.
    client_dataset = tf.data.Dataset.range(1).batch(4)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0]])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_split_dataset_other_args(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=2,
        recon_steps_max=3,
        post_recon_epochs=2,
        post_recon_steps_max=3,
        split_dataset=True)

    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list, [[2, 3], [2, 3]])

  def test_build_dataset_split_fn_split_dataset_steps_max_0(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs=1,
        recon_steps_max=3,
        post_recon_epochs=2,
        post_recon_steps_max=0,
        split_dataset=True)

    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [])

  def test_get_global_variables(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)

    global_weights = reconstruction_utils.get_global_variables(model)

    self.assertIsInstance(global_weights, model_utils.ModelWeights)
    # The last layer of the Keras model, which is a local Dense layer, contains
    # 2 trainable variables for the weights and bias.
    self.assertEqual(global_weights.trainable,
                     keras_model.trainable_variables[:-2])
    self.assertEmpty(global_weights.non_trainable)

  def test_get_local_variables(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)

    local_weights = reconstruction_utils.get_local_variables(model)

    self.assertIsInstance(local_weights, model_utils.ModelWeights)
    # The last layer of the Keras model, which is a local Dense layer, contains
    # 2 trainable variables for the weights and bias.
    self.assertEqual(local_weights.trainable,
                     keras_model.trainable_variables[-2:])
    self.assertEmpty(local_weights.non_trainable)

  def test_has_only_global_variables_true(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers,
        local_layers=[],
        input_spec=input_spec)
    self.assertTrue(reconstruction_utils.has_only_global_variables(model))

  def test_has_only_global_variables_false(self):
    keras_model = _create_keras_model()
    input_spec = _create_input_spec()
    model = keras_utils.from_keras_model(
        keras_model=keras_model,
        global_layers=keras_model.layers[:-1],
        local_layers=keras_model.layers[-1:],
        input_spec=input_spec)
    self.assertFalse(reconstruction_utils.has_only_global_variables(model))


if __name__ == '__main__':
  tf.test.main()
