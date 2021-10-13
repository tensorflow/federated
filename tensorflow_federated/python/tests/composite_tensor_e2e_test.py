# Copyright 2021, The TensorFlow Federated Authors.
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
"""Tests for end-to-end support of composite tensors.

Composite tensors are TensorFlow structures that consist of multiple different
underlying tensors, as well as potentially some additional metadata.

Currently supported composite tensors include:
- tf.RaggedTensor
- tf.SparseTensor

Future support is planned for:
- tf.IndexedSlices
- tf.Struct
"""

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff


def create_ragged():
  return tf.RaggedTensor.from_row_splits([0, 0, 0, 0], [0, 1, 4])


class RaggedTensorTest(absltest.TestCase):

  def test_inferred_type_assignable_to_type_spec(self):
    tf_comp = tff.tf_computation(create_ragged)
    type_from_return = tf_comp.type_signature.result

    ragged_tensor_spec = tf.RaggedTensorSpec.from_value(create_ragged())
    type_from_spec = tff.to_type(ragged_tensor_spec)

    type_from_spec.check_assignable_from(type_from_return)

  def test_passes_through_computation(self):
    ragged_tensor_spec = tf.RaggedTensorSpec.from_value(create_ragged())

    @tff.tf_computation(ragged_tensor_spec)
    def check_ragged(ragged):
      self.assertIsInstance(ragged, tf.RaggedTensor)
      return ragged

    out = check_ragged(create_ragged())
    self.assertIsInstance(out, tf.RaggedTensor)


def create_sparse():
  return tf.SparseTensor(indices=[[1]], values=[2], dense_shape=[5])


class SparseTensorTest(absltest.TestCase):

  def test_inferred_type_assignable_to_type_spec(self):
    tf_comp = tff.tf_computation(create_sparse)
    type_from_return = tf_comp.type_signature.result

    sparse_tensor_spec = tf.SparseTensorSpec.from_value(create_sparse())
    type_from_spec = tff.to_type(sparse_tensor_spec)

    type_from_spec.check_assignable_from(type_from_return)

  def test_passes_through_computation(self):
    sparse_tensor_spec = tf.SparseTensorSpec.from_value(create_sparse())

    @tff.tf_computation(sparse_tensor_spec)
    def check_sparse(sparse):
      self.assertIsInstance(sparse, tf.SparseTensor)
      self.assertEqual(sparse.shape, tf.TensorShape([5]))
      return sparse

    out = check_sparse(create_sparse())
    self.assertIsInstance(out, tf.SparseTensor)


if __name__ == '__main__':
  absltest.main()
