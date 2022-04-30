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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor


class MultiGPUTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.create_logical_multi_gpus()

  def test_check_dataset_reduce_for_multi_gpu_no_reduce_no_raise(self):
    with tf.Graph().as_default() as graph:
      tf.data.Dataset.range(10).map(lambda x: x + 1)
    eager_tf_executor._check_dataset_reduce_for_multi_gpu(graph.as_graph_def())

  def test_check_dataset_reduce_for_multi_gpu(self):
    with tf.Graph().as_default() as graph:
      tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)
    with self.assertRaisesRegex(
        ValueError, 'Detected dataset reduce op in multi-GPU TFF simulation.*'):
      eager_tf_executor._check_dataset_reduce_for_multi_gpu(
          graph.as_graph_def())

  def test_get_no_arg_wrapped_function_multi_gpu_no_reduce(self):

    @computations.tf_computation
    def comp():
      value = tf.constant(0, dtype=tf.int64)
      for d in iter(tf.data.Dataset.range(10)):
        value += d
      return value

    wrapped_fn = eager_tf_executor._get_wrapped_function_from_comp(
        computation_impl.ConcreteComputation.get_proto(comp),
        must_pin_function_to_cpu=False,
        param_type=None,
        device=None)
    self.assertEqual(wrapped_fn(), np.int64(45))

  def test_get_no_arg_wrapped_function_multi_gpu_tf_device(self):

    logical_gpus = tf.config.list_logical_devices('GPU')

    @computations.tf_computation
    def comp():
      with tf.device(logical_gpus[0].name):
        return tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)

    wrapped_fn = eager_tf_executor._get_wrapped_function_from_comp(
        computation_impl.ConcreteComputation.get_proto(comp),
        must_pin_function_to_cpu=False,
        param_type=None,
        device=None)
    self.assertEqual(wrapped_fn(), np.int64(45))


if __name__ == '__main__':
  tf.test.main()
