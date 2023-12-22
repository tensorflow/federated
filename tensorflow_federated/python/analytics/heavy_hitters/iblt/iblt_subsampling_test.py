# Copyright 2023, The TensorFlow Federated Authors.
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
import math
from typing import Union

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_subsampling
from tensorflow_federated.python.analytics.heavy_hitters.iblt import subsample_process
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types


DATA = [
    (
        ['seattle', 'hello', 'world', 'bye'],
        [[1], [4], [1], [2]],
    ),
    (['hi', 'seattle'], [[2], [5]]),
    (
        ['good', 'morning', 'hi', 'bye'],
        [[3], [6], [2], [3]],
    ),
]

AGGREGATED_DATA = {
    'seattle': [6],
    'hello': [4],
    'world': [1],
    'hi': [4],
    'good': [3],
    'morning': [6],
    'bye': [5],
}


def _generate_client_data(
    input_structure: list[tuple[list[Union[str, bytes]], list[list[int]]]]
) -> list[tf.data.Dataset]:
  client_data = []
  for input_strings, string_values in input_structure:
    client = collections.OrderedDict([
        (
            iblt_factory.DATASET_KEY,
            tf.constant(input_strings, dtype=tf.string),
        ),
        (
            iblt_factory.DATASET_VALUE,
            tf.constant(string_values, dtype=tf.int64),
        ),
    ])
    client_data.append(tf.data.Dataset.from_tensor_slices(client))
  return client_data


CLIENT_DATA = _generate_client_data(DATA)


class IbltSubsamplingTest(tf.test.TestCase, parameterized.TestCase):

  def test_incorrect_value_type(self):
    iblt_fac = iblt_factory.IbltFactory(
        capacity=100, string_max_bytes=10, repetitions=3, seed=0
    )
    sampling_process = subsample_process.ThresholdSamplingProcess(1.0)
    subsample_fac = iblt_subsampling.SubsampledIbltFactory(
        iblt_fac, sampling_process
    )
    wrong_type = computation_types.SequenceType(
        collections.OrderedDict([
            (iblt_factory.DATASET_KEY, tf.string),
            (
                iblt_factory.DATASET_VALUE,
                computation_types.TensorType(shape=[None], dtype=tf.int32),
            ),
        ])
    )
    with self.assertRaises(ValueError):
      subsample_fac.create(wrong_type)

  @parameterized.named_parameters(
      {'testcase_name': 'threshold 1.0', 'threshold': 1.0},
      {'testcase_name': 'threshold 2.0', 'threshold': 2.0},
      {'testcase_name': 'threshold 5.0', 'threshold': 5.0},
      {'testcase_name': 'threshold 10.0', 'threshold': 10.0},
  )
  def test_subsampling_factory(self, threshold: float):
    iblt_fac = iblt_factory.IbltFactory(
        capacity=100, string_max_bytes=10, repetitions=3
    )
    sampling_process = subsample_process.ThresholdSamplingProcess(threshold)
    subsample_fac = iblt_subsampling.SubsampledIbltFactory(
        iblt_fac, sampling_process
    )
    value_type = computation_types.SequenceType(
        collections.OrderedDict([
            (iblt_factory.DATASET_KEY, tf.string),
            (
                iblt_factory.DATASET_VALUE,
                computation_types.TensorType(shape=(1,), dtype=tf.int64),
            ),
        ])
    )
    agg_process = subsample_fac.create(value_type)
    state = agg_process.initialize()
    num_rounds = 100
    output_counts = {
        'seattle': 0,
        'hello': 0,
        'world': 0,
        'hi': 0,
        'bye': 0,
        'good': 0,
        'morning': 0,
    }
    for _ in range(num_rounds):
      process_output = agg_process.next(state, CLIENT_DATA)
      state = process_output.state
      heavy_hitters = process_output.result.output_strings
      heavy_hitters_counts = process_output.result.string_values[:, 0]
      hist_round = dict(zip(heavy_hitters, heavy_hitters_counts))
      for x in hist_round:
        output_counts[x.decode('utf-8')] += hist_round[x]
    for x in AGGREGATED_DATA:
      self.assertAllClose(
          output_counts[x] / float(num_rounds),
          AGGREGATED_DATA[x][0],
          atol=threshold / math.sqrt(num_rounds),
      )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
