# Copyright 2022, The TensorFlow Federated Authors.
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
from typing import Union

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib
from tensorflow_federated.python.analytics.heavy_hitters.iblt import subsample_process

# Repetition to be used throughout the tests
REPETITION = 3


def _get_count_from_dataset(dataset: tf.data.Dataset, key: str) -> int:
  for x in dataset:
    if x[iblt_factory.DATASET_KEY] == key:
      return x[iblt_factory.DATASET_VALUE][0].numpy()
  return 0  # Return 0 if x is not in the dataset.


def _generate_client_local_data(
    client_data: tuple[list[Union[str, bytes]], list[list[int]]]
) -> tf.data.Dataset:
  input_strings, string_values = client_data
  client_dict = collections.OrderedDict([
      (
          iblt_factory.DATASET_KEY,
          tf.constant(input_strings, dtype=tf.string),
      ),
      (
          iblt_factory.DATASET_VALUE,
          tf.constant(string_values, dtype=tf.int64),
      ),
  ])
  return tf.data.Dataset.from_tensor_slices(client_dict)


THRESHOLD = 4

DATA_ALL_ABOVE_THRESHOLD = _generate_client_local_data(
    (['seattle', 'hello', 'world'], [[4], [5], [10]])
)
DATA_WITH_NEGATIVE = _generate_client_local_data(
    (['hi', 'seattle'], [[2], [-5]])
)
DATA_SOME_IN_BETWEEN = _generate_client_local_data(
    (['good', 'morning', 'hi', 'bye'], [[3], [1], [2], [5]])
)
DATA_ALL_ZERO = _generate_client_local_data((['new', 'york'], [[0], [0]]))
DATA_SOME_IN_BETWEEN_WITH_UNIQUE_COUNT = _generate_client_local_data(
    (['good', 'morning', 'hi', 'bye'], [[3, 1], [1, 1], [2, 1], [5, 1]])
)


class ThresholdSubsampleProcessTest(tf.test.TestCase, parameterized.TestCase):

  def test_threshold_at_least_one(self):
    with self.assertRaisesRegex(ValueError, 'Threshold must be at least 1.'):
      subsample_process.ThresholdSamplingProcess(init_param=0)
    with self.assertRaisesRegex(ValueError, 'Threshold must be at least 1.'):
      subsample_process.ThresholdSamplingProcess(init_param=-1)
    # Should not raise
    subsample_process.ThresholdSamplingProcess(init_param=THRESHOLD)

  def test_not_adaptive(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=2
    )
    self.assertEqual(threshold_sampling.is_process_adaptive, False)

  def test_tuning_not_implemented(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=2
    )
    with self.assertRaisesRegex(
        NotImplementedError, 'Not an adaptive process.'
    ):
      threshold_sampling.update(
          subsampling_param_old=2,
          measurements=subsample_process.AggregationMeasurements(0, 0, 0, 0, 3),
      )

  def test_get_init_param(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    self.assertEqual(threshold_sampling.get_init_param(), THRESHOLD)

  def test_negative_value(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'Current implementation only supports positive values.',
    ):
      subsample_param = threshold_sampling.get_init_param()
      list(threshold_sampling.subsample_fn(DATA_WITH_NEGATIVE, subsample_param))

  def test_all_above_threshold(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    subsample_param = threshold_sampling.get_init_param()
    self.assertEqual(
        list(
            threshold_sampling.subsample_fn(
                DATA_ALL_ABOVE_THRESHOLD, subsample_param
            )
        ),
        list(DATA_ALL_ABOVE_THRESHOLD),
    )

  def test_all_zero(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    subsample_param = threshold_sampling.get_init_param()
    self.assertEmpty(
        list(threshold_sampling.subsample_fn(DATA_ALL_ZERO, subsample_param))
    )

  def test_sampling_in_between(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    rep = 300
    strings = ['good', 'morning', 'hi', 'bye']
    expected_freqs = np.array([3, 1, 2, 5])
    counts = np.zeros(len(strings))
    subsample_param = threshold_sampling.get_init_param()
    for i in range(rep):
      tf.random.set_seed(i)
      sampled_dataset = threshold_sampling.subsample_fn(
          DATA_SOME_IN_BETWEEN, subsample_param
      )
      for j, _ in enumerate(strings):
        counts[j] += _get_count_from_dataset(sampled_dataset, strings[j])
    self.assertAllClose(counts / rep, expected_freqs, atol=0.45)

  def test_sampling_in_between_with_unique_count(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD
    )
    rep = 300
    strings = ['good', 'morning', 'hi', 'bye']
    expected_freqs = np.array([3, 1, 2, 5])
    counts = np.zeros(len(strings))
    subsample_param = threshold_sampling.get_init_param()
    for i in range(rep):
      tf.random.set_seed(i)
      sampled_dataset = threshold_sampling.subsample_fn_with_unique_count(
          DATA_SOME_IN_BETWEEN_WITH_UNIQUE_COUNT, subsample_param
      )
      for j, _ in enumerate(strings):
        counts[j] += _get_count_from_dataset(sampled_dataset, strings[j])
    self.assertAllClose(counts / rep, expected_freqs, atol=0.45)

  @parameterized.named_parameters(
      {
          'testcase_name': 'init_threshold_one',
          'init_threshold': 1,
          'beta': 0.5,
          'capacity': 100,
          'num_recovered': 40,
      },
      {
          'testcase_name': 'init_threshold_larger_than_one',
          'init_threshold': 4,
          'beta': 0.3,
          'capacity': 200,
          'num_recovered': 100,
      },
  )
  def test_update_when_decoding_succeeds(
      self,
      init_threshold,
      beta,
      capacity,
      num_recovered,
  ):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=init_threshold, is_adaptive=True, beta=beta
    )
    repetition = REPETITION
    num_nonempty_buckets = 0
    iblt_size = (
        iblt_lib._internal_parameters(capacity, repetition)[0] * repetition
    )
    measurements = subsample_process.AggregationMeasurements(
        iblt_size=iblt_size,
        capacity=capacity,
        num_nonempty_buckets=num_nonempty_buckets,
        num_recovered=num_recovered,
        repetition=3,
    )
    expected_new_threshold = max(
        init_threshold * beta
        + (1 - beta) * (num_recovered / capacity) * init_threshold,
        1,
    )
    self.assertAlmostEqual(
        threshold_sampling.update(init_threshold, measurements),
        expected_new_threshold,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'num_nonempty_equal_iblt_size',
          'init_threshold': 5,
          'beta': 0.5,
          'capacity': 100,
          'num_recovered': 0,
          'num_nonempty_buckets': (
              iblt_lib._internal_parameters(100, REPETITION)[0] * REPETITION
          ),
      },
      {
          'testcase_name': 'num_nonempty_below_iblt_size',
          'init_threshold': 5,
          'beta': 0.3,
          'capacity': 200,
          'num_recovered': 40,
          'num_nonempty_buckets': int(
              iblt_lib._internal_parameters(100, REPETITION)[0] * REPETITION / 2
          ),
      },
  )
  def test_update_when_decoding_fails(
      self, init_threshold, beta, capacity, num_recovered, num_nonempty_buckets
  ):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=init_threshold, is_adaptive=True, beta=beta
    )
    repetition = REPETITION
    iblt_size = (
        iblt_lib._internal_parameters(capacity, repetition)[0] * repetition
    )
    measurements = subsample_process.AggregationMeasurements(
        iblt_size=iblt_size,
        capacity=capacity,
        num_nonempty_buckets=num_nonempty_buckets,
        num_recovered=num_recovered,
        repetition=3,
    )
    self.assertGreaterEqual(
        threshold_sampling.update(init_threshold, measurements),
        init_threshold,
    )


if __name__ == '__main__':
  tf.test.main()
