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
"""Subsampling and tuning interface for subsampled SecAggIBLT.

The class interface implements the subsampling function needed for subsampled
SecAggIBLT and the tuning function for adjusting the subsampling parameter based
on server measurements. A common use is to combine it with an IBLT aggregation
process and apply the subsampling function to users' local datasets before
passing them to the aggregation process. The subsampling parameter can be
updated at each round to obtain instance-dependent algorithm.
"""

import abc
import collections
import math
from typing import NamedTuple

import scipy
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory


class AggregationMeasurements(NamedTuple):
  iblt_size: int
  num_nonempty_buckets: int
  num_recovered: int
  capacity: int
  repetition: int


def _predict_num_unique_rep_is_3(
    num_nonempty_buckets: int, iblt_size: int
) -> float:
  """Predict the number of unique keys in a failed IBLT with repetition = 3.

  The prediction formula is based on Theorem 1 in
  https://tspace.library.utoronto.ca/bitstream/1807/9523/3/cores%20.pdf, which
  studies the process of generating a random hypergraph (IBLT encoding process),
  and dervies a formula that relates the core size (`num_nonempty_buckets`), the
  size of the hypergraph (`iblt_size`), and the number of random edges
  (number of unique keys, `num_unique`). To be more specific, for and IBLT with
  repetition = 3, the formula states that:

  Let x be the solution to "1 - e^{-x}(1+x) = num_nonempty_buckets/iblt_size",
  then num_unique is approximately iblt_size * x / (1 - e^{-x})^2 / 3.

  Based on the paper, the approximation is tight when `iblt_size` tends to
  infnity. Numerical experiments show that the approximation is fairly good
  when iblt_size > 3000 (relative error within 10%).

  Args:
    num_nonempty_buckets: the number of nonzero locations after decoding.
    iblt_size: size of the iblt table (table_size * repetition).

  Returns:
    Predicted number of unique inserted elements.
  """
  if num_nonempty_buckets <= 0 or num_nonempty_buckets > iblt_size:
    raise ValueError('num_nonempty_buckets must be in (0, iblt_size].')
  # If all hash buckets are nonempty, we use the prediction when (iblt_size - 1)
  # buckets are nonempty as an conservative approximate.
  if num_nonempty_buckets == iblt_size:
    return _predict_num_unique_rep_is_3(num_nonempty_buckets - 1, iblt_size)
  ratio = num_nonempty_buckets / iblt_size

  def func(x: float) -> float:
    return math.exp(x) * (1 - ratio) - (1 + x)

  root = scipy.optimize.fsolve(func, 100)  # 100 is an upper estimate for root
  x_sol = float(root[0])
  return iblt_size * x_sol / ((1 - math.exp(-x_sol)) ** 2) / 3


class SubsampleProcess(abc.ABC):
  """Describes the interface for subsampling tuning.

  The class abstracts methods needed for subsampling users' local datasets
  before aggregating them using the IBLT factory and the correponding tuning
  method based on server measurements.

  The class should provide the following methods needed in a (adaptive)
  subsampled IBLT process.

  1. Intialization of the subsampling parameter.
  2. Subsampling function performed on each user's local dataset.
  3. Check for whether the process is an adptive process.
  4. Tuning method for subsampling parameter based on the current subsampling
  parameter and the server measurements in the current round.

  The functions will be typically used within a `IterativeProcess` together with
  other federated computations. And the update function will be usually use the
  returned measurements as an input. Hence seperating the subsample function and
  the update function would offer more flexibility in their implementations.
  Below is an example using ThresholdSamplingProcess over multiple rounds.

  In the example, we assume each call of get_client_datasets(...) returns a list
  of client local datasets. We also assume iblt_aggregation is properly
  intialized and the next() returns measurements of the SecAggIBLT process.

  ```
  threshold_sampling = ThresholdSamplingProcess(init_param = 10.0)
  sampling_param = threshold_sampling.get_init_param()
  for round in range(num_rounds):
    client_datasets = get_client_datasets(...)
    subsampled_datasets = []
    for dataset in client_datasets:
      subsampled_dataset = threshold_sampling.subsample_fn(dataset,
                                                            sampling_param)
      subsampled_datasets.append(subsampled_dataset)
    measurements, _ = iblt_aggregation.next(..., subsampled_datasets)
    if threshold_sampling.is_process_adaptive():
      sampling_param = threshold_sampling.update(sampling_param, measurements)
  ```
  """

  @property
  @abc.abstractmethod
  def is_process_adaptive(self) -> bool:
    """Return whether the process is adaptive."""

  @abc.abstractmethod
  def update(
      self, subsampling_param_old: float, measurements: AggregationMeasurements
  ) -> float:
    """Update the sample parameter based on the return of an IBLT round.

    Args:
      subsampling_param_old: subsampling parameter for the current round.
      measurements: a `AggregationMeasurements` containing the returned
        measurements of an IBLT aggregation process in the current round.

    Returns:
      The subsampling parameter to use in the next round.
    """
    raise NotImplementedError('Not an adaptive process.')

  @abc.abstractmethod
  def get_init_param(self):
    """Returns the initial subsampling parameter."""

  @abc.abstractmethod
  def subsample_fn(
      self, client_data: tf.data.Dataset, subsampling_param: float
  ):
    """Performs subsampling at clients.

    Args:
      client_data: a `tf.data.Dataset` that represents client dataset.
      subsampling_param: subsampling parameter to use.

    Returns:
      subsampled client dataset with the same format as client_data.
    """

  @abc.abstractmethod
  def subsample_fn_with_unique_count(
      self, client_data: tf.data.Dataset, subsampling_param: float
  ):
    """Performs subsampling when client values have unique counts appended."""


class ThresholdSamplingProcess(SubsampleProcess):
  """Implements threshold sampling.

  Threshold sampling performs the following randomization of the user's local
  histogram. Let h(x) be the count of element x at user's local dataset.
  Given a threshold t, subsample_fn returns h' satisfying:

  ```
  if h(x) > t: return h(x)
  else if h(x)/t > random(): return t
  else: return 0
  ```

  The current implementation only supports histograms with nonnegative values.

  Example: if user's local histogram is {'a': 3, 'b': 5, 'c': 1}. After sampling
  with threshold 4. 'b' will always be in the subsampled dataset with count 5.
  'a' will appear with probability 3/4 and 'c' will appear with probability 1/4.
  The count of 'a' and 'c' will always be 4 if they appear.

  See more details at:
  https://nickduffield.net/download/papers/DLT05-optimal.pdf
  """

  def __init__(
      self,
      init_param: float,
      is_adaptive: bool = False,
      beta: float = 0.5,
  ):
    """Initialize the subsamping precoess.

    Args:
      init_param: initial threshold to use.
      is_adaptive: whether the process is adaptive.
      beta: the memory coefficient for the tuning process.

    Raises:
      ValueError if init_param < 1 or beta is not in (0, 1).
    """
    if init_param < 1:
      raise ValueError('Threshold must be at least 1.')
    if beta >= 1 or beta <= 0:
      raise ValueError('Beta must be in (0,1).')
    self._init_param = init_param
    self._is_adaptive = is_adaptive
    self._beta = beta

  @property
  def is_process_adaptive(self) -> bool:
    """Return whether the process is adaptive."""
    return self._is_adaptive

  def update(
      self, subsampling_param_old: float, measurements: AggregationMeasurements
  ) -> float:
    """See base class."""
    if not self.is_process_adaptive:
      raise NotImplementedError('Not an adaptive process.')
    iblt_size = measurements.iblt_size
    num_nonempty_buckets = measurements.num_nonempty_buckets
    num_recovered = measurements.num_recovered
    capacity = measurements.capacity
    if measurements.repetition != 3:
      raise ValueError('Current implementation only supports repetition = 3.')
    if num_nonempty_buckets == 0:
      num_unique_pred = num_recovered
    else:
      num_unique_pred = _predict_num_unique_rep_is_3(
          num_nonempty_buckets, iblt_size
      )
    subsampling_param = (
        self._beta * subsampling_param_old
        + (1 - self._beta)
        * (num_unique_pred / capacity)
        * subsampling_param_old
    )
    return max(subsampling_param, 1)

  def get_init_param(self):
    """Returns the initial subsampling parameter."""
    return self._init_param

  def subsample_fn(
      self,
      client_data: tf.data.Dataset,
      subsampling_param: float,
  ) -> tf.data.Dataset:
    """See base class. Raise ValueError if client data has negative counts."""

    @tf.function
    def threshold_sampling(element):
      count = element[iblt_factory.DATASET_VALUE][0]
      tf.debugging.assert_non_negative(
          count, 'Current implementation only supports positive values.'
      )
      if count >= tf.cast(subsampling_param, dtype=count.dtype):
        return element

      random_val = tf.random.uniform(
          shape=(),
          minval=0,
          maxval=tf.cast(subsampling_param, dtype=count.dtype),
          dtype=count.dtype,
      )
      if count > random_val:
        thresholded_val = tf.cast(subsampling_param, dtype=count.dtype)
      else:
        thresholded_val = tf.cast(0, dtype=count.dtype)
      return collections.OrderedDict([
          (iblt_factory.DATASET_KEY, element[iblt_factory.DATASET_KEY]),
          (
              iblt_factory.DATASET_VALUE,
              tf.cast([thresholded_val], dtype=count.dtype),
          ),
      ])
    subsampled_client_data = client_data.map(threshold_sampling)
    return subsampled_client_data.filter(
        lambda x: x[iblt_factory.DATASET_VALUE][0] > 0
    )

  def subsample_fn_with_unique_count(
      self,
      client_data: tf.data.Dataset,
      subsampling_param: float,
  ) -> tf.data.Dataset:
    """See base class. Raise ValueError if client data has negative counts."""

    @tf.function
    def threshold_sampling(element):
      count = element[iblt_factory.DATASET_VALUE][0]
      tf.debugging.assert_non_negative(
          count, 'Current implementation only supports positive values.'
      )
      if count >= tf.cast(subsampling_param, dtype=count.dtype):
        return element

      random_val = tf.random.uniform(
          shape=(),
          minval=0,
          maxval=tf.cast(subsampling_param, dtype=count.dtype),
          dtype=count.dtype,
      )
      if count > random_val:
        thresholded_val = tf.cast(subsampling_param, dtype=count.dtype)
      else:
        thresholded_val = tf.cast(0, dtype=count.dtype)
      return collections.OrderedDict([
          (iblt_factory.DATASET_KEY, element[iblt_factory.DATASET_KEY]),
          (
              iblt_factory.DATASET_VALUE,
              tf.cast([thresholded_val, 1], dtype=count.dtype),
          ),
      ])

    subsampled_client_data = client_data.map(threshold_sampling)
    return subsampled_client_data.filter(
        lambda x: x[iblt_factory.DATASET_VALUE][0] > 0
    )
