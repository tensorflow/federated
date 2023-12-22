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
"""Factory for subsampling client strings before aggregation via IBLT."""

import collections

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import subsample_process
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process


class SubsampledIbltFactory(factory.UnweightedAggregationFactory):
  """Factory for subsampling client data before aggregation."""

  def __init__(
      self,
      inner_iblt_agg: iblt_factory.IbltFactory,
      sampling_process: subsample_process.SubsampleProcess,
      unique_counts: bool = False,
  ):
    """Initializes ClientPreprocessingAggregationFactory.

    Args:
      inner_iblt_agg: An instance of IbltFactory.
      sampling_process: An instance of SubsampleProcess specifying parameters
        and methods related to dataset subsampling at client side.
      unique_counts: Whether the input dataset contain unique counts in its
        values, if yes, if value will be of form `[count, 1]`.
    """
    self.inner_iblt_agg = inner_iblt_agg
    self.sampling_process = sampling_process
    self.unique_counts = unique_counts

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    expected_value_type = computation_types.SequenceType(
        collections.OrderedDict([
            (iblt_factory.DATASET_KEY, tf.string),
            (
                iblt_factory.DATASET_VALUE,
                computation_types.TensorType(shape=[None], dtype=tf.int64),
            ),
        ])
    )

    if not expected_value_type.is_assignable_from(value_type):
      raise ValueError(
          'value_shape must be compatible with '
          f'{expected_value_type}. Found {value_type} instead.'
      )

    if self.sampling_process.is_process_adaptive:
      raise ValueError(
          'Current implementation only support nonadaptive process.'
      )

    subsample_param = self.sampling_process.get_init_param()
    if self.unique_counts:
      subsample_fn = self.sampling_process.subsample_fn_with_unique_count
    else:
      subsample_fn = self.sampling_process.subsample_fn

    @tensorflow_computation.tf_computation(value_type)
    @tf.function
    def subsample(client_data):
      return subsample_fn(client_data, subsample_param)

    inner_process = self.inner_iblt_agg.create(subsample.type_signature.result)

    @federated_computation.federated_computation(
        inner_process.initialize.type_signature.result,
        computation_types.at_clients(value_type),
    )
    def next_fn(state, client_data):
      preprocessed = intrinsics.federated_map(subsample, client_data)
      return inner_process.next(state, preprocessed)

    return aggregation_process.AggregationProcess(
        inner_process.initialize, next_fn
    )
