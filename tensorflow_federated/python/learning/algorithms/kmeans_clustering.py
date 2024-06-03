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
"""A federated version of the mini-batch k-means algorithm.

Based on the paper:

"Web-Scale K-Means Clustering" by D. Sculley. See
https://dl.acm.org/doi/10.1145/1772690.1772862 for the full paper.
"""

import collections
from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process

_INDEX_DTYPE = np.int32
_POINT_DTYPE = np.float32
_WEIGHT_DTYPE = np.float32
_MILLIS_PER_SECOND = 100000.0


@tf.function
def _find_closest_centroid(centroids: tf.Tensor, point: tf.Tensor):
  """Find the centroid closest to a given point.

  Note that `centroids` and `point` must have matching data types that are
  compatible with subtraction.

  Args:
    centroids: A tensor containing the k-means centroids, indexed by the first
      axis.
    point: A tensor whose shape matches `centroids.shape[1:]`.

  Returns:
     An integer representing the row of `centroids` closes to `point`.
  """
  num_axes = tf.rank(centroids)
  data_axes = tf.range(1, limit=num_axes)
  shifted_centroids = centroids - point
  square_distances = tf.math.reduce_sum(
      shifted_centroids * shifted_centroids, axis=data_axes
  )
  return tf.math.argmin(square_distances, axis=0, output_type=_INDEX_DTYPE)


@tf.function
def _compute_kmeans_step(centroids: tf.Tensor, data: tf.data.Dataset):
  """Performs a k-means step on a dataset.

  This method finds, for each point in `data`, the closest centroid in
  `centroids`. It returns a structure `tff.learning.templates.ClientResult`
  whose `update` attribute is a tuple `(cluster_sums, cluster_weights)`. Here,
  `cluster_sums` is a tensor of shape matching `centroids`, where
  `cluster_sums[i, :]` is the sum of all points closest to the i-th centroid,
  and `cluster_weights` is a `(num_centroids,)` dimensional tensor whose i-th
  component is the number of points closest to the i-th centroid. The
  `ClientResult.update_weight` attribute is left empty.

  Args:
    centroids: A `tf.Tensor` of centroids, indexed by the first axis.
    data: A `tf.data.Dataset` of points, each of which has shape matching that
      of `centroids.shape[1:]`.

  Returns:
   A `tff.learning.templates.ClientResult`.
  """
  cluster_sums = tf.zeros_like(centroids)
  cluster_weights = tf.zeros(shape=(centroids.shape[0],), dtype=_WEIGHT_DTYPE)
  num_examples = tf.constant(0, dtype=_WEIGHT_DTYPE)

  def reduce_fn(state, point):
    cluster_sums, cluster_weights, num_examples = state
    closest_centroid = _find_closest_centroid(centroids, point)
    scatter_index = [[closest_centroid]]
    cluster_sums = tf.tensor_scatter_nd_add(
        cluster_sums, scatter_index, tf.expand_dims(point, axis=0)
    )
    cluster_weights = tf.tensor_scatter_nd_add(
        cluster_weights, scatter_index, [1]
    )
    num_examples += 1
    return cluster_sums, cluster_weights, num_examples

  cluster_sums, cluster_weights, num_examples = data.reduce(
      initial_state=(cluster_sums, cluster_weights, num_examples),
      reduce_func=reduce_fn,
  )

  stat_output = collections.OrderedDict(num_examples=num_examples)
  return (
      client_works.ClientResult(
          update=(cluster_sums, cluster_weights), update_weight=()
      ),
      stat_output,
  )


def _build_kmeans_client_work(
    centroids_type: computation_types.TensorType,
    data_type: computation_types.SequenceType,
):
  """Creates a `tff.learning.templates.ClientWorkProcess` for k-means."""

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @tensorflow_computation.tf_computation(centroids_type, data_type)
  def client_update(centroids, client_data):
    return _compute_kmeans_step(centroids, client_data)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(centroids_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, cluster_centers, client_data):
    client_result, stat_output = intrinsics.federated_map(
        client_update, (cluster_centers, client_data)
    )
    stat_metrics = intrinsics.federated_sum(stat_output)
    return measured_process.MeasuredProcessOutput(
        state, client_result, stat_metrics
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


@tf.function
def _update_centroids(
    current_centroids, current_weights, new_cluster_sums, new_weights
):
  """Computes a weighted combination of previous and new centroids.

  Args:
    current_centroids: A tensor whose rows represent the current centroids.
    current_weights: A tensor of float weights associated to each cluster.
    new_cluster_sums: A tensor of shape matching `current_centroids`
      representing a sum of points newly associated to each centroid.
    new_weights: A tensor of float weights representing the number of new
      samples assigned to each centroid.

  Returns:
    A tuple `updated_centroids`, `updated_weights`, where `updated_centroids`
      represents the updated centroids, and `updated_weights` represents the
      updated weights.
  """
  total_weights = current_weights + new_weights

  # We convert the weights to floats in order to do division
  current_scale = tf.math.divide_no_nan(current_weights, total_weights)
  new_weights_indicator = tf.cast(tf.math.greater(new_weights, 0), _POINT_DTYPE)
  new_scale = tf.math.divide_no_nan(new_weights_indicator, total_weights)

  # We reshape so that we can do element-wise multiplication
  num_centroids = current_centroids.shape[0]
  num_dims_to_add = tf.rank(current_centroids) - 1
  scale_shape = tf.concat(
      [[num_centroids], tf.ones((num_dims_to_add,), dtype=tf.int32)], axis=0
  )
  current_scale = tf.reshape(current_scale, scale_shape)
  new_scale = tf.reshape(new_scale, scale_shape)

  # Compute the updated centroids as a weighted average of current and new.
  updated_centroids = (
      current_scale * current_centroids + new_scale * new_cluster_sums
  )
  return updated_centroids, total_weights


def _build_kmeans_finalizer(
    centroids_type: computation_types.Type, num_centroids: int
):
  """Builds a `tff.learning.templates.FinalizerProcess` for k-means."""

  @tensorflow_computation.tf_computation
  def initialize_weights():
    return tf.ones((num_centroids,), dtype=_WEIGHT_DTYPE)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_eval(initialize_weights, placements.SERVER)

  weights_type = initialize_weights.type_signature.result

  @tensorflow_computation.tf_computation(
      centroids_type, weights_type, centroids_type, weights_type
  )
  def server_update_tf(
      current_centroids, current_weights, new_centroid_sums, new_weights
  ):
    return _update_centroids(
        current_centroids, current_weights, new_centroid_sums, new_weights
    )

  summed_updates_type = computation_types.FederatedType(
      computation_types.to_type((centroids_type, weights_type)),
      placements.SERVER,
  )

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(centroids_type, placements.SERVER),
      summed_updates_type,
  )
  def next_fn(state, current_centroids, summed_updates):
    new_centroid_sums, new_weights = summed_updates
    updated_centroids, updated_weights = intrinsics.federated_map(
        server_update_tf,
        (current_centroids, state, new_centroid_sums, new_weights),
    )
    empty_measurements = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        updated_weights, updated_centroids, empty_measurements
    )

  return finalizers.FinalizerProcess(init_fn, next_fn)


def build_fed_kmeans(
    num_clusters: int,
    data_shape: tuple[int, ...],
    random_seed: Optional[tuple[int, int]] = None,
    distributor: Optional[distributors.DistributionProcess] = None,
    sum_aggregator: Optional[factory.UnweightedAggregationFactory] = None,
) -> learning_process.LearningProcess:
  """Builds a learning process for federated k-means clustering.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated k-means clustering. Specifically, this performs mini-batch k-means
  clustering. Note that mini-batch k-means only processes a mini-batch of the
  data at each round, and updates clusters in a weighted manner based on how
  many points in the mini-batch were assigned to each cluster. In the federated
  version, clients do the assignment of each of their point locally, and the
  server updates the clusters. Conceptually, the "mini-batch" being used is the
  union of all client datasets involved in a given round.

  The learning process has the following methods inherited from
  `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `LearningAlgorithmState` representing the
      initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches the output of `initialize`
      and `{B*}@CLIENTS` represents the client datasets. The output `L` is a
      `tff.learning.templates.LearningProcessOutput` containing the state `S`
      and metrics computed during training.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> W)`,
      where `W` represents the current k-means centroids.
  *   `set_model_weights`: A `tff.Computation` with type signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` a new set of k-means centroids.

  Here, `S` is a `tff.learning.templates.LearningAlgorithmState`. The centroids
  `W` is a tensor representing the current centroids, and is of shape
  `(num_clusters,) + data_shape`. The datasets `{B*}` must have elements of
  shape `data_shape`, and not employ batching.

  The centroids are updated at each round by assigning all clients' points to
  the nearest centroid, and then summing these points according to these
  centroids. The centroids are then updated at the server based on these points.
  To do so, we keep track of how many points have been assigned to each centroid
  overall, as a float tensor of shape `(num_clusters,)`. This information can
  be found in `state.finalizer`. Note that we begin with a "pseudo-count" of 1,
  in order to ensure that the centroids do not collapse to zero.

  Args:
    num_clusters: The number of clusters to use.
    data_shape: A tuple of integers specifying the shape of each data point.
      Note that this data shape should be unbatched, as this algorithm does not
      currently support batched data points.
    random_seed: A tuple of two integers used to seed the initialization phase.
    distributor: An optional `tff.learning.templates.DistributionProcess` that
      broadcasts the centroids on the server to the clients. If set to `None`,
      the distributor is constructed via
      `tff.learning.templates.build_broadcast_process`.
    sum_aggregator: An optional `tff.aggregators.UnweightedAggregationFactory`
      used to sum updates across clients. If `None`, we use
      `tff.aggregators.SumFactory`.

  Returns:
    A `LearningProcess`.
  """
  centroids_shape = (num_clusters,) + data_shape

  if not random_seed:
    random_seed = (
        tf.cast(tf.timestamp() * _MILLIS_PER_SECOND, tf.int64).numpy(),
        0,
    )

  @tensorflow_computation.tf_computation
  def initialize_centers():
    return tf.random.stateless_normal(
        centroids_shape, random_seed, dtype=_POINT_DTYPE
    )

  centroids_type = computation_types.TensorType(_POINT_DTYPE, centroids_shape)
  weights_type = computation_types.TensorType(
      _WEIGHT_DTYPE, shape=(num_clusters,)
  )
  point_type = computation_types.TensorType(_POINT_DTYPE, shape=data_shape)
  data_type = computation_types.SequenceType(point_type)

  if distributor is None:
    distributor = distributors.build_broadcast_process(centroids_type)

  client_work = _build_kmeans_client_work(centroids_type, data_type)

  if sum_aggregator is None:
    sum_aggregator = sum_factory.SumFactory()
  # We wrap the sum factory as a weighted aggregator for compatibility with
  # the learning process composer.
  weighted_aggregator = factory_utils.as_weighted_aggregator(sum_aggregator)
  value_type = computation_types.to_type((centroids_type, weights_type))
  aggregator = weighted_aggregator.create(
      value_type,
      computation_types.to_type(()),  # pytype: disable=wrong-arg-types
  )

  finalizer = _build_kmeans_finalizer(centroids_type, num_clusters)

  return composers.compose_learning_process(
      initialize_centers, distributor, client_work, aggregator, finalizer
  )
