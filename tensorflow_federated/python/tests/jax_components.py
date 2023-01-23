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
"""Experimental federated learning components for JAX."""

import collections
import operator

import jax
from jax import numpy as jnp
import tensorflow_federated as tff

# TODO(b/175888145): Evolve this to reach parity with TensorFlow-specific helper
# and eventually unify the two.


def build_jax_federated_averaging_process(batch_type, model_type, loss_fn,
                                          step_size):
  """Constructs an iterative process that implements simple federated averaging.

  Args:
    batch_type: An instance of `tff.Type` that represents the type of a single
      batch of data to use for training. This type should be constructed with
      standard Python containers (such as `collections.OrderedDict`) of the sort
      that are expected as parameters to `loss_fn`.
    model_type: An instance of `tff.Type` that represents the type of the model.
      Similarly to `batch_size`, this type should be constructed with standard
      Python containers (such as `collections.OrderedDict`) of the sort that are
      expected as parameters to `loss_fn`.
    loss_fn: A loss function for the model. Must be a Python function that takes
      two parameters, one of them being the model, and the other being a single
      batch of data (with types matching `batch_type` and `model_type`).
    step_size: The step size to use during training (an `np.float32`).

  Returns:
    An instance of `tff.templates.IterativeProcess` that implements federated
    training in JAX.
  """
  batch_type = tff.to_type(batch_type)
  model_type = tff.to_type(model_type)

  # py_typecheck.check_type(batch_type, computation_types.Type)
  # py_typecheck.check_type(model_type, computation_types.Type)
  # py_typecheck.check_callable(loss_fn)
  # py_typecheck.check_type(step_size, np.float)

  @tff.jax_computation
  def _create_zero_model():
    def _tensor_zeros(tensor_type):
      return jnp.zeros(
          shape=tensor_type.shape.dims, dtype=tensor_type.dtype.as_numpy_dtype
      )

    return tff.structure_from_tensor_type_tree(_tensor_zeros, model_type)

  @tff.jax_computation
  def create_zero_float():
    return jnp.zeros(shape=[])

  @tff.jax_computation
  def accumulate(accumulator, value):
    return jax.tree_util.tree_map(operator.add, accumulator, (value, 1))

  @tff.jax_computation
  def merge(a, b):
    return jax.tree_util.tree_map(operator.add, a, b)

  @tff.jax_computation
  def report(summed_model_update, total_count):
    normalize_fn = lambda weight: weight / total_count
    return jax.tree_util.tree_map(normalize_fn, summed_model_update)

  @tff.federated_computation
  def mean_as_aggregate(arg):
    return tff.federated_aggregate(
        arg,
        (_create_zero_model(), create_zero_float()),
        accumulate,
        merge,
        report,
    )

  @tff.federated_computation
  def _create_zero_model_on_server():
    return tff.federated_eval(_create_zero_model, tff.SERVER)

  @tff.jax_computation(model_type, batch_type)
  def _train_on_one_batch(model, batch):

    def _apply_update(model_param, param_delta):
      return model_param - step_size * param_delta

    grads = jax.grad(loss_fn)(model, batch)
    return jax.tree_util.tree_map(_apply_update, model, grads)

  local_dataset_type = tff.SequenceType(batch_type)

  @tff.federated_computation(model_type, local_dataset_type)
  def _train_on_one_client(model, batches):
    return tff.sequence_reduce(batches, model, _train_on_one_batch)

  @tff.federated_computation(
      tff.types.at_server(model_type), tff.types.at_clients(local_dataset_type)
  )
  def _train_one_round(model, federated_data):
    locally_trained_models = tff.federated_map(
        _train_on_one_client,
        collections.OrderedDict(
            model=tff.federated_broadcast(model), batches=federated_data
        ),
    )
    # We hand-implement an unweighted federated mean as a TFF aggregate, since
    # we cant effectively lower to jax yet.
    return mean_as_aggregate(locally_trained_models)

  return tff.templates.IterativeProcess(
      initialize_fn=_create_zero_model_on_server, next_fn=_train_one_round)
