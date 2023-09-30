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
"""Shared utils for Federated Reconstruction training and evaluation."""

from typing import Callable, Iterable, Optional, Tuple

import tensorflow as tf

from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.reconstruction import model as model_lib

# Type alias for a function that takes in a TF dataset and produces two TF
# datasets. This is consumed by training and evaluation computation builders.
# The first is iterated over during reconstruction and the second is iterated
# over post-reconstruction, for both training and evaluation. This can be useful
# for e.g. splitting the dataset into disjoint halves for each stage, doing
# multiple local epochs of reconstruction/training, skipping reconstruction
# entirely, etc. See `build_dataset_split_fn` for a builder, although users can
# also specify their own `DatasetSplitFn`s (see `simple_dataset_split_fn` for an
# example).
DatasetSplitFn = Callable[[tf.data.Dataset, tf.Tensor], Tuple[tf.data.Dataset,
                                                              tf.data.Dataset]]


def simple_dataset_split_fn(
    client_dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """An example of a `DatasetSplitFn` that returns the original client data.

  Both the reconstruction data and post-reconstruction data will result from
  iterating over the same tf.data.Dataset. Note that depending on any
  preprocessing steps applied to client tf.data.Datasets, this may not produce
  exactly the same data in the same order for both reconstruction and
  post-reconstruction. For example, if
  `client_dataset.shuffle(reshuffle_each_iteration=True)` was applied,
  post-reconstruction data will be in a different order than reconstruction
  data.

  Args:
    client_dataset: `tf.data.Dataset` representing client data.

  Returns:
    A tuple of two `tf.data.Datasets`, the first to be used for reconstruction,
    the second to be used post-reconstruction.
  """
  return client_dataset, client_dataset


def build_dataset_split_fn(recon_epochs: int = 1,
                           recon_steps_max: Optional[int] = None,
                           post_recon_epochs: int = 1,
                           post_recon_steps_max: Optional[int] = None,
                           split_dataset: bool = False) -> DatasetSplitFn:
  """Builds a `DatasetSplitFn` for Federated Reconstruction training/evaluation.

  Returned `DatasetSplitFn` parameterizes training and evaluation computations
  and enables reconstruction for multiple local epochs, multiple epochs of
  post-reconstruction training, limiting the number of steps for both stages,
  and splitting client datasets into disjoint halves for each stage.

  Note that the returned function is used during both training and evaluation:
  during training, "post-reconstruction" refers to training of global variables,
  and during evaluation, it refers to calculation of metrics using reconstructed
  local variables and fixed global variables.

  Args:
    recon_epochs: The integer number of iterations over the dataset to make
      during reconstruction.
    recon_steps_max: If not None, the integer maximum number of steps (batches)
      to iterate through during reconstruction. This maximum number of steps is
      across all reconstruction iterations, i.e. it is applied after
      `recon_epochs`. If None, this has no effect.
    post_recon_epochs: The integer constant number of iterations to make over
      client data after reconstruction.
    post_recon_steps_max: If not None, the integer maximum number of steps
      (batches) to iterate through after reconstruction. This maximum number of
      steps is across all post-reconstruction iterations, i.e. it is applied
      after `post_recon_epochs`. If None, this has no effect.
    split_dataset: If True, splits `client_dataset` in half for each user, using
      even-indexed entries in reconstruction and odd-indexed entries after
      reconstruction. If False, `client_dataset` is used for both reconstruction
      and post-reconstruction, with the above arguments applied. If True,
      splitting requires that mupltiple iterations through the dataset yield the
      same ordering. For example if
      `client_dataset.shuffle(reshuffle_each_iteration=True)` has been called,
      then the split datasets may have overlap. If True, note that the dataset
      should have more than one batch for reasonable results, since the
      splitting does not occur within batches.

  Returns:
    A `SplitDatasetFn`.
  """
  # Functions for splitting dataset if needed.
  recon_condition = lambda i, entry: tf.equal(tf.math.floormod(i, 2), 0)
  post_recon_condition = lambda i, entry: tf.greater(tf.math.floormod(i, 2), 0)
  get_entry = lambda i, entry: entry

  def dataset_split_fn(
      client_dataset: tf.data.Dataset
  ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """A `DatasetSplitFn` built with the given arguments.

    Args:
      client_dataset: `tf.data.Dataset` representing client data.

    Returns:
      A tuple of two `tf.data.Datasets`, the first to be used for
      reconstruction, the second to be used post-reconstruction.
    """
    # Split dataset if needed. This assumes the dataset has a consistent
    # order across iterations.
    if split_dataset:
      recon_dataset = client_dataset.enumerate().filter(recon_condition).map(
          get_entry)
      post_recon_dataset = client_dataset.enumerate().filter(
          post_recon_condition).map(get_entry)
    else:
      recon_dataset = client_dataset
      post_recon_dataset = client_dataset

    # Apply `recon_epochs` before limiting to a maximum number of batches
    # if needed.
    recon_dataset = recon_dataset.repeat(recon_epochs)
    if recon_steps_max is not None:
      recon_dataset = recon_dataset.take(recon_steps_max)

    # Do the same for post-reconstruction.
    post_recon_dataset = post_recon_dataset.repeat(post_recon_epochs)
    if post_recon_steps_max is not None:
      post_recon_dataset = post_recon_dataset.take(post_recon_steps_max)

    return recon_dataset, post_recon_dataset

  return dataset_split_fn


def get_global_variables(model: model_lib.Model) -> model_utils.ModelWeights:
  """Gets global variables from a `Model` as `ModelWeights`."""
  return model_utils.ModelWeights(
      trainable=model.global_trainable_variables,
      non_trainable=model.global_non_trainable_variables)


def get_local_variables(model: model_lib.Model) -> model_utils.ModelWeights:
  """Gets local variables from a `Model` as `ModelWeights`."""
  return model_utils.ModelWeights(
      trainable=model.local_trainable_variables,
      non_trainable=model.local_non_trainable_variables)


def has_only_global_variables(model: model_lib.Model) -> bool:
  """Returns `True` if the model has no local variables."""
  local_variables_list = (
      list(model.local_trainable_variables) +
      list(model.local_non_trainable_variables))
  if local_variables_list:
    return False
  return True


def create_optimizer_vars(
    model: model_lib.Model,
    optimizer: tf.keras.optimizers.Optimizer) -> Iterable[tf.Variable]:
  """Applies a placeholder update to optimizer to enable getting its variables."""
  delta = tf.nest.map_structure(tf.zeros_like,
                                get_global_variables(model).trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(get_global_variables(model).trainable))
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  return optimizer.variables()
