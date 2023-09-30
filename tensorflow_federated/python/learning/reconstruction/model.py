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
"""Abstractions for Federated Reconstruction Models."""

import abc

import attr


@attr.s(eq=False, frozen=True, slots=True)
class BatchOutput(object):
  """A structure that holds the output of a `tff.learning.reconstruction.Model`.

  Note: All fields are optional (may be None).

  -   `predictions`: Tensor of predictions on the examples.
  -   `labels`: Tensor of labels for the examples.
  -   `num_examples`: tf.int32 scalar number of examples seen in the batch.
  """
  predictions = attr.ib()
  labels = attr.ib()
  num_examples = attr.ib()


class Model(object, metaclass=abc.ABCMeta):
  """Represents a reconstruction model for use in Tensorflow Federated.

  `tff.learning.reconstruction.Model`s are used to train models that reconstruct
  a set of their variables on device, never sharing those variables with the
  server.

  Each `tff.learning.reconstruction.Model` will work on a set of `tf.Variables`,
  and each method should be a computation that can be implemented as a
  `tf.function`; this implies the class should essentially be stateless from a
  Python perspective, as each method will generally only be traced once (per set
  of arguments) to create the corresponding TensorFlow graph functions. Thus,
  `tff.learning.reconstruction.Model` instances should behave as expected in
  both eager and graph (TF 1.0) usage.

  In general, `tf.Variables` may be either:
    * Weights, the variables needed to make predictions with the model.
    * Local variables, e.g. to accumulate aggregated metrics across
      calls to forward_pass.

  The weights can be broken down into:
    * Global variables: Variables that are allowed to be aggregated on the
      server.
    * Local variables: Variables that cannot leave the device.

  Furthermore, both of these types of variables can be:
    * Trainable variables: These can and should be trained using gradient-based
      methods.
    * Non-trainable variables: Could include fixed pre-trained layers or static
      model data.

  These variables are provided via:
    * `global_trainable_variables`
    * `global_non_trainable_variables`
    * `local_trainable_variables`
    * `local_non_trainable_variables`

  properties, and must be initialized by the user of the
  `tff.learning.reconstruction.Model`.

  While training a reconstruction model, global trainable variables will
  generally be provided by the server. Local trainable variables will then be
  reconstructed locally. Updates to the global trainable variables will be sent
  back to the server. Local variables are not transmitted.

  All `tf.Variables` should be introduced in `__init__`; this could move to a
  `build` method more inline with Keras (see
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) in
  the future.
  """

  @abc.abstractproperty
  def global_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def global_non_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def local_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def local_non_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def input_spec(self):
    """The type specification of the `batch_input` parameter for `forward_pass`.

    A nested structure of `tf.TensorSpec` objects, that matches the structure of
    arguments that will be passed as the `batch_input` argument of
    `forward_pass`. The tensors must include a batch dimension as the first
    dimension, but the batch dimension may be undefined.
    """
    pass

  @abc.abstractmethod
  def forward_pass(self, batch_input, training=True):
    """Runs the forward pass and returns results.

    This method should not modify any variables that are part of the model
    parameters, that is, variables that influence the predictions. Rather, this
    is done by the training loop.

    Args:
      batch_input: a nested structure that matches the structure of
        `Model.input_spec` and each tensor in `batch_input` satisfies
        `tf.TensorSpec.is_compatible_with()` for the corresponding
        `tf.TensorSpec` in `Model.input_spec`.
      training: If `True`, run the training forward pass, otherwise, run in
        evaluation mode. The semantics are generally the same as the `training`
        argument to `keras.Model.__call__`; this might e.g. influence how
        dropout or batch normalization is handled.

    Returns:
      A `BatchOutput` object.
    """
    pass
