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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A package of primitive (stateless) aggregations."""

import collections
import attr

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _validate_value_on_clients(value):
  py_typecheck.check_type(value, value_impl.Value)
  py_typecheck.check_type(value.type_signature, computation_types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError(
        '`value` argument must be a tff.Value placed at CLIENTS. Got: {!s}'
        .format(value.type_signature))


def _validate_dtype_is_min_max_compatible(dtype):
  if not (dtype.is_integer or dtype.is_floating):
    raise TypeError(
        f'Unsupported dtype. The dtype for min and max must be either an '
        f'integer or floating:. Got: {dtype}.')


def _federated_reduce_with_func(value, tf_func, zeros):
  """Applies to `tf_func` to accumulated `value`s.

  This utility provides a generic aggregation for accumulating a value and
  applying a simple aggregation (like minimum or maximum aggregations).

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`.
    tf_func: A function to be applied to the accumulated values. Must be a
      binary operation where both parameters are of type `U` and the return type
      is also `U`.
    zeros: The zero of the same type as `value` in the algebra of reduction
      operators.

  Returns:
    A representation on the `tff.SERVER` of the result of aggregating `value`.
  """
  value_type = value.type_signature.member

  @computations.tf_computation(value_type, value_type)
  def accumulate(current, value):
    return tf.nest.map_structure(tf_func, current, value)

  @computations.tf_computation(value_type)
  def report(value):
    return value

  return intrinsics.federated_aggregate(value, zeros, accumulate, accumulate,
                                        report)


def _initial_values(initial_value_fn, member_type):
  """Create a nested structure of initial values for the reduction.

  Args:
    initial_value_fn: A function that maps a tff.TensorType to a specific value
      constant for initialization.
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    A function of the result of reducing a value with no constituents.
  """

  @computations.tf_computation
  def zeros_fn():
    if member_type.is_struct():
      structure.map_structure(
          lambda v: _validate_dtype_is_min_max_compatible(v.dtype), member_type)
      return structure.map_structure(
          lambda v: tf.fill(v.shape, value=initial_value_fn(v)), member_type)
    _validate_dtype_is_min_max_compatible(member_type.dtype)
    return tf.fill(member_type.shape, value=initial_value_fn(member_type))

  return zeros_fn()


def federated_min(value):
  """Computes the minimum at `tff.SERVER` of a `value` placed at `tff.CLIENTS`.

  The minimum is computed element-wise, for each scalar and every scalar in a
  tensor contained in `value`.

  In the degenerate scenario that the `value` is aggregated over an empty set
  of `tff.CLIENTS`, the tensor constituents of the result are set to the
  maximum of the underlying numeric data type.

  Args:
    value: A value of a TFF federated type placed at the tff.CLIENTS.

  Returns:
    A representation of the min of the member constituents of `value` placed at
    `tff.SERVER`.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  # Explicit cast because v.dtype.max returns a Python constant, which could be
  # implicitly converted to a tensor of different dtype by TensorFlow.
  zeros = _initial_values(lambda v: tf.cast(v.dtype.max, v.dtype), member_type)
  return _federated_reduce_with_func(value, tf.minimum, zeros)


def federated_max(value):
  """Computes the maximum at `tff.SERVER` of a `value` placed at `tff.CLIENTS`.

  The maximum is computed element-wise, for each scalar and every scalar in a
  tensor contained in `value`.

  In the degenerate scenario that the `value` is aggregated over an empty set
  of `tff.CLIENTS`, the tensor constituents of the result are set to the
  minimum of the underlying numeric data type.

  Args:
    value: A value of a TFF federated type placed at the tff.CLIENTS.

  Returns:
    A representation of the min of the member constituents of `value` placed at
    `tff.SERVER`.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  # Explicit cast because v.dtype.min returns a Python constant, which could be
  # implicitly converted to a tensor of different dtype by TensorFlow.
  zeros = _initial_values(lambda v: tf.cast(v.dtype.min, v.dtype), member_type)
  return _federated_reduce_with_func(value, tf.maximum, zeros)


@attr.s
class _Samples(object):
  """Class representing internal sample data structure.

  The class contains two parts, `accumulators` and `rands`, that are parallel
  lists (e.g. the i-th index in one corresponds to the i-th index in the other).
  These two lists are used to sample from the accumulators with equal
  probability.
  """
  accumulators = attr.ib()
  rands = attr.ib()


def _zeros_for_sample(member_type):
  """Create an empty nested structure for the sample aggregation.

  Args:
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    A function of the result of zeros to first concatenate.
  """

  @computations.tf_computation
  def accumlator_type_fn():
    """Gets the type for the accumulators."""
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      a = structure.map_structure(
          lambda v: tf.zeros([0] + v.shape.dims, v.dtype), member_type)
      return _Samples(structure.to_odict(a, True), tf.zeros([0], tf.float32))
    if member_type.shape:
      s = [0] + member_type.shape.dims
    return _Samples(tf.zeros(s, member_type.dtype), tf.zeros([0], tf.float32))

  return accumlator_type_fn()


def _get_accumulator_type(member_type):
  """Constructs a `tff.Type` for the accumulator in sample aggregation.

  Args:
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    The `tff.StructType` associated with the accumulator. The tuple contains
    two parts, `accumulators` and `rands`, that are parallel lists (e.g. the
    i-th index in one corresponds to the i-th index in the other). These two
    lists are used to sample from the accumulators with equal probability.
  """
  # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
  if member_type.is_struct():
    a = structure.map_structure(
        lambda v: computation_types.TensorType(v.dtype, [None] + v.shape.dims),
        member_type)
    return computation_types.StructType(
        collections.OrderedDict({
            'accumulators':
                computation_types.StructType(structure.to_odict(a, True)),
            'rands':
                computation_types.TensorType(tf.float32, shape=[None]),
        }))
  return computation_types.StructType(
      collections.OrderedDict({
          'accumulators':
              computation_types.TensorType(
                  member_type.dtype, shape=[None] + member_type.shape.dims),
          'rands':
              computation_types.TensorType(tf.float32, shape=[None]),
      }))


def federated_sample(value, max_num_samples=100):
  """Aggregation to produce uniform sample of at most `max_num_samples` values.

  Each client value is assigned a random number when it is examined during each
  accumulation. Each accumulate and merge only keeps the top N values based
  on the random number. Report drops the random numbers and only returns the
  at most N values sampled from the accumulated client values using standard
  reservoir sampling (https://en.wikipedia.org/wiki/Reservoir_sampling), where
  N is user provided `max_num_samples`.

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`.
    max_num_samples: The maximum number of samples to collect from client
      values. If fewer clients than the defined max sample size participated in
      the round of computation, the actual number of samples will equal the
      number of clients in the round.

  Returns:
    At most `max_num_samples` samples of the value from the `tff.CLIENTS`.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  accumulator_type = _get_accumulator_type(member_type)
  zeros = _zeros_for_sample(member_type)

  @tf.function
  def fed_concat_expand_dims(a, b):
    b = tf.expand_dims(b, axis=0)
    return tf.concat([a, b], axis=0)

  @tf.function
  def fed_concat(a, b):
    return tf.concat([a, b], axis=0)

  @tf.function
  def fed_gather(value, indices):
    return tf.gather(value, indices)

  def apply_sampling(accumulators, rands):
    size = tf.shape(rands)[0]
    k = tf.minimum(size, max_num_samples)
    indices = tf.math.top_k(rands, k=k).indices
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      return structure.map_structure(lambda v: fed_gather(v, indices),
                                     accumulators), fed_gather(rands, indices)
    return fed_gather(accumulators, indices), fed_gather(rands, indices)

  @computations.tf_computation(accumulator_type, value.type_signature.member)
  def accumulate(current, value):
    """Accumulates samples through concatenation."""
    rands = fed_concat_expand_dims(current['rands'],
                                   tf.random.uniform(shape=()))
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      accumulators = structure.map_structure(
          fed_concat_expand_dims, _ensure_structure(current.accumulators),
          _ensure_structure(value))
    else:
      accumulators = fed_concat_expand_dims(current.accumulators, value)

    accumulators, rands = apply_sampling(accumulators, rands)
    return _Samples(accumulators, rands)

  @computations.tf_computation(accumulator_type, accumulator_type)
  def merge(a, b):
    """Merges accumulators through concatenation."""
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if accumulator_type.is_struct():
      samples = structure.map_structure(fed_concat, _ensure_structure(a),
                                        _ensure_structure(b))
    else:
      samples = fed_concat(a, b)
    accumulators, rands = apply_sampling(samples.accumulators, samples.rands)
    return _Samples(accumulators, rands)

  @computations.tf_computation(accumulator_type)
  def report(value):
    return value['accumulators']

  return intrinsics.federated_aggregate(value, zeros, accumulate, merge, report)


def _ensure_structure(obj):
  return structure.from_container(obj, True)


# Lower precision types are not supported to avoid potential hard to discover
# numerical issues in conversion to/from format compatible with secure sum.
_SECURE_QUANTIZED_SUM_ALLOWED_DTYPES = (tf.int32, tf.int64, tf.float32,
                                        tf.float64)

# The largest integer value provided to federated_secure_sum_bitwidth operator.
_SECAGG_MAX = 2**32 - 1


class BoundsDifferentTypesError(TypeError):

  def __init__(self, lower_bound, upper_bound):
    message = (f'Both lower_bound and upper_bound must be either federated '
               f'values or Python constants. Found: type(lower_bound): '
               f'{type(lower_bound)}, type(upper_bound): {type(upper_bound)}')
    super().__init__(message)


class BoundsDifferentSignaturesError(TypeError):

  def __init__(self, lower_bound, upper_bound):
    message = (f'Provided lower_bound and upper_bound must have the same '
               f'type_signature. Found: lower_bound signature: '
               f'{lower_bound.type_signature}, upper_bound signature: '
               f'{upper_bound.type_signature}.')
    super().__init__(message)


class BoundsNotPlacedAtServerError(TypeError):

  def __init__(self, placement):
    message = (f'Provided lower_bound and upper_bound must be placed at '
               f'tff.SERVER. Placement found: {placement}')
    super().__init__(message)


class StructuredBoundsTypeMismatchError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (f'If bounds are specified as structures (not scalars), the '
               f'structures must match the structure of provided client_value, '
               f'with identical dtypes, but not necessarily shapes. Found: '
               f'client_value type: {value_type}, bounds type: {bounds_type}.')
    super().__init__(message)


class ScalarBoundStructValueDTypeError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (f'If scalar bounds are provided, all parts of client_value must '
               f'be of matching dtype. Found: client_value type: {value_type}, '
               f'bounds type: {bounds_type}.')
    super().__init__(message)


class ScalarBoundSimpleValueDTypeError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (f'Bounds must have the same dtype as client_value. Found: '
               f'client_value type: {value_type}, bounds type: {bounds_type}.')
    super().__init__(message)


class UnsupportedDTypeError(TypeError):

  def __init__(self, dtype):
    message = (f'Value is of unsupported dtype {dtype}. Currently supported '
               f'types are {_SECURE_QUANTIZED_SUM_ALLOWED_DTYPES}.')
    super().__init__(message)


def _check_secure_quantized_sum_dtype(dtype):
  if dtype not in _SECURE_QUANTIZED_SUM_ALLOWED_DTYPES:
    raise UnsupportedDTypeError(dtype)


# pylint: disable=g-doc-exception
def _normalize_secure_quantized_sum_args(client_value, lower_bound,
                                         upper_bound):
  """Normalizes inputs to `secure_quantized_sum` method.

  Validates the epxected structure of arguments, as documented in the docstring
  of `secure_quantized_sum` method. The arguments provided are also returned,
  possibly normalized to meet those expectations. In particular, if
  `lower_bound` and `upper_bound` are Python constants, these are converted to
  `tff.SERVER`-placed federated values.

  Args:
    client_value: A `tff.Value` placed at `tff.CLIENTS`.
    lower_bound: The smallest possible value for `client_value` (inclusive).
      Values smaller than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `tff.Value` placed at `tff.SERVER`,
      with dtype matching that of `client_value`.
    upper_bound: The largest possible value for `client_value` (inclusive).
      Values greater than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `tff.Value` placed at `tff.SERVER`,
      with dtype matching that of `client_value`.

  Returns:
    Normalized `(client_value, lower_bound, upper_bound)` tuple.
  """
  # Validation of client_value.
  _validate_value_on_clients(client_value)
  client_value_member = client_value.type_signature.member
  if client_value.type_signature.member.is_struct():
    dtypes = [v.dtype for v in structure.flatten(client_value_member)]
    for dtype in dtypes:
      _check_secure_quantized_sum_dtype(dtype)
  else:
    dtypes = client_value_member.dtype
    _check_secure_quantized_sum_dtype(dtypes)

  # Validation of bounds.
  if isinstance(lower_bound, value_impl.Value) != isinstance(
      upper_bound, value_impl.Value):
    raise BoundsDifferentTypesError(lower_bound, upper_bound)
  elif not isinstance(lower_bound, value_impl.Value):
    # Normalization of bounds to federated values.
    lower_bound = intrinsics.federated_value(lower_bound, placements.SERVER)
    upper_bound = intrinsics.federated_value(upper_bound, placements.SERVER)

  if lower_bound.type_signature != upper_bound.type_signature:
    raise BoundsDifferentSignaturesError(lower_bound, upper_bound)
  # The remaining type checks only use lower_bound as the upper_bound has
  # itendical type_signature.
  if lower_bound.type_signature.placement != placements.SERVER:
    raise BoundsNotPlacedAtServerError(lower_bound.type_signature.placement)

  # Validation of client_value and bounds compatibility.
  bound_member = lower_bound.type_signature.member
  if bound_member.is_struct():
    if not client_value_member.is_struct() or (structure.map_structure(
        lambda v: v.dtype, bound_member) != structure.map_structure(
            lambda v: v.dtype, client_value_member)):
      raise StructuredBoundsTypeMismatchError(client_value_member, bound_member)
  else:
    # If bounds are scalar, must be compatible with all tensors in client_value.
    if client_value_member.is_struct():
      if len(set(dtypes)) > 1 or (bound_member.dtype != dtypes[0]):
        raise ScalarBoundStructValueDTypeError(client_value_member,
                                               bound_member)
    else:
      if bound_member.dtype != client_value_member.dtype:
        raise ScalarBoundSimpleValueDTypeError(client_value_member,
                                               bound_member)

  return client_value, lower_bound, upper_bound


@tf.function
def _client_tensor_shift_for_secure_sum(value, lower_bound, upper_bound):
  """Mapping to be applied to every tensor before secure sum.

  This operation is performed on `tff.CLIENTS` to prepare values to format
  compatible with `tff.federated_secure_sum_bitwidth` operator.

  This clips elements of `value` to `[lower_bound, upper_bound]`, shifts and
  scales it to range `[0, 2**32-1]` and casts it to `tf.int64`. The specific
  operation depends on dtype of `value`.

  Args:
    value: A Tensor to be shifted for compatibility with
      `federated_secure_sum_bitwidth`.
    lower_bound: The smallest value expected in `value`.
    upper_bound: The largest value expected in `value`.

  Returns:
    Shifted value of dtype `tf.int64`.
  """
  tf.Assert(lower_bound <= upper_bound, [lower_bound, upper_bound])
  if value.dtype == tf.int32:
    clipped_val = tf.clip_by_value(value, lower_bound, upper_bound)
    # Cast BEFORE shift in order to avoid overflow if full int32 range is used.
    return tf.cast(clipped_val, tf.int64) - tf.cast(lower_bound, tf.int64)
  elif value.dtype == tf.int64:
    clipped_val = tf.clip_by_value(value, lower_bound, upper_bound)
    range_span = upper_bound - lower_bound
    scale_factor = tf.math.floordiv(range_span, _SECAGG_MAX) + 1
    shifted_value = tf.cond(
        scale_factor > 1,
        lambda: tf.math.floordiv(clipped_val - lower_bound, scale_factor),
        lambda: clipped_val - lower_bound)
    return shifted_value
  else:
    # This should be ensured earlier and thus not user-facing.
    assert value.dtype in [tf.float32, tf.float64]
    clipped_value = tf.clip_by_value(value, lower_bound, upper_bound)
    # Prevent NaNs if `lower_bound` and `upper_bound` are the same.
    scale_factor = tf.math.divide_no_nan(
        tf.constant(_SECAGG_MAX, tf.float64),
        tf.cast(upper_bound - lower_bound, tf.float64))
    scaled_value = tf.cast(clipped_value, tf.float64) * scale_factor
    # Perform deterministic rounding here, which may introduce bias as every
    # value may be rounded in the same direction for some input data.
    rounded_value = tf.saturate_cast(tf.round(scaled_value), tf.int64)
    # Perform shift in integer space to minimize float precision errors.
    shifted_value = rounded_value - tf.saturate_cast(
        tf.round(tf.cast(lower_bound, tf.float64) * scale_factor), tf.int64)
    # Clip to expected range in case of numerical stability issues.
    quantized_value = tf.clip_by_value(shifted_value,
                                       tf.constant(0, dtype=tf.int64),
                                       tf.constant(_SECAGG_MAX, dtype=tf.int64))
    return quantized_value


@tf.function
def _server_tensor_shift_for_secure_sum(num_summands, value, lower_bound,
                                        upper_bound, output_dtype):
  """Mapping to be applied to every tensor after secure sum.

  This operation is performed on `tff.SERVER` to dequantize outputs of the
  `tff.federated_secure_sum_bitwidth` operator.

  It is reverse of `_client_tensor_shift_for_secure_sum` taking into account
  that `num_summands` elements were summed, so the inverse shift needs to be
  appropriately scaled.

  Args:
    num_summands: The number of summands that formed `value`.
    value: A summed Tensor to be shifted to original representation.
    lower_bound: The smallest value expected in `value` before it was summed.
    upper_bound: The largest value expected in `value` before it was summed.
    output_dtype: The dtype of value after being shifted.

  Returns:
    Shifted value of dtype `output_dtype`.
  """
  # Ensure summed `value` is within the expected range given `num_summands`.
  min_valid_value = tf.constant(0, tf.int64)
  # Cast to tf.int64 before multiplication to prevent overflow.
  max_valid_value = tf.constant(_SECAGG_MAX, tf.int64) * tf.cast(
      num_summands, tf.int64)
  tf.Assert(
      tf.math.logical_and(
          tf.math.reduce_min(value) >= min_valid_value,
          tf.math.reduce_max(value) <= max_valid_value),
      [value, min_valid_value, max_valid_value])

  if output_dtype == tf.int32:
    value = value + tf.cast(num_summands, tf.int64) * tf.cast(
        lower_bound, tf.int64)
  elif output_dtype == tf.int64:
    range_span = upper_bound - lower_bound
    scale_factor = tf.math.floordiv(range_span, _SECAGG_MAX) + 1
    num_summands = tf.cast(num_summands, tf.int64)
    value = tf.cond(scale_factor > 1,
                    lambda: value * scale_factor + num_summands * lower_bound,
                    lambda: value + num_summands * lower_bound)
  else:
    # This should be ensured earlier and thus not user-facing.
    assert output_dtype in [tf.float32, tf.float64]
    # Use exactly the same `scale_factor` as during client quantization so that
    # float precision errors (which are deterministic) cancel out. This ensures
    # that the sum of [0] is exactly 0 for any clipping range.
    scale_factor = tf.math.divide_no_nan(
        tf.constant(_SECAGG_MAX, tf.float64),
        tf.cast(upper_bound - lower_bound, tf.float64))
    # Scale the shift by `num_summands` as an integer to prevent additional
    # float precision errors for multiple summands. This also ensures that the
    # sum of [0] * num_summands is exactly 0 for any clipping range.
    value = value + tf.saturate_cast(
        tf.round(tf.cast(lower_bound, tf.float64) * scale_factor),
        tf.int64) * tf.cast(num_summands, tf.int64)
    value = tf.cast(value, tf.float64)
    value = value * (
        tf.cast(upper_bound - lower_bound, tf.float64) / _SECAGG_MAX)
    # If `lower_bound` and `upper_bound` are the same, the above shift had no
    # effect since `scale_factor` is 0. Shift here instead.
    shifted_value = value + tf.cast(num_summands, tf.float64) * tf.cast(
        lower_bound, tf.float64)
    value = tf.cond(
        tf.equal(lower_bound, upper_bound), lambda: shifted_value,
        lambda: value)

  return tf.cast(value, output_dtype)


def secure_quantized_sum(client_value, lower_bound, upper_bound):
  """Quantizes and sums values securely.

  Provided `client_value` can be either a Tensor or a nested structure of
  Tensors. If it is a nested structure, `lower_bound` and `upper_bound` must be
  either both scalars, or both have the same structure as `client_value`, with
  each element being a scalar, representing the bounds to be used for each
  corresponding Tensor in `client_value`.

  This method converts each Tensor in provided `client_value` to appropriate
  format and uses the `tff.federated_secure_sum_bitwidth` operator to realize
  the sum.

  The dtype of Tensors in provided `client_value` can be one of `[tf.int32,
  tf.int64, tf.float32, tf.float64]`.

  If the dtype of `client_value` is `tf.int32` or `tf.int64`, the summation is
  possibly exact, depending on `lower_bound` and `upper_bound`: In the case that
  `upper_bound - lower_bound < 2**32`, the summation will be exact. If it is
  not, `client_value` will be quantized to precision of 32 bits, so the worst
  case error introduced for the value of each client will be approximately
  `(upper_bound - lower_bound) / 2**32`. Deterministic rounding to nearest value
  is used in such cases.

  If the dtype of `client_value` is `tf.float32` or `tf.float64`, the summation
  is generally *not* accurate up to full floating point precision. Instead, the
  values are first clipped to the `[lower_bound, upper_bound]` range. These
  values are then uniformly quantized to 32 bit resolution, using deterministic
  rounding to round the values to the quantization points. Rounding happens
  roughly as follows (implementation is a bit more complex to mitigate numerical
  stability issues):

  ```
  values = tf.round(
      (client_value - lower_bound) * ((2**32 - 1) / (upper_bound - lower_bound))
  ```

  After summation, the inverse operation if performed, so the return value
  is of the same dtype as the input `client_value`.

  In terms of accuracy, it is safe to assume accuracy within 7-8 significant
  digits for `tf.float32` inputs, and 8-9 significant digits for `tf.float64`
  inputs, where the significant digits refer to precision relative to the range
  of the provided bounds. Thus, these bounds should not be set extremely wide.
  Accuracy losses arise due to (1) quantization within the given clipping range,
  (2) float precision of final outputs (e.g. `tf.float32` has 23 bits in its
  mantissa), and (3) precision losses that arise in doing math on `tf.float32`
  and `tf.float64` inputs.

  As a concrete example, if the range is `+/- 1000`, errors up to `1e-4` per
  element should be expected for `tf.float32` and up to `1e-5` for `tf.float64`.

  Args:
    client_value: A `tff.Value` placed at `tff.CLIENTS`.
    lower_bound: The smallest possible value for `client_value` (inclusive).
      Values smaller than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `tff.Value` placed at `tff.SERVER`,
      with dtype matching that of `client_value`.
    upper_bound: The largest possible value for `client_value` (inclusive).
      Values greater than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `tff.Value` placed at `tff.SERVER`,
      with dtype matching that of `client_value`.

  Returns:
    Summed `client_value` placed at `tff.SERVER`, of the same dtype as
    `client_value`.

  Raises:
    TypeError (or its subclasses): If input arguments do not satisfy the type
      constraints specified above.
  """

  # Possibly converts Python constants to federated values.
  client_value, lower_bound, upper_bound = _normalize_secure_quantized_sum_args(
      client_value, lower_bound, upper_bound)

  # This object is used during decoration of the `client_shift` method, and the
  # value stored in this mutable container is used during decoration of the
  # `server_shift` method. The reason for this is that we cannot currently get
  # the needed information out of `client_value.type_signature.member` as we
  # need both the `TensorType` information as well as the Python container
  # attached to them.
  temp_box = []

  # These tf_computations assume the inputs were already validated. In
  # particular, that lower_bnd and upper_bnd have the same structure, and if not
  # scalar, the structure matches the structure of value.
  @computations.tf_computation()
  def client_shift(value, lower_bnd, upper_bnd):
    assert not temp_box
    temp_box.append(tf.nest.map_structure(lambda v: v.dtype, value))
    fn = _client_tensor_shift_for_secure_sum
    if tf.is_tensor(lower_bnd):
      return tf.nest.map_structure(lambda v: fn(v, lower_bnd, upper_bnd), value)
    else:
      return tf.nest.map_structure(fn, value, lower_bnd, upper_bnd)

  @computations.tf_computation()
  def server_shift(value, lower_bnd, upper_bnd, summands):
    fn = _server_tensor_shift_for_secure_sum
    if tf.is_tensor(lower_bnd):
      return tf.nest.map_structure(
          lambda v, dtype: fn(summands, v, lower_bnd, upper_bnd, dtype), value,
          temp_box[0])
    else:
      return tf.nest.map_structure(lambda *args: fn(summands, *args), value,
                                   lower_bnd, upper_bnd, temp_box[0])

  client_one = intrinsics.federated_value(1, placements.CLIENTS)

  # Orchestration.
  client_lower_bound = intrinsics.federated_broadcast(lower_bound)
  client_upper_bound = intrinsics.federated_broadcast(upper_bound)

  value = intrinsics.federated_map(
      client_shift, (client_value, client_lower_bound, client_upper_bound))
  num_summands = intrinsics.federated_secure_sum_bitwidth(
      client_one, bitwidth=1)

  secagg_value_type = value.type_signature.member
  assert secagg_value_type.is_tensor() or secagg_value_type.is_struct()
  if secagg_value_type.is_tensor():
    bitwidths = 32
  else:
    bitwidths = structure.map_structure(lambda t: 32, secagg_value_type)

  value = intrinsics.federated_secure_sum_bitwidth(value, bitwidth=bitwidths)
  value = intrinsics.federated_map(
      server_shift, (value, lower_bound, upper_bound, num_summands))
  return value
