# Copyright 2024, The TensorFlow Federated Authors.
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
"""Utilities for working with arrays."""

from typing import Optional, Union

import numpy as np

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import dtype_utils

# Array is the Python representation of the `Array` protobuf, and is the native
# representation of an array.
Array = Union[
    # Python types
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    # Numpy types
    np.generic,
    np.ndarray,
]


def from_proto(array_pb: array_pb2.Array) -> Array:
  """Returns an `Array` for the `array_pb`."""
  dtype = dtype_utils.from_proto(array_pb.dtype)
  shape = array_shape.from_proto(array_pb.shape)

  if dtype is np.bool_:
    value = array_pb.bool_list.value
  elif dtype is np.int8:
    value = array_pb.int8_list.value
  elif dtype is np.int16:
    value = array_pb.int16_list.value
  elif dtype is np.int32:
    value = array_pb.int32_list.value
  elif dtype is np.int64:
    value = array_pb.int64_list.value
  elif dtype is np.uint8:
    value = array_pb.uint8_list.value
  elif dtype is np.uint16:
    value = array_pb.uint16_list.value
  elif dtype is np.uint32:
    value = array_pb.uint32_list.value
  elif dtype is np.uint64:
    value = array_pb.uint64_list.value
  elif dtype is np.float16:
    value = array_pb.float16_list.value
    # Values of dtype `np.float16` are packed to and unpacked from a protobuf
    # field of type `int32` using the following logic in order to maintain
    # compatibility with how other external environments (e.g., TensorFlow, JAX)
    # represent values of `np.float16`.
    value = np.array(value, np.uint16).astype(np.float16).tolist()
  elif dtype is np.float32:
    value = array_pb.float32_list.value
  elif dtype is np.float64:
    value = array_pb.float64_list.value
  elif dtype is np.complex64:
    if len(array_pb.complex64_list.value) % 2 != 0:
      raise ValueError(
          'Expected the number of complex values to be even, one real and one'
          ' imaginary part for each complex value.'
      )
    value = iter(array_pb.complex64_list.value)
    value = [complex(real, imag) for real, imag in zip(value, value)]
  elif dtype is np.complex128:
    if len(array_pb.complex128_list.value) % 2 != 0:
      raise ValueError(
          'Expected the number of complex values to be even, one real and one'
          ' imaginary part for each complex value.'
      )
    value = iter(array_pb.complex128_list.value)
    value = [complex(real, imag) for real, imag in zip(value, value)]
  elif dtype is np.str_:
    value = array_pb.string_list.value
  else:
    raise NotImplementedError(f'Unexpected `dtype` found: {dtype}.')

  # Strings are stored as bytes in `array_pb2.Array` and trailing null values
  # are dropped when using `np.bytes_`, use `np.object_` instead.
  if dtype is np.str_:
    dtype = np.object_

  # `Array` is a `Union` of native Python types and numpy types. However, the
  # protobuf representation of `Array` contains additional information like
  # dtype and shape. This information is lost when returning native Python types
  # making it impossible to infer the original dtype later. Therefore, a numpy
  # value should almost always be returned from this function. String values are
  # an exception to this because it's not possible to represent null-terminated
  # scalar strings using numpy and this is ok because string types can only be
  # inferred as string types.
  if not array_shape.is_shape_scalar(shape):
    value = np.array(value, dtype).reshape(shape)
  else:
    (value,) = value
    value = dtype(value)

  return value


def to_proto(
    value: Array, *, dtype_hint: Optional[type[np.generic]] = None
) -> array_pb2.Array:
  """Returns an `array_pb2.Array` for the `value`."""

  if dtype_hint is not None:
    if not dtype_utils.is_valid_dtype(dtype_hint):
      raise ValueError(
          f'Expected `dtype_hint` to be a valid dtype, found {dtype_hint}.'
      )
    if not is_compatible_dtype(value, dtype_hint):
      raise ValueError(f'Expected {value} to be compatible with {dtype_hint}.')
    dtype = dtype_hint
    # Cast values with a dtype of `np.number`, no other dtypes are compatible.
  else:
    if isinstance(value, (np.ndarray, np.generic)):
      dtype = value.dtype.type
      # If the value has a dtype of `np.bytes_` or `np.object_`, the serialized
      # dtype should still be a `np.str_`.
      if np.issubdtype(dtype, np.bytes_) or np.issubdtype(dtype, np.object_):
        dtype = np.str_
    else:
      dtype = dtype_utils.infer_dtype(value)

  def _contains_type(value, classinfo):
    if isinstance(value, (np.ndarray, np.generic)):
      item = value.item(0)
    else:
      item = value
    return isinstance(item, classinfo)

  # Normalize to a numpy value; strings are stored as bytes in `array_pb2.Array`
  # and trailing null values are dropped when using `np.bytes_`, so use
  # `np.object_` instead.
  if dtype is np.str_:
    if _contains_type(value, str):
      value = np.asarray(value, dtype=np.bytes_)
    else:
      value = np.asarray(value, dtype=np.object_)
  else:
    value = np.asarray(value, dtype)

  dtype_pb = dtype_utils.to_proto(dtype)
  shape_pb = array_shape.to_proto(value.shape)
  value = value.flatten().tolist()

  if dtype is np.bool_:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        bool_list=array_pb2.Array.BoolList(value=value),
    )
  elif dtype is np.int8:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int8_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int16:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int32_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int64_list=array_pb2.Array.Int64List(value=value),
    )
  elif dtype is np.uint8:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint8_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.uint16:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.uint32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint32_list=array_pb2.Array.Uint32List(value=value),
    )
  elif dtype is np.uint64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint64_list=array_pb2.Array.Uint64List(value=value),
    )
  elif dtype is np.float16:
    # Values of dtype `np.float16` are packed to and unpacked from a protobuf
    # field of type `int32` using the following logic in order to maintain
    # compatibility with how other external environments (e.g., TensorFlow, JAX)
    # represent values of `np.float16`.
    value = np.array(value, np.float16).astype(np.uint16).tolist()
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.float32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float32_list=array_pb2.Array.FloatList(value=value),
    )
  elif dtype is np.float64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float64_list=array_pb2.Array.DoubleList(value=value),
    )
  elif dtype is np.complex64:
    packed_value = []
    for x in value:
      if not isinstance(x, complex):
        raise ValueError(f'Expected a complex type, found {type(x)}.')
      packed_value.extend([x.real, x.imag])
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        complex64_list=array_pb2.Array.FloatList(value=packed_value),
    )
  elif dtype is np.complex128:
    packed_value = []
    for x in value:
      if not isinstance(x, complex):
        raise ValueError(f'Expected a complex type, found {type(x)}.')
      packed_value.extend([x.real, x.imag])
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        complex128_list=array_pb2.Array.DoubleList(value=packed_value),
    )
  elif dtype is np.str_:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        string_list=array_pb2.Array.BytesList(value=value),
    )
  else:
    raise NotImplementedError(f'Unexpected `dtype` found: {dtype}.')


def from_proto_content(array_pb: array_pb2.Array) -> Array:
  """Returns an `Array` for the `array_pb`."""
  dtype = dtype_utils.from_proto(array_pb.dtype)
  shape = array_shape.from_proto(array_pb.shape)

  if array_pb.HasField('content'):
    if dtype is not np.str_:
      value = np.frombuffer(array_pb.content, dtype)
    else:
      raise NotImplementedError(f'Unexpected `dtype` found: {dtype}.')
  else:
    raise ValueError('Expected `array_pb` to have a `content` field.')

  if not array_shape.is_shape_scalar(shape):
    value = value.reshape(shape)
  else:
    value = value.item()

  if array_shape.is_shape_scalar(shape):
    value = dtype(value)

  return value


def to_proto_content(
    value: Array, *, dtype_hint: Optional[type[np.generic]] = None
) -> array_pb2.Array:
  """Returns an `Array` for the `value`."""

  if dtype_hint is not None:
    if not dtype_utils.is_valid_dtype(dtype_hint):
      raise ValueError(
          f'Expected `dtype_hint` to be a valid dtype, found {dtype_hint}.'
      )
    if not is_compatible_dtype(value, dtype_hint):
      raise ValueError(f'Expected {value} to be compatible with {dtype_hint}.')
    dtype = dtype_hint
  else:
    if isinstance(value, (np.ndarray, np.generic)):
      dtype = value.dtype.type
      # If the value has a dtype of `np.bytes_` or `np.object_`, the serialized
      # dtype should still be a `np.str_`.
      if np.issubdtype(dtype, np.bytes_) or np.issubdtype(dtype, np.object_):
        dtype = np.str_
    else:
      dtype = dtype_utils.infer_dtype(value)

  # Normalize to a numpy value; strings are stored as bytes in `array_pb2.Array`
  # and trailing null values are dropped when using `np.bytes_`, so use
  # `np.object_` instead.
  if dtype is not np.str_:
    value = np.asarray(value, dtype)
  else:
    raise NotImplementedError(f'Unexpected `dtype` found: {dtype}.')

  dtype_pb = dtype_utils.to_proto(dtype)
  shape_pb = array_shape.to_proto(value.shape)
  content = value.tobytes()

  return array_pb2.Array(dtype=dtype_pb, shape=shape_pb, content=content)


def _can_cast(obj: object, dtype: type[np.generic]) -> bool:
  """Returns `True` if `obj` can be cast to the `dtype`."""
  if isinstance(obj, np.ndarray):
    # `np.can_cast` does not operate on non-scalar arrays. See
    # https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html for
    # more information.
    return all(_can_cast(x, dtype) for x in obj.flatten())
  elif isinstance(obj, (np.generic, bool, int, float, complex)):
    return np.can_cast(obj, dtype)
  elif isinstance(obj, (str, bytes)):
    # `np.can_cast` interprets strings as dtype-like specifications rather than
    # strings.
    return np.issubdtype(dtype, np.character)
  else:
    return False


def is_compatible_dtype(value: Array, dtype: type[np.generic]) -> bool:
  """Returns `True` if `value` is compatible with `dtype`, otherwise `False`.

  This functions checks that the `value` has the same scalar kind (e.g. integer,
  floating) and has a compatible size (e.g. 32-bits, 16-bits) as `dtype` .

  See https://numpy.org/doc/stable/reference/arrays.scalars.html for more
  information.

  Args:
    value: The value to check.
    dtype: The scalar `np.generic` to check against.
  """
  if isinstance(value, (np.ndarray, np.generic)):
    value_dtype = value.dtype
  else:
    value_dtype = type(value)

  # Check dtype kind and skip checking dtype size because `np.bool_` does not
  # have a size and values with a dtype `np.str_`, `np.bytes_`, and `np.object_`
  # have a variable length.
  if np.issubdtype(value_dtype, np.bool_):
    return dtype is np.bool_
  elif np.issubdtype(value_dtype, np.character):
    return dtype is np.str_
  elif np.issubdtype(value_dtype, np.object_):
    return dtype is np.str_

  # Check dtype kind.
  if np.issubdtype(value_dtype, np.integer):
    if not np.issubdtype(dtype, np.integer):
      return False
  elif np.issubdtype(value_dtype, np.floating):
    if not np.issubdtype(dtype, np.floating):
      return False
  elif np.issubdtype(value_dtype, np.complexfloating):
    if not np.issubdtype(dtype, np.complexfloating):
      return False
  else:
    return False

  # Check dtype size. After checking that the `value` has a compatible kind, it
  # is simple to check that the `value` has a compatible size by checking if it
  # can be cast to the `dtype`.
  if not _can_cast(value, dtype):
    return False

  return True


def is_compatible_shape(value: Array, shape: array_shape.ArrayShape) -> bool:
  """Returns `True` if `value` is compatible with `shape`, otherwise `False`.

  Args:
    value: The value to check.
    shape: The `tff.types.ArrayShape` to check against.
  """
  if isinstance(value, np.ndarray):
    return array_shape.is_compatible_with(value.shape, shape)
  else:
    return array_shape.is_shape_scalar(shape)


# def infer_shape(value: Array) -> array_shape.ArrayShape:
#   if isinstance(value, (np.ndarray, np.generic)):
#     return value.shape
#   else:
#     return ()
