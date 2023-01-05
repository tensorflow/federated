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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A set of utility methods for serializing Value protos using pybind11 bindings."""

import collections
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.utils import tensorflow_utils

_SerializeReturnType = tuple[executor_pb2.Value, computation_types.Type]
_DeserializeReturnType = tuple[Any, computation_types.Type]

# The maximum size allowed for serialized sequence values. Sequence that
# serialize to values larger than this will result in errors being raised.  This
# likely occurs when the sequence is dependent on, and thus pulling in, many of
# variables from the graph.
_DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES = 100 * (1024**2)  # 100 MB


class DatasetSerializationError(Exception):
  """Error raised during Dataset serialization or deserialization."""
  pass


@tracing.trace
def _serialize_computation(
    comp: computation_pb2.Computation,
    type_spec: Optional[computation_types.Type]) -> _SerializeReturnType:
  """Serializes a TFF computation."""
  type_spec = executor_utils.reconcile_value_type_with_type_spec(
      type_serialization.deserialize_type(comp.type), type_spec)
  return executor_pb2.Value(computation=comp), type_spec


def _value_proto_for_np_array(
    value, type_spec: computation_types.Type) -> executor_pb2.Value:
  """Creates value proto for np array, assumed to be assignable to type_spec."""
  tensor_proto = tf.make_tensor_proto(
      value, dtype=type_spec.dtype, verify_shape=True)
  any_pb = any_pb2.Any()
  any_pb.Pack(tensor_proto)
  return executor_pb2.Value(tensor=any_pb)


@tracing.trace
def _serialize_tensor_value(
    value: Any, type_spec: computation_types.TensorType
) -> tuple[executor_pb2.Value, computation_types.TensorType]:
  """Serializes a tensor value into `executor_pb2.Value`.

  Args:
    value: A Numpy array or other object understood by `tf.make_tensor_proto`.
    type_spec: A `tff.TensorType`.

  Returns:
    A tuple `(value_proto, ret_type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`,
    and `ret_type_spec` is the type of the serialized value. The `ret_type_spec`
    is the same as the argument `type_spec` if that argument was not `None`. If
    the argument was `None`, `ret_type_spec` is a type determined from `value`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  original_value = value
  if tf.is_tensor(value):
    if isinstance(value, tf.Variable):
      value = value.read_value()
    if tf.executing_eagerly():
      value = value.numpy()
    else:
      # Attempt to extract the value using the current graph context.
      with tf.compat.v1.Session() as sess:
        value = sess.run(value)
  # If we got a string or bytes scalar, wrap it in numpy so it has a dtype and
  # shape.
  if isinstance(value, bytes):
    value = np.bytes_(value)
  elif isinstance(value, str):
    value = np.str_(value)
  else:
    value = np.asarray(value)
  if not tf.TensorShape(value.shape).is_compatible_with(type_spec.shape):
    raise TypeError(f'Cannot serialize tensor with shape {value.shape} to '
                    f'shape {type_spec.shape}.')
  if value.dtype != type_spec.dtype.as_numpy_dtype:
    try:
      value = value.astype(type_spec.dtype.as_numpy_dtype, casting='same_kind')
    except TypeError as te:
      value_type_string = py_typecheck.type_string(type(original_value))
      raise TypeError(
          f'Failed to serialize value of Python type {value_type_string} to '
          f'a tensor of type {type_spec}.\nValue: {original_value}') from te

  value_proto = _value_proto_for_np_array(value, type_spec)
  return value_proto, type_spec


def _serialize_dataset(
    dataset,
    max_serialized_size_bytes=_DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES):
  """Serializes a `tf.data.Dataset` value into a `bytes` object.

  Args:
    dataset: A `tf.data.Dataset`.
    max_serialized_size_bytes: An `int` size in bytes designating the threshold
      on when to raise an error if the resulting serialization is too big.

  Returns:
    A `bytes` object that can be sent to
  `tensorflow_serialization.deserialize_dataset` to recover the original
  `tf.data.Dataset`.

  Raises:
    SerializationError: if there was an error in TensorFlow during
      serialization.
  """
  py_typecheck.check_type(dataset,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
  dataset_graph = tf.raw_ops.DatasetToGraphV2(
      input_dataset=tf.data.experimental.to_variant(dataset))
  if tf.executing_eagerly():
    dataset_graph_def_bytes = dataset_graph.numpy()
  else:
    dataset_graph_def_bytes = tf.compat.v1.Session().run(dataset_graph)
  if len(dataset_graph_def_bytes) > max_serialized_size_bytes:
    raise ValueError('Serialized size of Dataset ({:d} bytes) exceeds maximum '
                     'allowed ({:d} bytes)'.format(
                         len(dataset_graph_def_bytes),
                         max_serialized_size_bytes))
  return dataset_graph_def_bytes


def _check_container_compat_with_tf_nest(type_spec: computation_types.Type):
  """Asserts that all `StructTypes` with names have OrderedDict containers."""

  def _names_are_in_sorted_order(name_sequence: Sequence[str]) -> bool:
    return sorted(name_sequence) == name_sequence

  def _check_ordereddict_container_for_struct(type_to_check):
    if not type_to_check.is_struct():
      return type_to_check, False
    # We can't use `dir` here, since it sorts the names before returning. We
    # also must filter to names which are actually present.
    names_in_sequence_order = structure.name_list(type_to_check)
    names_are_sorted = _names_are_in_sorted_order(names_in_sequence_order)
    has_no_names = not bool(names_in_sequence_order)
    if has_no_names or (names_in_sequence_order and names_are_sorted):
      # If alphabetical order matches sequence order, TFF's deserialization will
      # traverse the structure correctly; there is no ambiguity here. On the
      # other hand, if there are no names, sequence order is the only method of
      # traversal, so there is no ambiguity here either.
      return type_to_check, False
    elif not type_to_check.is_struct_with_python():
      raise ValueError('Attempting to serialize a named struct type with '
                       'ambiguous traversal order (sequence order distinct '
                       'from alphabetical order) without a Python container; '
                       'this is an unsafe operation, as TFF cannot determine '
                       'the intended traversal order after deserializing the '
                       'proto due to inconsistent behavior of tf.nest.')

    container_type = computation_types.StructWithPythonType.get_container_type(
        type_to_check)
    if (not names_are_sorted) and container_type is not collections.OrderedDict:
      raise ValueError('Attempted to serialize a dataset yielding named '
                       'elements in non-sorted sequence order with '
                       f'non-OrderedDict container (type {container_type}). '
                       'This is an ambiguous operation; `tf.nest` behaves in '
                       'a manner which depends on the Python type of this '
                       'container, so coercing the dataset reconstructed '
                       'from the resulting Value proto depends on assuming a '
                       'single Python type here. Please prefer to use '
                       '`collections.OrderedDict` containers for the elements '
                       'your dataset yields.')
    return type_to_check, False

  type_transformations.transform_type_postorder(
      type_spec, _check_ordereddict_container_for_struct)


@tracing.trace
def _serialize_sequence_value(
    value: Union[Union[type_conversions.TF_DATASET_REPRESENTATION_TYPES],
                 list[Any]],
    type_spec: computation_types.SequenceType) -> _SerializeReturnType:
  """Serializes a `tf.data.Dataset` value into `executor_pb2.Value`.

  Args:
    value: A `tf.data.Dataset`, or equivalent list of values convertible to
      (potentially structures of) tensors.
    type_spec: A `computation_types.Type` specifying the TFF sequence type of
      `value.`

  Returns:
    A tuple `(value_proto, type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`,
    and `type_spec` is the type of the serialized value.
  """
  if isinstance(value, list):
    value = tensorflow_utils.make_data_set_from_elements(
        None, value, type_spec.element)
  if not isinstance(value, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
    raise TypeError(
        'Cannot serialize Python type {!s} as TFF type {!s}.'.format(
            py_typecheck.type_string(type(value)),
            type_spec if type_spec is not None else 'unknown'))
  value = tensorflow_utils.coerce_dataset_to_yield_structures(value)
  element_type = computation_types.to_type(value.element_spec)
  _check_container_compat_with_tf_nest(element_type)
  value_type = computation_types.SequenceType(element_type)
  if not type_spec.is_assignable_from(value_type):
    raise TypeError(
        'Cannot serialize dataset with elements of type {!s} as TFF type {!s}.'
        .format(value_type, type_spec if type_spec is not None else 'unknown'))
  value_proto = executor_pb2.Value()
  # TFF must store the type spec here because TF will lose the ordering of the
  # names for `tf.data.Dataset` that return elements of
  # `collections.abc.Mapping` type. This allows TFF to preserve and restore the
  # key ordering upon deserialization.
  value_proto.sequence.serialized_graph_def = _serialize_dataset(value)
  value_proto.sequence.element_type.CopyFrom(
      type_serialization.serialize_type(element_type))
  return value_proto, type_spec


@tracing.trace
def _serialize_struct_type(
    struct_typed_value: Any,
    type_spec: computation_types.StructType,
) -> computation_types.StructType:
  """Serializes a value of tuple type."""
  value_structure = structure.from_container(struct_typed_value)
  if len(value_structure) != len(type_spec):
    raise TypeError('Cannot serialize a struct value of '
                    f'{len(value_structure)} elements to a struct type '
                    f'requiring {len(type_spec)} elements. Trying to serialize'
                    f'\n{struct_typed_value!r}\nto\n{type_spec}.')
  type_elem_iter = structure.iter_elements(type_spec)
  val_elem_iter = structure.iter_elements(value_structure)
  elements = []
  for (e_name, e_type), (_, e_val) in zip(type_elem_iter, val_elem_iter):
    e_value, _ = serialize_value(e_val, e_type)
    if e_name:
      element = executor_pb2.Value.Struct.Element(name=e_name, value=e_value)
    else:
      element = executor_pb2.Value.Struct.Element(value=e_value)
    elements.append(element)
  value_proto = executor_pb2.Value(
      struct=executor_pb2.Value.Struct(element=elements))
  return value_proto, type_spec


@tracing.trace
def _serialize_federated_value(
    federated_value: Any, type_spec: computation_types.FederatedType
) -> computation_types.FederatedType:
  """Serializes a value of federated type."""
  if type_spec.all_equal:
    value = [federated_value]
  else:
    value = federated_value
  py_typecheck.check_type(value, list)
  value_proto = executor_pb2.Value()
  for v in value:
    federated_value_proto, it_type = serialize_value(v, type_spec.member)
    type_spec.member.check_assignable_from(it_type)
    value_proto.federated.value.append(federated_value_proto)
  value_proto.federated.type.CopyFrom(
      type_serialization.serialize_type(type_spec).federated)
  return value_proto, type_spec


@tracing.trace
def serialize_value(
    value: Any,
    type_spec: Optional[computation_types.Type] = None,
) -> _SerializeReturnType:
  """Serializes a value into `executor_pb2.Value`.

  We use a switch/function pattern in the body here (and in `deserialize_value`
  below in order to persist more information in traces and profiling.

  Args:
    value: A value to be serialized.
    type_spec: Optional type spec, a `tff.Type` or something convertible to it.

  Returns:
    A 2-tuple of serialized value and `tff.Type` that represents the TFF type of
    the serialized value.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, computation_pb2.Computation):
    return _serialize_computation(value, type_spec)
  elif isinstance(value, computation_impl.ConcreteComputation):
    return _serialize_computation(
        computation_impl.ConcreteComputation.get_proto(value),
        executor_utils.reconcile_value_with_type_spec(value, type_spec))
  elif type_spec is None:
    raise TypeError('A type hint is required when serializing a value which '
                    'is not a TFF computation. Asked to serialized value {v} '
                    ' of type {t} with None type spec.'.format(
                        v=value, t=type(value)))
  elif type_spec.is_tensor():
    return _serialize_tensor_value(value, type_spec)
  elif type_spec.is_sequence():
    return _serialize_sequence_value(value, type_spec)
  elif type_spec.is_struct():
    return _serialize_struct_type(value, type_spec)
  elif type_spec.is_federated():
    return _serialize_federated_value(value, type_spec)
  else:
    raise ValueError(
        'Unable to serialize value with Python type {} and {} TFF type.'.format(
            str(py_typecheck.type_string(type(value))),
            str(type_spec) if type_spec is not None else 'unknown'))


@tracing.trace
def _deserialize_computation(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a TFF computation."""
  return (value_proto.computation,
          type_serialization.deserialize_type(value_proto.computation.type))


def _tensor_for_value(value_proto: executor_pb2.Value) -> tf.Tensor:
  tensor_proto = tf.make_tensor_proto(values=0)
  if not value_proto.tensor.Unpack(tensor_proto):
    raise ValueError('Unable to unpack the received tensor value.')
  tensor_value = tf.make_ndarray(tensor_proto)
  return tensor_value


@tracing.trace
def _deserialize_tensor_value(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a tensor value from `.Value`.

  Args:
    value_proto: An instance of `executor_pb2.Value`.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a Numpy array that represents
    the deserialized value, and `type_spec` is an instance of `tff.TensorType`
    that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  value = _tensor_for_value(value_proto)
  value_type = computation_types.TensorType(
      dtype=value.dtype, shape=value.shape)
  if not value.shape:
    # Unwrap the scalar array as just a primitive numeric.
    value = value.dtype.type(value)
  return value, value_type


def _deserialize_dataset_from_graph_def(serialized_graph_def: bytes,
                                        element_type: computation_types.Type):
  """Deserializes a serialized `tf.compat.v1.GraphDef` to a `tf.data.Dataset`.

  The `tf.data.Dataset` we return here *only* yields Python tuple or
  OrderedDict containers, depending on whether containers containing elements
  have names or have no names. Any coercion to Python containers must be
  performed after calling this function.

  Args:
    serialized_graph_def: `bytes` object produced by
      `tensorflow_serialization.serialize_dataset`
    element_type: a `tff.Type` object representing the type structure of the
      elements yielded from the dataset. Any Python container information in
      `element_type` will be dropped, and the dataset returned from this
      function will yield either tuples or OrderedDicts, but no other
      containers.

  Returns:
    A `tf.data.Dataset` instance.
  """
  py_typecheck.check_type(element_type, computation_types.Type)
  type_analysis.check_tensorflow_compatible_type(element_type)

  def transform_to_tff_known_type(
      type_spec: computation_types.Type) -> tuple[computation_types.Type, bool]:
    """Transforms `StructType` to `StructWithPythonType` of known structure."""
    if type_spec.is_struct():
      field_is_named = tuple(
          name is not None for name, _ in structure.iter_elements(type_spec))
      has_names = any(field_is_named)
      is_all_named = all(field_is_named)
      if is_all_named:
        return computation_types.StructWithPythonType(
            elements=structure.iter_elements(type_spec),
            container_type=collections.OrderedDict), True
      elif not has_names:
        return computation_types.StructWithPythonType(
            elements=structure.iter_elements(type_spec),
            container_type=tuple), True
      else:
        raise TypeError('Cannot represent TFF type in TF because it contains '
                        f'partially named structures. Type: {type_spec}')
    return type_spec, False

  if element_type.is_struct():
    # TF doesn't support `structure.Struct` types, so we must transform the
    # `StructType` into a `StructWithPythonType` for use as the
    # `tf.data.Dataset.element_spec` later. We must additionally drop any
    # further type specification the user may have made, to prevent tf.data from
    # performing optimizations in yielding elements which are incompatible with
    # the way TFF consumes composite tensors.
    tf_compatible_type, _ = type_transformations.transform_type_postorder(
        element_type, transform_to_tff_known_type)
  else:
    # We've checked this is only a struct or tensors, so we know this is a
    # `TensorType` here and will use as-is.
    tf_compatible_type = element_type

  tf_type_spec = type_conversions.type_to_tf_structure(tf_compatible_type)
  ds = tf.data.experimental.from_variant(
      tf.raw_ops.DatasetFromGraph(graph_def=serialized_graph_def),
      structure=tf_type_spec)
  # Repopulate the Python type information we potentially threw away above.
  return _coerce_deserialized_dataset_elements_to_tff_type_spec(
      ds, element_type)


def _ensure_sparse_tensor(
    x: Union[Mapping[str, tf.Tensor], tf.SparseTensor]) -> tf.SparseTensor:
  """Converts argument from mapping to SparseTensor if necessary."""
  if isinstance(x, tf.SparseTensor):
    return x
  else:
    if {'dense_shape', 'indices', 'values'} != set(x.keys()):
      raise TypeError('Expected value corresponding to SparseTensor typespec '
                      'to contain keys: \'dense_shape\', \'indices\', and '
                      f'\'values\'; found instead keys: {x.keys()}')
    # Unpack the mapping in x directly to sparse tensor, by passing the
    # keywords.
    return tf.sparse.SparseTensor(**x)


def _coerce_deserialized_dataset_elements_to_tff_type_spec(
    dataset: tf.data.Dataset, element_type: Union[computation_types.StructType,
                                                  computation_types.TensorType]
) -> tf.data.Dataset:
  """Map the elements of a deserialized dataset to a specified type.

  This is used to re-inject Python type information from `element_type` to the
  elements of `dataset` after deserialization.

  This function is tightly coupled with `_deserialize_dataset_from_graph_def`
  above, which ensures that the dataset arguments here yield only ODict, tuple,
  or tensor-typed elements.

  Args:
    dataset: a `tf.data.Dataset` instance.
    element_type: a `tff.Type` specifying the type of the elements of `dataset`.
      Must be a `tff.TensorType` or `tff.StructType`.

  Returns:
    A `tf.data.Dataset` whose output types are compatible with
    `element_type`.

  Raises:
    ValueError: if the elements of `dataset` cannot be coerced into
      `element_type`.
  """
  py_typecheck.check_type(dataset,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
  py_typecheck.check_type(element_type, computation_types.Type)

  if element_type.is_tensor():
    return dataset

  def _to_representative_value(type_spec, elements):
    """Converts container to a type understood by TF and TFF."""
    if type_spec.is_tensor():
      # Nothing to do here; we shortcut to focus on the structure case.
      tf.debugging.Assert(tf.is_tensor(elements), [elements])
      return elements
    elif tf.is_tensor(elements):
      # In this case we have a singleton tuple tensor that may have been
      # unwrapped by tf.data.
      elements = (elements,)

    if not isinstance(elements, (collections.OrderedDict, tuple)):
      raise TypeError('Structures expected to be represented as ODicts or '
                      f'lists; got value {elements} of Python type '
                      f'{type(elements)}')

    if type_spec.is_struct_with_python():
      py_type = computation_types.StructWithPythonType.get_container_type(
          type_spec)

      if py_type is tf.sparse.SparseTensor:
        return _ensure_sparse_tensor(elements)
      elif py_type is tf.RaggedTensor:
        # TODO(b/237423976): Presumably we need a handling like SparseTensor
        # here.
        return elements

      field_types = structure.iter_elements(type_spec)
      if len(structure.name_list(type_spec)) == len(type_spec):
        # We know that we have dict-like elements if we're inside this case;
        # this is coupled with the internal function transform_to_tff_known_type
        # above, which coerces *all* fully-named structures to ODicts during
        # deserialization.
        values = collections.OrderedDict(
            (name, _to_representative_value(field_type, elements[name]))
            for name, field_type in field_types)
        return py_type(**values)
      else:
        values = [
            _to_representative_value(field_type, e)
            for (_, field_type), e in zip(field_types, elements)
        ]
        if py_typecheck.is_named_tuple(py_type):
          return py_type(*values)
        return py_type(values)
    elif type_spec.is_struct():
      # Though these values should be represented as expected by the guarantees
      # of deserialization, we must continue the traversal to ensure that all
      # their leaves respect the associated python types.
      field_types = structure.to_elements(type_spec)
      is_all_named = all([name is not None for name, _ in field_types])
      if is_all_named:
        # Similarly, here, we know we have dict-like elements.
        values = [(name, _to_representative_value(field_type, elements[name]))
                  for name, field_type in field_types]
        return collections.OrderedDict(values)
      else:
        return tuple(
            _to_representative_value(t, e) for t, e in zip(type_spec, elements))
    else:
      raise ValueError(
          f'Coercing a dataset with elements of expected type {type_spec}, '
          f'produced a value with incompatible type `{type(elements)}`. Value: '
          f'{elements}')

  def _unwrap_args(*args):
    if len(args) == 1:
      return _to_representative_value(element_type, args[0])
    else:
      return _to_representative_value(element_type, args)

  return dataset.map(_unwrap_args)


@tracing.trace
def _deserialize_sequence_value(
    sequence_value_proto: executor_pb2.Value.Sequence,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a `tf.data.Dataset`.

  Args:
    sequence_value_proto: `Sequence` protocol buffer message.
    type_hint: A `computation_types.Type` that hints at what the value type
      should be for executors that only return values. If the type hint is not
      set, the `sequence_value_proto.element_type` field will be read and used.
      At least one must be present.

  Returns:
    A tuple of `(tf.data.Dataset, tff.Type)`.
  """
  if type_hint is not None:
    element_type = type_hint.element
  elif sequence_value_proto.HasField('element_type'):
    element_type = type_serialization.deserialize_type(
        sequence_value_proto.element_type)
  else:
    raise ValueError(
        'Cannot deserialize a sequence Value proto that without one of '
        '`element_type` proto field or `element_type_hint`')
  which_value = sequence_value_proto.WhichOneof('value')
  if which_value == 'zipped_saved_model':
    raise ValueError('Deserializing dataset from zipped save model no longer '
                     'supported.')
  elif which_value == 'serialized_graph_def':
    ds = _deserialize_dataset_from_graph_def(
        sequence_value_proto.serialized_graph_def, element_type)
  else:
    raise NotImplementedError(
        'Deserializing Sequences encoded as {!s} has not been implemented'
        .format(which_value))
  return ds, computation_types.SequenceType(element=element_type)


@tracing.trace
def _deserialize_struct_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value of struct type."""
  val_elems = []
  type_elems = []
  if type_hint is not None:
    element_types = tuple(type_hint)
  else:
    element_types = [None] * len(value_proto.struct.element)
  for e, e_type in zip(value_proto.struct.element, element_types):
    name = e.name if e.name else None
    e_val, e_type = deserialize_value(e.value, e_type)
    val_elems.append((name, e_val))
    type_elems.append((name, e_type) if name else e_type)
  return (structure.Struct(val_elems), computation_types.StructType(type_elems))


def _ensure_deserialized_types_compatible(
    previous_type: Optional[computation_types.Type],
    next_type: computation_types.Type) -> computation_types.Type:
  """Ensures one of `previous_type` or `next_type` is assignable to the other.

  Returns the type which is assignable from the other.

  Args:
    previous_type: Instance of `computation_types.Type` or `None`.
    next_type: Instance of `computation_types.Type`.

  Returns:
    The supertype of `previous_type` and `next_type`.

  Raises:
    TypeError if neither type is assignable from the other.
  """
  if previous_type is None:
    return next_type
  else:
    if next_type.is_assignable_from(previous_type):
      return next_type
    elif previous_type.is_assignable_from(next_type):
      return previous_type
    raise TypeError('Type mismatch checking member assignability under a '
                    'federated value. Deserialized type {} is incompatible '
                    'with previously deserialized {}.'.format(
                        next_type, previous_type))


@tracing.trace
def _deserialize_federated_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value of federated type."""
  if not value_proto.federated.value:
    raise ValueError('Attempting to deserialize federated value with no data.')
  # The C++ runtime doesn't use the `all_equal` boolean (and doesn't report it
  # in returned values), however the type_hint on the computation may contain
  # it.
  if type_hint is not None:
    all_equal = type_hint.all_equal
  else:
    all_equal = value_proto.federated.type.all_equal
  placement_uri = value_proto.federated.type.placement.value.uri
  # item_type will represent a supertype of all deserialized member types in the
  # federated value. This will be the hint used for deserialize member values.
  if type_hint is not None:
    item_type_hint = type_hint.member
  else:
    item_type_hint = None
  item_type = None
  if all_equal:
    # As an optimization, we only deserialize the first value of an
    # `all_equal=True` federated value.
    items = [value_proto.federated.value[0]]
  else:
    items = value_proto.federated.value
  value = []
  for item in items:
    item_value, next_item_type = deserialize_value(item, item_type_hint)
    item_type = _ensure_deserialized_types_compatible(item_type, next_item_type)
    value.append(item_value)
  type_spec = computation_types.FederatedType(
      item_type,
      placement=placements.uri_to_placement_literal(placement_uri),
      all_equal=all_equal)
  if all_equal:
    value = value[0]
  return value, type_spec


@tracing.trace
def deserialize_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value (of any type) from `executor_pb2.Value`.

  Args:
    value_proto: An instance of `executor_pb2.Value`.
    type_hint: A `comptuations_types.Type` that hints at what the value type
      should be for executors that only return values.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a deserialized
    representation of the transmitted value (e.g., Numpy array, or a
    `pb.Computation` instance), and `type_spec` is an instance of
    `tff.TensorType` that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if not hasattr(value_proto, 'WhichOneof'):
    raise TypeError('`value_proto` must be a protocol buffer message with a '
                    '`value` oneof field.')
  which_value = value_proto.WhichOneof('value')
  if which_value == 'tensor':
    return _deserialize_tensor_value(value_proto)
  elif which_value == 'computation':
    return _deserialize_computation(value_proto)
  elif which_value == 'sequence':
    return _deserialize_sequence_value(value_proto.sequence, type_hint)
  elif which_value == 'struct':
    return _deserialize_struct_value(value_proto, type_hint)
  elif which_value == 'federated':
    return _deserialize_federated_value(value_proto, type_hint)
  else:
    raise ValueError(
        'Unable to deserialize a value of type {}.'.format(which_value))


CardinalitiesType = Mapping[placements.PlacementLiteral, int]


def serialize_cardinalities(
    cardinalities: CardinalitiesType) -> list[executor_pb2.Cardinality]:
  serialized_cardinalities = []
  for placement, cardinality in cardinalities.items():
    cardinality_message = executor_pb2.Cardinality(
        placement=computation_pb2.Placement(uri=placement.uri),
        cardinality=cardinality)
    serialized_cardinalities.append(cardinality_message)
  return serialized_cardinalities


def deserialize_cardinalities(
    serialized_cardinalities: Collection[executor_pb2.Cardinality]
) -> CardinalitiesType:
  cardinalities_dict = {}
  for cardinality_spec in serialized_cardinalities:
    literal = placements.uri_to_placement_literal(
        cardinality_spec.placement.uri)
    cardinalities_dict[literal] = cardinality_spec.cardinality
  return cardinalities_dict
