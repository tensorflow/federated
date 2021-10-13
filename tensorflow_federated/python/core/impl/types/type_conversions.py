# Copyright 2018, The TensorFlow Federated Authors.
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
# limitations under the License.
"""Utilities for type conversion, type checking, type inference, etc."""

import collections
from typing import Any, Callable, Optional

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import typed_object

# This symbol being defined here is somewhat unfortunate. Likely, this symbol
# should be factored into a module that encapsulates the type functions related
# to the TensorFlow platform. However, it seems useful to consider how to
# organize such a boundary in the context of the entire type system. For
# example, we have an abstraction for a TensorFlow computation, but we do not
# have such an abstraction for a Tensor type.
TF_DATASET_REPRESENTATION_TYPES = (
    tf.data.Dataset,
    tf.compat.v1.data.Dataset,
)


def infer_type(arg: Any) -> Optional[computation_types.Type]:
  """Infers the TFF type of the argument (a `computation_types.Type` instance).

  WARNING: This function is only partially implemented.

  The kinds of arguments that are currently correctly recognized:
  - tensors, variables, and data sets,
  - things that are convertible to tensors (including numpy arrays, builtin
    types, as well as lists and tuples of any of the above, etc.),
  - nested lists, tuples, namedtuples, anonymous tuples, dict, and OrderedDicts.

  Args:
    arg: The argument, the TFF type of which to infer.

  Returns:
    Either an instance of `computation_types.Type`, or `None` if the argument is
    `None`.
  """
  if arg is None:
    return None
  elif isinstance(arg, typed_object.TypedObject):
    return arg.type_signature
  elif tf.is_tensor(arg):
    # `tf.is_tensor` returns true for some things that are not actually single
    # `tf.Tensor`s, including `tf.SparseTensor`s and `tf.RaggedTensor`s.
    if isinstance(arg, tf.RaggedTensor):
      return computation_types.StructWithPythonType(
          (('flat_values', infer_type(arg.flat_values)),
           ('nested_row_splits', infer_type(arg.nested_row_splits))),
          tf.RaggedTensor)
    elif isinstance(arg, tf.SparseTensor):
      shape_marker = computation_types.marker_type_from_static_shape(
          arg.shape, '`tf.SparseTensor`')
      return computation_types.StructWithPythonType(
          (('indices', infer_type(arg.indices)),
           ('values', infer_type(arg.values)),
           ('dense_shape', infer_type(arg.dense_shape)),
           ('static_dense_shape_marker', shape_marker)), tf.SparseTensor)
    else:
      return computation_types.TensorType(arg.dtype.base_dtype, arg.shape)
  elif isinstance(arg, TF_DATASET_REPRESENTATION_TYPES):
    element_type = computation_types.to_type(arg.element_spec)
    return computation_types.SequenceType(element_type)
  elif isinstance(arg, structure.Struct):
    return computation_types.StructType([
        (k, infer_type(v)) if k else infer_type(v)
        for k, v in structure.iter_elements(arg)
    ])
  elif py_typecheck.is_attrs(arg):
    items = attr.asdict(
        arg, dict_factory=collections.OrderedDict, recurse=False)
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items.items()], type(arg))
  elif py_typecheck.is_named_tuple(arg):
    # In Python 3.8 and later `_asdict` no longer return OrdereDict, rather a
    # regular `dict`.
    items = collections.OrderedDict(arg._asdict())
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items.items()], type(arg))
  elif isinstance(arg, dict):
    if isinstance(arg, collections.OrderedDict):
      items = arg.items()
    else:
      items = sorted(arg.items())
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items], type(arg))
  elif isinstance(arg, (tuple, list)):
    elements = []
    all_elements_named = True
    for element in arg:
      all_elements_named &= py_typecheck.is_name_value_pair(element)
      elements.append(infer_type(element))
    # If this is a tuple of (name, value) pairs, the caller most likely intended
    # this to be a StructType, so we avoid storing the Python container.
    if elements and all_elements_named:
      return computation_types.StructType(elements)
    else:
      return computation_types.StructWithPythonType(elements, type(arg))
  elif isinstance(arg, str):
    return computation_types.TensorType(tf.string)
  elif isinstance(arg, (np.generic, np.ndarray)):
    return computation_types.TensorType(
        tf.dtypes.as_dtype(arg.dtype), arg.shape)
  else:
    arg_type = type(arg)
    if arg_type is bool:
      return computation_types.TensorType(tf.bool)
    elif arg_type is int:
      # Chose the integral type based on value.
      if arg > tf.int64.max or arg < tf.int64.min:
        raise TypeError('No integral type support for values outside range '
                        f'[{tf.int64.min}, {tf.int64.max}]. Got: {arg}')
      elif arg > tf.int32.max or arg < tf.int32.min:
        return computation_types.TensorType(tf.int64)
      else:
        return computation_types.TensorType(tf.int32)
    elif arg_type is float:
      return computation_types.TensorType(tf.float32)
    else:
      # Now fall back onto the heavier-weight processing, as all else failed.
      # Use make_tensor_proto() to make sure to handle it consistently with
      # how TensorFlow is handling values (e.g., recognizing int as int32, as
      # opposed to int64 as in NumPy).
      try:
        # TODO(b/113112885): Find something more lightweight we could use here.
        tensor_proto = tf.make_tensor_proto(arg)
        return computation_types.TensorType(
            tf.dtypes.as_dtype(tensor_proto.dtype),
            tf.TensorShape(tensor_proto.tensor_shape))
      except TypeError as err:
        raise TypeError('Could not infer the TFF type of {}: {}'.format(
            py_typecheck.type_string(type(arg)), err))


def type_to_tf_dtypes_and_shapes(type_spec: computation_types.Type):
  """Returns nested structures of tensor dtypes and shapes for a given TFF type.

  The returned dtypes and shapes match those used by `tf.data.Dataset`s to
  indicate the type and shape of their elements. They can be used, e.g., as
  arguments in constructing an iterator over a string handle.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    A pair of parallel nested structures with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the two structures returned is the
    same as the layout of the nested type defined by `type_spec`. Named tuples
    are represented as dictionaries.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec.is_tensor():
    return (type_spec.dtype, type_spec.shape)
  elif type_spec.is_struct():
    elements = structure.to_elements(type_spec)
    if not elements:
      output_dtypes = []
      output_shapes = []
    elif elements[0][0] is not None:
      output_dtypes = collections.OrderedDict()
      output_shapes = collections.OrderedDict()
      for e in elements:
        element_name = e[0]
        element_spec = e[1]
        if element_name is None:
          raise ValueError(
              'When a sequence appears as a part of a parameter to a section '
              'of TensorFlow code, in the type signature of elements of that '
              'sequence all named tuples must have their elements explicitly '
              'named, and this does not appear to be the case in {}.'.format(
                  type_spec))
        element_output = type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes[element_name] = element_output[0]
        output_shapes[element_name] = element_output[1]
    else:
      output_dtypes = []
      output_shapes = []
      for e in elements:
        element_name = e[0]
        element_spec = e[1]
        if element_name is not None:
          raise ValueError(
              'When a sequence appears as a part of a parameter to a section '
              'of TensorFlow code, in the type signature of elements of that '
              'sequence all named tuples must have their elements explicitly '
              'named, and this does not appear to be the case in {}.'.format(
                  type_spec))
        element_output = type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes.append(element_output[0])
        output_shapes.append(element_output[1])
    if type_spec.python_container is not None:
      container_type = type_spec.python_container

      def build_py_container(elements):
        if (py_typecheck.is_named_tuple(container_type) or
            py_typecheck.is_attrs(container_type)):
          return container_type(**dict(elements))
        else:
          return container_type(elements)

      output_dtypes = build_py_container(output_dtypes)
      output_shapes = build_py_container(output_shapes)
    else:
      output_dtypes = tuple(output_dtypes)
      output_shapes = tuple(output_shapes)
    return (output_dtypes, output_shapes)
  else:
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


def type_to_tf_tensor_specs(type_spec: computation_types.Type):
  """Returns nested structure of `tf.TensorSpec`s for a given TFF type.

  The dtypes and shapes of the returned `tf.TensorSpec`s match those used by
  `tf.data.Dataset`s to indicate the type and shape of their elements. They can
  be used, e.g., as arguments in constructing an iterator over a string handle.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    A nested structure of `tf.TensorSpec`s with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the structure returned is the same as
    the layout of the nested type defined by `type_spec`. Named tuples are
    represented as dictionaries.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  dtypes, shapes = type_to_tf_dtypes_and_shapes(type_spec)
  return tf.nest.map_structure(lambda dtype, shape: tf.TensorSpec(shape, dtype),
                               dtypes, shapes)


def type_to_tf_structure(type_spec: computation_types.Type):
  """Returns nested `tf.data.experimental.Structure` for a given TFF type.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    An instance of `tf.data.experimental.Structure`, possibly nested, that
    corresponds to `type_spec`.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec.is_tensor():
    return tf.TensorSpec(type_spec.shape, type_spec.dtype)
  elif type_spec.is_struct():
    elements = structure.to_elements(type_spec)
    if not elements:
      return ()
    element_outputs = [(k, type_to_tf_structure(v)) for k, v in elements]
    named = element_outputs[0][0] is not None
    if not all((e[0] is not None) == named for e in element_outputs):
      raise ValueError('Tuple elements inconsistently named.')
    if type_spec.python_container is None:
      if named:
        return collections.OrderedDict(element_outputs)
      else:
        return tuple(v for _, v in element_outputs)
    else:
      container_type = type_spec.python_container
      if (py_typecheck.is_named_tuple(container_type) or
          py_typecheck.is_attrs(container_type)):
        return container_type(**dict(element_outputs))
      elif container_type is tf.RaggedTensor:
        flat_values = type_spec.flat_values
        nested_row_splits = type_spec.nested_row_splits
        return tf.RaggedTensorSpec(
            shape=None,
            dtype=flat_values.dtype,
            ragged_rank=len(nested_row_splits),
            row_splits_dtype=nested_row_splits[0].dtype,
            flat_values_spec=tf.TensorSpec(flat_values.shape,
                                           flat_values.dtype))
      elif container_type is tf.SparseTensor:
        if type_spec.static_dense_shape_marker.shape.dims is not None:
          shape = computation_types.static_shape_from_marker_type(
              type_spec.static_dense_shape_marker)
        elif (type_spec.indices.shape is not None and
              type_spec.indices.shape.dims[1] is not None):
          # We can't generally infer the shape from the type of the tensors, but
          # we *can* infer the rank based on the shapes of `indices` or
          # `dense_shape`.
          rank = type_spec.indices.shape.dims[1]
          shape = tf.TensorShape([None] * rank)
        elif (type_spec.dense_shape.shape is not None and
              type_spec.dense_shape.shape.dims[0] is not None):
          rank = type_spec.dense_shape.shape.dims[0]
          shape = tf.TensorShape([None] * rank)
        else:
          shape = None
        return tf.SparseTensorSpec(shape=shape, dtype=type_spec.values.dtype)
      elif named:
        return container_type(element_outputs)
      else:
        return container_type(
            e if e[0] is not None else e[1] for e in element_outputs)
  else:
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


def type_from_tensors(tensors):
  """Builds a `tff.Type` from supplied tensors.

  Args:
    tensors: A nested structure of tensors.

  Returns:
    The nested TensorType structure.
  """

  def _mapping_fn(x):
    if not tf.is_tensor(x):
      x = tf.convert_to_tensor(x)
    return computation_types.TensorType(x.dtype.base_dtype, x.shape)

  if isinstance(tensors, structure.Struct):
    type_spec = structure.map_structure(_mapping_fn, tensors)
  else:
    type_spec = tf.nest.map_structure(_mapping_fn, tensors)
  return computation_types.to_type(type_spec)


def type_to_py_container(value, type_spec):
  """Recursively convert `structure.Struct`s to Python containers.

  This is in some sense the inverse operation to
  `structure.from_container`.

  Args:
    value: A structure of anonymous tuples of values corresponding to
      `type_spec`.
    type_spec: The `tff.Type` to which value should conform, possibly including
      `computation_types.StructWithPythonType`.

  Returns:
    The input value, with containers converted to appropriate Python
    containers as specified by the `type_spec`.

  Raises:
    ValueError: If the conversion is not possible due to a mix of named
      and unnamed values.
  """
  if type_spec.is_federated():
    if type_spec.all_equal:
      structure_type_spec = type_spec.member
    else:
      if not isinstance(value, list):
        raise TypeError('Unexpected Python type for non-all-equal TFF type '
                        f'{type_spec}: expected `list`, found `{type(value)}`.')
      return [
          type_to_py_container(element, type_spec.member) for element in value
      ]
  else:
    structure_type_spec = type_spec

  if structure_type_spec.is_sequence():
    element_type = structure_type_spec.element
    if isinstance(value, list):
      return [type_to_py_container(element, element_type) for element in value]
    if isinstance(value, tf.data.Dataset):
      # `tf.data.Dataset` does not understand `Struct`, so the dataset
      # in `value` must already be yielding Python containers. This is because
      # when TFF is constructing datasets it always uses the proper Python
      # container, so we simply return `value` here without modification.
      return value
    raise TypeError('Unexpected Python type for TFF type {}: {}'.format(
        structure_type_spec, type(value)))

  if not structure_type_spec.is_struct():
    return value

  if not isinstance(value, structure.Struct):
    # NOTE: When encountering non-anonymous tuples, we assume that
    # this means that we're attempting to re-convert a value that
    # already has the proper containers, and we short-circuit to
    # avoid re-converting. This is a possibly dangerous assumption.
    return value
  anon_tuple = value

  def is_container_type_without_names(container_type):
    return (issubclass(container_type, (list, tuple)) and
            not py_typecheck.is_named_tuple(container_type))

  def is_container_type_with_names(container_type):
    return (py_typecheck.is_named_tuple(container_type) or
            py_typecheck.is_attrs(container_type) or
            issubclass(container_type, dict))

  # TODO(b/133228705): Consider requiring StructWithPythonType.
  container_type = structure_type_spec.python_container or structure.Struct
  container_is_anon_tuple = structure_type_spec.python_container is None

  # Avoid projecting the `structure.StructType`d TFF value into a Python
  # container that is not supported.
  if not container_is_anon_tuple:
    num_named_elements = len(dir(anon_tuple))
    num_unnamed_elements = len(anon_tuple) - num_named_elements
    if num_named_elements > 0 and num_unnamed_elements > 0:
      raise ValueError('Cannot represent value {} with container type {}, '
                       'because value contains a mix of named and unnamed '
                       'elements.'.format(anon_tuple, container_type))
    if (num_named_elements > 0 and
        is_container_type_without_names(container_type)):
      raise ValueError(
          'Cannot represent value {} with named elements '
          'using container type {} which does not support names. In TFF\'s '
          'typesystem, this corresponds to an implicit downcast'.format(
              anon_tuple, container_type))
  if (is_container_type_with_names(container_type) and
      len(dir(structure_type_spec)) != len(anon_tuple)):
    # If the type specifies the names, we have all the information we need.
    # Otherwise we must raise here.
    raise ValueError('When packaging as a Python value which requires names, '
                     'the TFF type spec must have all names specified. Found '
                     '{} names in type spec {} of length {}, with requested'
                     'python type {}.'.format(
                         len(dir(structure_type_spec)), structure_type_spec,
                         len(anon_tuple), container_type))

  elements = []
  for index, (elem_name, elem_type) in enumerate(
      structure.iter_elements(structure_type_spec)):
    value = type_to_py_container(anon_tuple[index], elem_type)

    if elem_name is None and not container_is_anon_tuple:
      elements.append(value)
    else:
      elements.append((elem_name, value))

  if (py_typecheck.is_named_tuple(container_type) or
      py_typecheck.is_attrs(container_type)):
    # The namedtuple and attr.s class constructors cannot interpret a list of
    # (name, value) tuples; instead call constructor using kwargs. Note that
    # these classes already define an order of names internally, so order does
    # not matter.
    return container_type(**dict(elements))
  elif container_type is tf.SparseTensor:
    elements = dict(elements)
    indices = elements['indices']
    values = elements['values']
    dense_shape = elements['dense_shape']
    # If the type contains a static `dense_shape`, we discard the provided
    # 'dense_shape' value and instead pass in the shape as a tensor constant,
    # allowing it to be visible statically.
    static_shape = computation_types.static_shape_from_marker_type(
        type_spec.static_dense_shape_marker)
    if static_shape is not None:
      dense_shape = tf.constant(static_shape.dims, tf.int64,
                                [len(static_shape.dims)])
    return tf.SparseTensor(indices, values, dense_shape)
  elif container_type is tf.RaggedTensor:
    elements = dict(elements)
    return tf.RaggedTensor.from_nested_row_splits(elements['flat_values'],
                                                  elements['nested_row_splits'])
  else:
    # E.g., tuple and list when elements only has values, but also `dict`,
    # `collections.OrderedDict`, or `structure.Struct` when
    # elements has (name, value) tuples.
    return container_type(elements)


def _structure_from_tensor_type_tree_inner(fn, type_spec):
  """Helper for `structure_from_tensor_type_tree`."""
  if type_spec.is_struct():

    def _map_element(element):
      name, nested_type = element
      return (name, _structure_from_tensor_type_tree_inner(fn, nested_type))

    return structure.Struct(
        map(_map_element, structure.iter_elements(type_spec)))
  elif type_spec.is_tensor():
    return fn(type_spec)
  else:
    raise ValueError('Expected tensor or structure type, found type:\n' +
                     type_spec.formatted_representation())


def structure_from_tensor_type_tree(fn: Callable[[computation_types.TensorType],
                                                 Any], type_spec) -> Any:
  """Constructs a structure from a `type_spec` tree of `tff.TensorType`s.

  Args:
    fn: A callable used to generate the elements with which to fill the
      resulting structure. `fn` will be called exactly once per leaf
      `tff.TensorType` in the order they appear in the `type_spec` structure.
    type_spec: A TFF type or value convertible to TFF type. Once converted,
      `type_spec` must be a `tff.TensorType` or `tff.StructType` containing only
      other `tff.TensorType`s and `tff.StructType`s.

  Returns:
    A structure with the same shape and Python containers as `type_spec` but
    with the `tff.TensorType` elements replaced with the results of `fn`.

  Raises:
    ValueError: if the provided `type_spec` is not a structural or tensor type.
  """
  type_spec = computation_types.to_type(type_spec)
  non_python_typed = _structure_from_tensor_type_tree_inner(fn, type_spec)
  return type_to_py_container(non_python_typed, type_spec)


def type_to_non_all_equal(type_spec):
  """Constructs a non-`all_equal` version of the federated type `type_spec`.

  Args:
    type_spec: An instance of `tff.FederatedType`.

  Returns:
    A federated type with the same member and placement, but `all_equal=False`.
  """
  py_typecheck.check_type(type_spec, computation_types.FederatedType)
  return computation_types.FederatedType(
      type_spec.member, type_spec.placement, all_equal=False)
