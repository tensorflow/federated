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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines functions and classes for building and manipulating TFF types."""

import abc
import atexit
import collections
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
import difflib
import enum
from typing import Optional, TypeVar, Union
import weakref

import attrs
import numpy as np
import tensorflow as tf
import tree
from typing_extensions import TypeGuard

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import placements

T = TypeVar('T')


class UnexpectedTypeError(TypeError):

  def __init__(self, expected: type['Type'], actual: 'Type'):
    message = f'Expected type of kind {expected}, found type {actual}'
    super().__init__(message)
    self.actual = actual
    self.expected = expected


# Prevent wrapping on a 100-character terminal.
MAX_LINE_LEN = 100


@enum.unique
class TypeRelation(enum.Enum):
  EQUIVALENT = 'equivalent'
  IDENTICAL = 'identical'
  ASSIGNABLE = 'assignable'


def type_mismatch_error_message(
    first: 'Type',
    second: 'Type',
    relation: TypeRelation,
    second_is_expected: bool = False,
) -> str:
  """Returns an error message describing the mismatch between two types."""
  maybe_expected = 'expected ' if second_is_expected else ''
  first_str = first.compact_representation()
  second_str = second.compact_representation()
  diff = None
  if first_str == second_str:
    # The two only differ in container types or some other property not
    # visible via the compact representation, so show `repr` instead.
    # No diff is used because `repr` prints to a single line.
    first_str = repr(first)
    second_str = repr(second)
    diff = None
  elif len(first_str) > MAX_LINE_LEN or len(second_str) > MAX_LINE_LEN:
    # The types are large structures, and so the formatted representation is
    # used and a summary diff is added. The logic here is that large types
    # may be easier to diff visually with a more structured representation,
    # and logical line breaks are required to make diff output useful.
    first_str = first.formatted_representation()
    second_str = second.formatted_representation()
    split_first = first_str.split('\n')
    split_second = second_str.split('\n')
    diff = '\n'.join(difflib.unified_diff(split_first, split_second))
  message = [
      'Type',
      f'`{first_str}`',
      f'is not {relation.value} to {maybe_expected}type',
      f'`{second_str}`',
  ]
  if diff:
    message += [f'\nDiff:\n{diff}']
  single_line = ' '.join(message)
  if len(single_line) > MAX_LINE_LEN or '\n' in single_line:
    return '\n'.join(message)
  else:
    return single_line


class TypeNotAssignableError(TypeError):

  def __init__(self, source_type, target_type):
    self.message = type_mismatch_error_message(
        source_type, target_type, TypeRelation.ASSIGNABLE
    )
    super().__init__(self.message)
    self.source_type = source_type
    self.target_type = target_type


class TypesNotEquivalentError(TypeError):

  def __init__(self, first_type, second_type):
    self.message = type_mismatch_error_message(
        first_type, second_type, TypeRelation.EQUIVALENT
    )
    super().__init__(self.message)
    self.first_type = first_type
    self.second_type = second_type


class TypesNotIdenticalError(TypeError):

  def __init__(self, first_type, second_type):
    self.message = type_mismatch_error_message(
        first_type, second_type, TypeRelation.IDENTICAL
    )
    super().__init__(self.message)
    self.first_type = first_type
    self.second_type = second_type


class Type(metaclass=abc.ABCMeta):
  """An abstract interface for all classes that represent TFF types."""

  def compact_representation(self) -> str:
    """Returns the compact string representation of this type."""
    return _string_representation(self, formatted=False)

  def formatted_representation(self) -> str:
    """Returns the formatted string representation of this type."""
    return _string_representation(self, formatted=True)

  @abc.abstractmethod
  def children(self) -> Iterator['Type']:
    """Returns a generator yielding immediate child types."""
    raise NotImplementedError

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this type."""
    raise NotImplementedError

  def __str__(self):
    """Returns a concise representation of this type."""
    return self.compact_representation()

  @abc.abstractmethod
  def __hash__(self):
    """Produces a hash value for this type."""
    raise NotImplementedError

  @abc.abstractmethod
  def __eq__(self, other):
    """Determines whether two type definitions are identical.

    Note that this notion of equality is stronger than equivalence. Two types
    with equivalent definitions may not be identical, e.g., if they represent
    templates with differently named type variables in their definitions.

    Args:
      other: The other type to compare against.

    Returns:
      `True` if type definitions are syntactically identical (as defined above),
      otherwise `False`.

    Raises:
      NotImplementedError: If not implemented in the derived class.
    """
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def check_assignable_from(self, source_type: 'Type') -> None:
    """Raises if values of `source_type` cannot be cast to this type."""
    if not self.is_assignable_from(source_type):
      raise TypeNotAssignableError(source_type=source_type, target_type=self)

  @abc.abstractmethod
  def is_assignable_from(self, source_type: 'Type') -> bool:
    """Returns whether values of `source_type` can be cast to this type."""
    raise NotImplementedError

  def check_equivalent_to(self, other: 'Type') -> None:
    """Raises if values of 'other' cannot be cast to and from this type."""
    if not self.is_equivalent_to(other):
      raise TypesNotEquivalentError(self, other)

  def is_equivalent_to(self, other: 'Type') -> bool:
    """Returns whether values of `other` can be cast to and from this type."""
    return self.is_assignable_from(other) and other.is_assignable_from(self)

  def check_identical_to(self, other: 'Type') -> None:
    """Raises if `other` and `Type` are not exactly identical."""
    if not self.is_identical_to(other):
      raise TypesNotIdenticalError(self, other)

  def is_identical_to(self, other: 'Type') -> bool:
    """Returns whether or not `self` and `other` are exactly identical."""
    return self == other


class _Intern(abc.ABCMeta):
  """A metaclass which interns instances.

  This is used to create classes where the following predicate holds:
  `MyClass(some_args) is MyClass(some_args)`

  That is, objects of the class with the same constructor parameters result
  in values with the same object identity. This can make comparison of deep
  structures much cheaper, since a shallow equality check can short-circuit
  comparison.

  Classes which set `_Intern` as a metaclass must have a
  `_hashable_from_init_args` classmethod which defines exactly the parameters
  passed to the `__init__` method. If one of the parameters passed to the
  `_Intern.__call__` is an iterator it will be converted to a list before
  `_hashable_from_init_args` and `__init__` are called.

  Note: also that this metaclass must only be used with *immutable* values, as
  mutation would cause all similarly-constructed instances to be mutated
  together.

  Inherits from `abc.ABCMeta` to prevent subclass conflicts.
  """

  @classmethod
  def _hashable_from_init_args(mcs, *args, **kwargs) -> Hashable:
    raise NotImplementedError

  def __call__(cls, *args, **kwargs):

    # Convert all `Iterator`s in both `args` and `kwargs` to `list`s so they can
    # be used in both `_hashable_from_init_args` and `__init__`.
    def _normalize(obj):
      if isinstance(obj, Iterator):
        return list(obj)
      else:
        return obj

    args = [_normalize(x) for x in args]
    kwargs = {k: _normalize(v) for k, v in kwargs.items()}

    # Salt the key with `cls` to account for two different classes that return
    # the same result from `_hashable_from_init_args`.
    key = (cls, cls._hashable_from_init_args(*args, **kwargs))
    intern_pool = _intern_pool[cls]
    instance = intern_pool.get(key, None)
    if instance is None:
      instance = super().__call__(*args, **kwargs)
      intern_pool[key] = instance
    return instance


# A per-`typing.Type` map from `__init__` arguments to object instances.
#
# This is used by the `_Intern` metaclass to allow reuse of object instances
# when new objects are requested with the same `__init__` arguments as
# existing object instances.
#
# Implementation note: this double-map is used rather than a single map
# stored as a field of each class because some class objects themselves would
# begin destruction before the map fields of other classes, causing errors
# during destruction.
_intern_pool: MutableMapping[type[Type], MutableMapping[Hashable, Type]] = (
    collections.defaultdict(dict)
)


def _clear_intern_pool() -> None:
  # We must clear our `WeakKeyValueDictionary`s at the end of the program to
  # prevent Python from deleting the standard library out from under us before
  # removing the  entries from the dictionary. Yes, this is cursed.
  #
  # If this isn't done, Python will call `__eq__` on our types after
  # `abc.ABCMeta` has already been deleted from the world, resulting in
  # exceptions after main.
  global _intern_pool
  _intern_pool = None


atexit.register(_clear_intern_pool)


_DtypeLike = Union[type[np.generic], np.dtype]


def _is_dtype_like(obj: object) -> TypeGuard[_DtypeLike]:
  """Returns `True` if `obj` is dtype like, otherwise `False`."""
  if isinstance(obj, type) and issubclass(obj, np.generic):
    return True
  else:
    return isinstance(obj, np.dtype)


_ALLOWED_NP_DTYPES = [
    np.bool_,
    np.bytes_,
    np.complex128,
    np.complex64,
    np.float16,
    np.float32,
    np.float64,
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.str_,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uint8,
]


def _is_allowed_np_dtype(dtype: np.dtype) -> bool:
  return dtype in _ALLOWED_NP_DTYPES


def _is_array_shape_like(
    obj: object,
) -> TypeGuard[Union[array_shape._ArrayShapeLike]]:
  """Returns `True` if `obj` is an `_ArrayShapeLike`, otherwise `False`."""
  if obj is None:
    return True
  elif isinstance(obj, Sequence):
    # If iterating over the `Sequence` fails, then `obj` is not an
    # `array_shape._ArrayShapeLike`.
    try:
      return all(isinstance(x, int) or x is None for x in obj)
    except Exception:  # pylint: disable=broad-exception-caught
      return False
  else:
    return False


def _to_dtype(dtype: _DtypeLike) -> np.dtype:
  """Returns a `np.dtype` for the dtype like object.

  Normalize `dtype` to an instance of `np.dtype` that describes an array
  scalar. see https://numpy.org/doc/stable/reference/arrays.scalars.html.

  Args:
    dtype: A dtype like object.
  """
  if isinstance(dtype, np.dtype):
    dtype = dtype.type
  if dtype == np.bytes_:
    dtype = np.str_
  dtype = np.dtype(dtype)

  if not _is_allowed_np_dtype(dtype):
    raise NotImplementedError(f'Unexpected `dtype` found: {dtype}.')
  return dtype


class TensorType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing types of tensors in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls,
      dtype: _DtypeLike,
      shape: array_shape._ArrayShapeLike = (),
  ) -> Hashable:
    """Returns hashable `TensorType.__init__` args."""
    dtype = _to_dtype(dtype)
    if shape is not None:
      shape = tuple(shape)
    return (dtype, shape)

  def __init__(
      self,
      dtype: _DtypeLike,
      shape: array_shape._ArrayShapeLike = (),
  ):
    """Constructs a new instance from the given `dtype` and `shape`.

    Args:
      dtype: The `np.dtype` of the array.
      shape: The shape of the array.

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    self._dtype = _to_dtype(dtype)
    if shape is not None:
      shape = tuple(shape)
    self._shape = shape

  def children(self) -> Iterator[Type]:
    return iter(())

  @property
  def dtype(self) -> np.dtype:
    return self._dtype

  @property
  def shape(self) -> array_shape.ArrayShape:
    return self._shape

  def __repr__(self):
    dtype_repr = f'np.{self._dtype.type.__name__}'
    if array_shape.is_shape_scalar(self._shape):
      return f'TensorType({dtype_repr})'
    else:
      return f'TensorType({dtype_repr}, {self._shape!r})'

  def __hash__(self):
    return hash((self._dtype, self._shape))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, TensorType)
        and self._dtype == other.dtype
        and self._shape == other.shape
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True

    if (
        not isinstance(source_type, TensorType)
        or self.dtype != source_type.dtype
    ):
      return False

    target_shape = self.shape
    source_shape = source_type.shape
    if target_shape is None:
      return True
    elif source_shape is None:
      return False

    if len(target_shape) != len(source_shape):
      return False

    def _dimension_is_assignable_from(target_dim, source_dim):
      return (target_dim is None) or (target_dim == source_dim)

    return all(
        _dimension_is_assignable_from(x, y)
        for x, y in zip(target_shape, source_shape)
    )


def _format_struct_type_members(struct_type: 'StructType') -> str:
  def _element_repr(element):
    name, value = element
    if name is not None:
      return "('{}', {!r})".format(name, value)
    return repr(value)

  return ', '.join(
      _element_repr(e) for e in structure.iter_elements(struct_type)
  )


def _to_named_types(
    elements: Iterable[object],
) -> Iterable[tuple[Optional[str], Type]]:
  """Creates an `Iterable` of optionally named types from `elements`.

  This function creates an `Iterable` of optionally named types by iterating
  over `elements` and normalizing each element.

  If `elements` is an `Iterable` with named elements (e.g. `Mapping` or
  `NamedTuple`), the normalize element will have a name equal to the name of the
  element and a value equal to the value of the element convereted to a type
  using `to_type`.

  If `elements` is an `Iterable` with unnamed elements (e.g. list), the
  normalized element will have a name of `None` and a value equal to the element
  convereted to a type using `to_type`.

  Note: This function treats a single element being passed in as `elements` as
  if it were an iterable of that element.

  Args:
    elements: An iterable of named or unnamed objects to convert to `tff.Types`.
      See `tff.types.to_type` for more information.

  Returns:
    An `Iterable` where each each element is `tuple[Optional[str], Type]`.
  """

  if py_typecheck.is_name_value_pair(elements):
    elements = [elements]
  elif isinstance(elements, py_typecheck.SupportsNamedTuple):
    elements = elements._asdict().items()
  elif isinstance(elements, Mapping):
    elements = elements.items()

  def _to_named_value_pair(element: object) -> tuple[Optional[str], Type]:
    if py_typecheck.is_name_value_pair(element):
      name, value = element
    else:
      name = None
      value = element
    value = to_type(value)
    return (name, value)

  return [_to_named_value_pair(x) for x in elements]


class StructType(structure.Struct, Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing structural types in TFF.

  Elements initialized by name can be accessed as `foo.name`, and otherwise by
  index, `foo[index]`.
  """

  @classmethod
  def _hashable_from_init_args(
      cls,
      elements: Iterable[object],
      *,
      convert: bool = True,
  ) -> Hashable:
    if convert:
      elements = _to_named_types(elements)
    return (tuple(elements), convert)

  def __init__(
      self,
      elements: Iterable[object],
      *,
      convert: bool = True,
  ):
    """Constructs a new instance from the given element types.

    Args:
      elements: An iterable of element specifications. Each element
        specification is either a type spec (an instance of `tff.Type` or
        something convertible to it via `tff.types.to_type`) for the element, or
        a (name, spec) for elements that have defined names. Alternatively, one
        can supply here an instance of `collections.OrderedDict` mapping element
        names to their types (or things that are convertible to types).
      convert: A flag to determine if the elements should be converted using
        `tff.types.to_type` or not.
    """
    if convert:
      elements = _to_named_types(elements)
    structure.Struct.__init__(self, elements)

  def children(self) -> Iterator[Type]:
    return (element for _, element in structure.iter_elements(self))

  @property
  def python_container(self) -> Optional[type[object]]:
    return None

  def __repr__(self):
    members = _format_struct_type_members(self)
    return f'StructType([{members}])'

  def __hash__(self):
    # Salt to avoid overlap.
    return hash((structure.Struct.__hash__(self), 'NTT'))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, StructType) and structure.Struct.__eq__(self, other)
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True
    if not isinstance(source_type, StructType):
      return False
    target_elements = structure.to_elements(self)
    source_elements = structure.to_elements(source_type)
    if len(target_elements) != len(source_elements):
      return False
    for (target_name, target_element), (source_name, source_element) in zip(
        target_elements, source_elements
    ):
      if source_name is not None and source_name != target_name:
        return False
      if not target_element.is_assignable_from(source_element):
        return False
    return True


class StructWithPythonType(StructType, metaclass=_Intern):
  """A representation of a structure paired with a Python container type."""

  @classmethod
  def _hashable_from_init_args(
      cls, elements: Iterable[object], container_type: type[object]
  ) -> Hashable:
    named_types = _to_named_types(elements)
    return (tuple(named_types), container_type)

  def __init__(self, elements: Iterable[object], container_type: type[object]):
    super().__init__(elements)
    self._container_type = container_type

  @property
  def python_container(self) -> type[object]:
    return self._container_type

  def __repr__(self):
    members = _format_struct_type_members(self)
    return 'StructType([{}]) as {}'.format(
        members, self._container_type.__name__
    )

  def __hash__(self):
    # Salt to avoid overlap.
    return hash((structure.Struct.__hash__(self), 'NTTWPCT'))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, StructWithPythonType)
        and (self._container_type == other._container_type)
        and structure.Struct.__eq__(self, other)
    )


class SequenceType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing types of sequences in TFF.

  IMPORTANT: since `SequenceType` is frequently backed by `tf.data.Dataset`
  which converts `list` to `tuple`, any `SequenceType` constructed with
  `StructWithPythonType` elements will convert any `list` python container type
  to `tuple` python container types for interoperability.
  """

  @classmethod
  def _hashable_from_init_args(cls, element: object) -> Hashable:
    element = to_type(element)
    return (element,)

  def __init__(self, element: object):
    """Constructs a new instance from the given `element` type.

    Args:
      element: A specification of the element type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
    """

    def convert_struct_with_list_to_struct_with_tuple(type_spec: T) -> T:
      """Convert any StructWithPythonType using lists to use tuples."""
      # We ignore non-struct, non-tensor types, these are not well formed types
      # for sequence elements.
      if not isinstance(type_spec, StructType):
        return type_spec
      elements = [
          (name, convert_struct_with_list_to_struct_with_tuple(value))
          for name, value in structure.iter_elements(type_spec)
      ]
      if not isinstance(type_spec, StructWithPythonType):
        return StructType(elements=elements)
      container_cls = type_spec.python_container
      return StructWithPythonType(
          elements=elements,
          container_type=tuple if container_cls is list else container_cls,
      )

    element = to_type(element)
    self._element = convert_struct_with_list_to_struct_with_tuple(element)

    children_types = _get_contained_children_types(self)
    if (
        children_types.federated
        or children_types.function
        or children_types.sequence
    ):
      raise ValueError(
          'Expected a `tff.SequenceType` to not contain `tff.FederatedType`s, '
          f'`tff.FunctionType`s, or `tff.SequenceType`s, found {self}.'
      )

  def children(self) -> Iterator[Type]:
    yield self._element

  @property
  def element(self) -> Type:
    return self._element

  def __repr__(self):
    return 'SequenceType({!r})'.format(self._element)

  def __hash__(self):
    return hash(self._element)

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, SequenceType) and self._element == other.element
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True
    return isinstance(
        source_type, SequenceType
    ) and self.element.is_assignable_from(source_type.element)


class FunctionType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing functional types in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls, parameter: Optional[object], result: object
  ) -> Hashable:
    if parameter is not None:
      parameter = to_type(parameter)
    result = to_type(result)
    return (parameter, result)

  def __init__(self, parameter: Optional[object], result: object):
    """Constructs a new instance from the given `parameter` and `result` types.

    Args:
      parameter: A specification of the parameter type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
        Multiple input arguments can be specified as a single `tff.StructType`.
      result: A specification of the result type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
    """
    if parameter is not None:
      parameter = to_type(parameter)
    self._parameter = parameter
    self._result = to_type(result)

  def children(self) -> Iterator[Type]:
    if self._parameter is not None:
      yield self._parameter
    yield self._result

  @property
  def parameter(self) -> Optional[Type]:
    return self._parameter

  @property
  def result(self) -> Type:
    return self._result

  def __repr__(self):
    return 'FunctionType({!r}, {!r})'.format(self._parameter, self._result)

  def __hash__(self):
    return hash((self._parameter, self._result))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, FunctionType)
        and self._parameter == other.parameter
        and self._result == other.result
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True
    if not isinstance(source_type, FunctionType):
      return False
    if (self.parameter is None) != (source_type.parameter is None):
      return False
    # Note that function parameters are contravariant, so we invert the check.
    if (
        self.parameter is not None
        and not source_type.parameter.is_assignable_from(self.parameter)
    ):
      return False
    return self.result.is_assignable_from(source_type.result)


class AbstractType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing abstract types in TFF."""

  @classmethod
  def _hashable_from_init_args(cls, label: str) -> Hashable:
    return (label,)

  def __init__(self, label: str):
    """Constructs a new instance from the given string `label`.

    Args:
      label: A string label of an abstract type. All occurrences of the label
        within a computation's type signature refer to the same concrete type.
    """
    self._label = label

  def children(self) -> Iterator[Type]:
    return iter(())

  @property
  def label(self) -> str:
    return self._label

  def __repr__(self):
    return "AbstractType('{}')".format(self._label)

  def __hash__(self):
    return hash(self._label)

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, AbstractType) and self._label == other.label
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    del source_type  # Unused.
    # TODO: b/113112108 - Revise this to extend the relation of assignability to
    # abstract types.
    raise TypeError('Abstract types are not comparable.')


class PlacementType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing the placement type in TFF.

  There is only one placement type, a TFF built-in, just as there is only one
  `int` or `str` type in Python. All instances of this class represent the same
  built-in TFF placement type.
  """

  @classmethod
  def _hashable_from_init_args(cls, *args, **kwargs) -> Hashable:
    del args, kwargs  # Unused.
    return ()

  def children(self) -> Iterator[Type]:
    return iter(())

  def __repr__(self):
    return 'PlacementType()'

  def __hash__(self):
    return 0

  def __eq__(self, other):
    return (self is other) or isinstance(other, PlacementType)

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True
    return isinstance(source_type, PlacementType)


class FederatedType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing federated types in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls,
      member: object,
      placement: placements.PlacementLiteral,
      all_equal: Optional[bool] = None,
  ) -> Hashable:
    member = to_type(member)
    return (member, placement, all_equal)

  def __init__(
      self,
      member: object,
      placement: placements.PlacementLiteral,
      all_equal: Optional[bool] = None,
  ):
    """Constructs a new federated type instance.

    Args:
      member: An instance of `tff.Type` or something convertible to it, that
        represents the type of the member components of each value of this
        federated type.
      placement: The specification of placement that the member components of
        this federated type are hosted on. Must be either a placement literal
        such as `tff.SERVER` or `tff.CLIENTS` to refer to a globally defined
        placement, or a placement label to refer to a placement defined in other
        parts of a type signature. Specifying placement labels is not
        implemented yet.
      all_equal: A `bool` value that indicates whether all members of the
        federated type are equal (`True`), or are allowed to differ (`False`).
        If `all_equal` is `None`, the value is selected as the default for the
        placement, e.g., `True` for `tff.SERVER` and `False` for `tff.CLIENTS`.
    """
    self._member = to_type(member)
    self._placement = placement
    if all_equal is None:
      all_equal = placement.default_all_equal
    self._all_equal = all_equal

    children_types = _get_contained_children_types(self)
    if children_types.federated or children_types.function:
      raise ValueError(
          'Expected a `tff.FederatedType` to not contain `tff.FederatedType`s '
          f'or `tff.FunctionType`s, found {self}.'
      )

  # TODO: b/113112108 - Extend this to support federated types parameterized
  # by abstract placement labels, such as those used in generic types of
  # federated operators.

  def children(self) -> Iterator[Type]:
    yield self._member

  @property
  def member(self) -> Type:
    return self._member

  @property
  def placement(self) -> placements.PlacementLiteral:
    return self._placement

  @property
  def all_equal(self) -> bool:
    return self._all_equal

  def __repr__(self):
    return 'FederatedType({!r}, {!r}, {!r})'.format(
        self._member, self._placement, self._all_equal
    )

  def __hash__(self):
    return hash((self._member, self._placement, self._all_equal))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, FederatedType)
        and self._member == other.member
        and self._placement == other.placement
        and self._all_equal == other.all_equal
    )

  def is_assignable_from(self, source_type: Type) -> bool:
    if self is source_type:
      return True
    return (
        isinstance(source_type, FederatedType)
        and self.member.is_assignable_from(source_type.member)
        and (not self.all_equal or source_type.all_equal)
        and self.placement is source_type.placement
    )


def to_type(obj: object) -> Type:
  """Converts the argument into an instance of `tff.Type`.

  Examples of arguments convertible to tensor types:

  ```python
  np.int32
  (np.int32, [10])
  (np.int32, [None])
  ```

  Examples of arguments convertible to flat named tuple types:

  ```python
  [np.int32, np.bool]
  (np.int32, np.bool)
  [('a', np.int32), ('b', np.bool)]
  ('a', np.int32)
  collections.OrderedDict([('a', np.int32), ('b', np.bool)])
  ```

  Examples of arguments convertible to nested named tuple types:

  ```python
  (np.int32, (np.float32, np.bool))
  (np.int32, (('x', np.float32), np.bool))
  ((np.int32, [1]), (('x', (np.float32, [2])), (np.bool, [3])))
  ```

  `attr.s` class instances can also be used to describe TFF types by populating
  the fields with the corresponding types:

  ```python
  @attr.s(auto_attribs=True)
  class MyDataClass:
    int_scalar
    string_array

  obj = MyDataClass(...)
  type_spec = tff.types.to_type(obj)

  @tff.tensorflow.computation(type_spec)
  def work(my_data):
    assert isinstance(my_data, MyDataClass)
    ...
  ```

  Args:
    obj: Either an instance of `tff.Type`, or an argument convertible to
      `tff.Type`.

  Returns:
    An instance of `tff.Type` corresponding to the given `obj`.
  """
  # TODO: b/113112108 - Add multiple examples of valid type specs here in the
  # comments, in addition to the unit test.
  if isinstance(obj, Type):
    return obj
  elif _is_dtype_like(obj):
    return TensorType(obj)  # pytype: disable=wrong-arg-types  # b/290661340
  elif (
      isinstance(obj, tuple)
      and len(obj) == 2
      and _is_dtype_like(obj[0])
      and _is_array_shape_like(obj[1])
  ):
    dtype, shape = obj
    return TensorType(dtype, shape)
  elif isinstance(obj, (list, tuple)):
    if any(py_typecheck.is_name_value_pair(e, name_type=str) for e in obj):
      # The sequence has a (name, value) elements, the whole sequence is most
      # likely intended to be a `Struct`, do not store the Python container.
      return StructType(obj)
    else:
      return StructWithPythonType(obj, type(obj))
  elif attrs.has(type(obj)):
    return StructWithPythonType(attrs.asdict(obj, recurse=False), type(obj))
  elif isinstance(obj, Mapping):
    return StructWithPythonType(obj, type(obj))
  elif isinstance(obj, structure.Struct):
    return StructType(structure.to_elements(obj))
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a type spec.'.format(
            py_typecheck.type_string(type(obj))
        )
    )


def _tensorflow_dtype_to_numpy_dtype(
    dtype: tf.dtypes.DType,
) -> type[np.generic]:
  """Returns a numpy dtype for the `dtype`."""
  # TensorFlow converts a dtype of `tf.string` into a dtype of `np.object_`.
  # However, this is not a valid dtype for TFF, instead use `np.str_`.
  if dtype.base_dtype == tf.string:
    dtype = np.str_
  else:
    dtype = dtype.base_dtype.as_numpy_dtype
  return dtype


def _tensor_shape_to_array_shape(
    tensor_shape: tf.TensorShape,
) -> array_shape.ArrayShape:
  """Returns a `tff.types.ArrayShape` for the `tensor_shape`."""
  if tensor_shape.rank is not None:
    shape = tensor_shape.as_list()
  else:
    shape = None
  return shape


def _tensor_spec_to_type(tensor_spec: tf.TensorSpec) -> Type:
  """Returns a `tff.Type` for the `tensor_spec`."""
  dtype = _tensorflow_dtype_to_numpy_dtype(tensor_spec.dtype)
  shape = _tensor_shape_to_array_shape(tensor_spec.shape)
  return TensorType(dtype, shape)


def _dataset_spec_to_type(dataset_spec: tf.data.DatasetSpec) -> Type:
  """Returns a `tff.Type` for the `dataset_spec`."""
  element_type = tensorflow_to_type(dataset_spec.element_spec)
  return SequenceType(element_type)


def tensorflow_to_type(obj: object) -> Type:
  """Returns a `tff.Type` for an `obj` containing TensorFlow type specs.

  This function extends `tff.types.to_type` to handle TensorFlow type specs and
  Python structures containing TensorFlow type specs:

  *   `tf.dtypes.DType`
  *   tensor-like objects (e.g. `(tf.int32, tf.TensorShape([2, 3]))`)
  *   `tf.TensorSpec`
  *   `tf.data.DatasetSpec`

  For example:

  >>> tensorflow_to_type(tf.int32)
  tff.TensorType(np.int32)

  >>> tensorflow_to_type((tf.int32, tf.TensorShape([2, 3])))
  tff.TensorType(np.int32, (2, 3))

  >>> spec = tf.TensorSpec(shape=[2, 3], dtype=tf.int32)
  >>> tensorflow_to_type(spec)
  tff.TensorType(np.int32, (2, 3))

  >>> spec = tf.data.DatasetSpec(tf.TensorSpec([2, 3], dtype=tf.int32))
  >>> tensorflow_to_type(spec)
  tff.SequenceType(tff.TensorType(np.int32, (2, 3)))

  Args:
    obj: A `tff.Type` or an argument convertible to a `tff.Type`.
  """

  def _to_type(obj):
    if isinstance(obj, tf.dtypes.DType):
      return _tensorflow_dtype_to_numpy_dtype(obj)
    elif isinstance(obj, tf.TensorShape):
      shape = _tensor_shape_to_array_shape(obj)
      if shape is None:
        return tree.MAP_TO_NONE
      else:
        return shape
    elif isinstance(obj, tf.TensorSpec):
      return _tensor_spec_to_type(obj)
    elif isinstance(obj, tf.data.DatasetSpec):
      return _dataset_spec_to_type(obj)
    else:
      return None

  partial_type = tree.traverse(_to_type, obj)
  return to_type(partial_type)


@attrs.define(frozen=True)
class _ContainedChildrenTypes:
  """The types of children `tff.Types` contained by a `tff.Type`.

  This data structure is used by `_get_contained_children_types` to package
  the types of children `tff.Types` contained by a `tff.Type` in a more
  convenient way.
  """

  tensor: bool = False
  struct: bool = False
  struct_with_python_type: bool = False
  sequence: bool = False
  function: bool = False
  abstract: bool = False
  placement: bool = False
  federated: bool = False


# Manual cache used rather than `cachetools.cached` due to incompatibility
# with `WeakKeyDictionary`. We want to use a `WeakKeyDictionary` so that
# cache entries are destroyed once the types they index no longer exist.
_contained_children_types_cache: MutableMapping[
    Type, _ContainedChildrenTypes
] = weakref.WeakKeyDictionary({})


def _clear_contained_children_types_cache():
  # We must clear our `WeakKeyValueDictionary`s at the end of the program to
  # prevent Python from deleting the standard library out from under us before
  # removing the  entries from the dictionary. Yes, this is cursed.
  #
  # If this isn't done, Python will call `__eq__` on our types after
  # `abc.ABCMeta` has already been deleted from the world, resulting in
  # exceptions after main.
  global _contained_children_types_cache
  _contained_children_types_cache = None


atexit.register(_clear_contained_children_types_cache)


def _get_contained_children_types(type_spec: Type) -> _ContainedChildrenTypes:
  """Returns the types of children `tff.Types` contained by `type_spec`.

  The `_ContainedChildrenTypes` is cached so that this function can be used in
  performance sensitive operations.

  Args:
    type_spec: A `tff.Type`.

  Raises:
    RuntimeError: If the cache becomes corrupted in some unexpected way.
  """
  if _contained_children_types_cache is None:
    raise RuntimeError('Unexpected runtime error.')
  children_types = _contained_children_types_cache.get(type_spec, None)
  if children_types is not None:
    return children_types

  children_types = _ContainedChildrenTypes()
  for child_type in type_spec.children():
    # Create a mutable dict from the frozen `_ContainedChildrenTypes` instance;
    # add the child and grandchildren updates; and then evolve the instance.
    updates = attrs.asdict(children_types)
    if isinstance(child_type, TensorType):
      updates['tensor'] = True
    elif isinstance(child_type, StructType):
      updates['struct'] = True
    elif isinstance(child_type, StructWithPythonType):
      updates['struct_with_python_type'] = True
    elif isinstance(child_type, SequenceType):
      updates['sequence'] = True
    elif isinstance(child_type, FunctionType):
      updates['function'] = True
    elif isinstance(child_type, AbstractType):
      updates['abstract'] = True
    elif isinstance(child_type, PlacementType):
      updates['placement'] = True
    elif isinstance(child_type, FederatedType):
      updates['federated'] = True
    else:
      raise NotImplementedError(f'Unexpected type found: {type(child_type)}.')
    grandchildren_types = _get_contained_children_types(child_type)
    for key, value in attrs.asdict(grandchildren_types).items():
      if value:
        updates[key] = True
    children_types = attrs.evolve(children_types, **updates)

  _contained_children_types_cache[type_spec] = children_types
  return children_types


def _string_representation(type_spec: Type, formatted: bool) -> str:
  """Returns the string representation of a TFF `Type`.

  This function creates a `list` of strings representing the given `type_spec`;
  combines the strings in either a formatted or un-formatted representation; and
  returns the resulting string representation.

  Args:
    type_spec: An instance of a TFF `Type`.
    formatted: A boolean indicating if the returned string should be formatted.

  Raises:
    TypeError: If `type_spec` has an unexpected type.
  """

  def _combine(components):
    """Returns a `list` of strings by combining `components`.

    This function creates and returns a `list` of strings by combining a `list`
    of `components`. Each `component` is a `list` of strings representing a part
    of the string of a TFF `Type`. The `components` are combined by iteratively
    **appending** the last element of the result with the first element of the
    `component` and then **extending** the result with remaining elements of the
    `component`.

    For example:

    >>> _combine([['a'], ['b'], ['c']])
    ['abc']

    >>> _combine([['a', 'b', 'c'], ['d', 'e', 'f']])
    ['abcd', 'ef']

    This function is used to help track where new-lines should be inserted into
    the string representation if the lines are formatted.

    Args:
      components: A `list` where each element is a `list` of strings
        representing a part of the string of a TFF `Type`.
    """
    lines = ['']
    for component in components:
      lines[-1] = '{}{}'.format(lines[-1], component[0])
      lines.extend(component[1:])
    return lines

  def _indent(lines, indent_chars='  '):
    """Returns an indented `list` of strings."""
    return ['{}{}'.format(indent_chars, e) for e in lines]

  def _lines_for_named_types(named_type_specs, formatted):
    """Returns a `list` of strings representing the given `named_type_specs`.

    Args:
      named_type_specs: A `list` of named computations, each being a pair
        consisting of a name (either a string, or `None`) and a
        `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    lines = []
    for index, (name, type_spec) in enumerate(named_type_specs):
      if index != 0:
        if formatted:
          lines.append([',', ''])
        else:
          lines.append([','])
      element_lines = _lines_for_type(type_spec, formatted)
      if name is not None:
        element_lines = _combine([
            ['{}='.format(name)],
            element_lines,
        ])
      lines.append(element_lines)
    return _combine(lines)

  def _lines_for_type(type_spec, formatted):
    """Returns a `list` of strings representing the given `type_spec`.

    Args:
      type_spec: An instance of a TFF `Type`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    if isinstance(type_spec, AbstractType):
      return [type_spec.label]
    elif isinstance(type_spec, FederatedType):
      member_lines = _lines_for_type(type_spec.member, formatted)
      placement_line = '@{}'.format(type_spec.placement)
      if type_spec.all_equal:
        return _combine([member_lines, [placement_line]])
      else:
        return _combine([['{'], member_lines, ['}'], [placement_line]])
    elif isinstance(type_spec, FunctionType):
      if type_spec.parameter is not None:
        parameter_lines = _lines_for_type(type_spec.parameter, formatted)
      else:
        parameter_lines = ['']
      result_lines = _lines_for_type(type_spec.result, formatted)
      return _combine([['('], parameter_lines, [' -> '], result_lines, [')']])
    elif isinstance(type_spec, StructType):
      if not type_spec:
        return ['<>']
      elements = structure.to_elements(type_spec)
      elements_lines = _lines_for_named_types(elements, formatted)
      if formatted:
        elements_lines = _indent(elements_lines)
        lines = [['<', ''], elements_lines, ['', '>']]
      else:
        lines = [['<'], elements_lines, ['>']]
      return _combine(lines)
    elif isinstance(type_spec, PlacementType):
      return ['placement']
    elif isinstance(type_spec, SequenceType):
      element_lines = _lines_for_type(type_spec.element, formatted)
      return _combine([element_lines, ['*']])
    elif isinstance(type_spec, TensorType):
      if type_spec.shape is None:
        return ['{!r}(shape=None)'.format(type_spec.dtype.name)]
      elif type_spec.shape:
        def _value_string(value):
          return str(value) if value is not None else '?'

        value_strings = [_value_string(e) for e in type_spec.shape]
        values_strings = ','.join(value_strings)
        return ['{}[{}]'.format(type_spec.dtype.name, values_strings)]
      else:
        return [type_spec.dtype.name]
    else:
      raise NotImplementedError(
          'Unexpected type found: {}.'.format(type(type_spec))
      )

  lines = _lines_for_type(type_spec, formatted)
  lines = [line.rstrip() for line in lines]
  if formatted:
    return '\n'.join(lines)
  else:
    return ''.join(lines)
