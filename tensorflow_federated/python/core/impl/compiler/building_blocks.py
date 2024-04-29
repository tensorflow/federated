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
"""A library of classes representing computations in a deserialized form."""

import abc
from collections.abc import Iterable, Iterator
import enum
import typing
from typing import Optional, Union
import zlib

import numpy as np

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import array
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import typed_object


def _check_computation_oneof(
    computation_proto: pb.Computation,
    expected_oneof: str,
):
  """Checks that `computation_proto` is a oneof of the expected variant."""
  computation_oneof = computation_proto.WhichOneof('computation')
  if computation_oneof != expected_oneof:
    raise ValueError(
        f'Expected the computation to be a {expected_oneof}, found'
        f' {computation_oneof}.'
    )


class UnexpectedBlockError(TypeError):

  def __init__(
      self,
      expected: type['ComputationBuildingBlock'],
      actual: 'ComputationBuildingBlock',
  ):
    message = f'Expected block of kind {expected}, found block {actual}'
    super().__init__(message)
    self.actual = actual
    self.expected = expected


class ComputationBuildingBlock(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """The abstract base class for abstractions in the TFF's internal language.

  Instances of this class correspond roughly one-to-one to the abstractions
  defined in the `Computation` message in TFF's `computation.proto`, and are
  intended primarily for the ease of manipulating the abstract syntax trees
  (AST) of federated computations as they are transformed by TFF's compiler
  pipeline to mold into the needs of a particular execution backend. The only
  abstraction that does not have a dedicated Python equivalent is a section
  of TensorFlow code (it's represented by `tff.framework.CompiledComputation`).
  """

  @classmethod
  def from_proto(
      cls, computation_proto: pb.Computation
  ) -> 'ComputationBuildingBlock':
    """Returns an instance of a derived class based on 'computation_proto'.

    Args:
      computation_proto: An instance of pb.Computation.

    Returns:
      An instance of a class that implements 'ComputationBuildingBlock' and
      that contains the deserialized logic from in 'computation_proto'.

    Raises:
      NotImplementedError: if computation_proto contains a kind of computation
        for which deserialization has not been implemented yet.
      ValueError: if deserialization failed due to the argument being invalid.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    computation_oneof = computation_proto.WhichOneof('computation')
    deserializer = _deserializer_dict.get(computation_oneof)
    if deserializer is not None:
      deserialized = deserializer(computation_proto)
      type_spec = type_serialization.deserialize_type(computation_proto.type)
      if not deserialized.type_signature.is_equivalent_to(type_spec):
        raise ValueError(
            'The type {} derived from the computation structure does not '
            'match the type {} declared in its signature'.format(
                deserialized.type_signature, type_spec
            )
        )
      return deserialized
    else:
      raise NotImplementedError(
          'Deserialization for computations of type {} has not been '
          'implemented yet.'.format(computation_oneof)
      )
    return deserializer(computation_proto)

  def __init__(self, type_spec):
    """Constructs a computation building block with the given TFF type.

    Args:
      type_spec: An instance of types.Type, or something convertible to it via
        types.to_type().
    """
    type_signature = computation_types.to_type(type_spec)
    self._type_signature = type_signature
    self._cached_hash = None
    self._cached_proto = None

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  @abc.abstractmethod
  def children(self) -> Iterator['ComputationBuildingBlock']:
    """Returns an iterator yielding immediate child building blocks."""
    raise NotImplementedError

  def compact_representation(self):
    """Returns the compact string representation of this building block."""
    return _string_representation(self, formatted=False)

  def formatted_representation(self):
    """Returns the formatted string representation of this building block."""
    return _string_representation(self, formatted=True)

  def structural_representation(self):
    """Returns the structural string representation of this building block."""
    return _structural_representation(self)

  @property
  def proto(self):
    """Returns a serialized form of this object as a pb.Computation instance."""
    if self._cached_proto is None:
      self._cached_proto = self._proto()
    return self._cached_proto

  @abc.abstractmethod
  def _proto(self):
    """Uncached, internal version of `proto`."""
    raise NotImplementedError

  # TODO: b/113112885 - Add memoization after identifying a suitable externally
  # available standard library that works in Python 2/3.

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this computation building block."""
    raise NotImplementedError

  def __str__(self):
    """Returns a concise representation of this computation building block."""
    return self.compact_representation()


class Reference(ComputationBuildingBlock):
  """A reference to a name defined earlier in TFF's internal language.

  Names are defined by lambda expressions (which have formal named parameters),
  and block structures (which can have one or more locals). The reference
  construct is used to refer to those parameters or locals by a string name.
  The usual hiding rules apply. A reference binds to the closest definition of
  the given name in the most deeply nested surrounding lambda or block.

  A concise notation for a reference to name `foo` is `foo`. For example, in
  a lambda expression `(x -> f(x))` there are two references, one to `x` that
  is defined as the formal parameter of the lambda epxression, and one to `f`
  that must have been defined somewhere in the surrounding context.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Reference':
    _check_computation_oneof(computation_proto, 'reference')
    return cls(
        str(computation_proto.reference.name),
        type_serialization.deserialize_type(computation_proto.type),
    )

  def __init__(self, name: str, type_spec: object, context=None):
    """Creates a reference to 'name' of type 'type_spec' in context 'context'.

    Args:
      name: The name of the referenced entity.
      type_spec: The type spec of the referenced entity.
      context: The optional context in which the referenced entity is defined.
        This class does not prescribe what Python type the 'context' needs to be
        and merely exposes it as a property (see below). The only requirement is
        that the context implements str() and repr().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(name, str)
    super().__init__(type_spec)
    self._name = name
    self._context = context

  def _proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        reference=pb.Reference(name=self._name),
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def name(self) -> str:
    return self._name

  @property
  def context(self):
    return self._context

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Reference):
      return NotImplemented
    # Important: References are only equal to each other if they are the same
    # object because two references with the same `name` are different if they
    # are in different locations within the same scope, in different scopes, or
    # in different contexts.
    return False

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((self._name, self._type_signature))
    return self._cached_hash

  def __repr__(self):
    return "Reference('{}', {!r}{})".format(
        self._name,
        self.type_signature,
        ', {!r}'.format(self._context) if self._context is not None else '',
    )


class Selection(ComputationBuildingBlock):
  """A selection by name or index from a struct-typed value in TFF's language.

  The concise syntax for selections is `foo.bar` (selecting a named `bar` from
  the value of expression `foo`), and `foo[n]` (selecting element at index `n`
  from the value of `foo`).
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Selection':
    _check_computation_oneof(computation_proto, 'selection')
    selection = ComputationBuildingBlock.from_proto(
        computation_proto.selection.source
    )
    return cls(selection, index=computation_proto.selection.index)

  def __init__(
      self,
      source: ComputationBuildingBlock,
      name: Optional[str] = None,
      index: Optional[int] = None,
  ):
    """A selection from 'source' by a string or numeric 'name_or_index'.

    Exactly one of 'name' or 'index' must be specified (not None).

    Args:
      source: The source value to select from (an instance of
        ComputationBuildingBlock).
      name: A string name of the element to be selected.
      index: A numeric index of the element to be selected.

    Raises:
      TypeError: if arguments are of the wrong types.
      ValueError: if the name is empty or index is negative, or the name/index
        is not compatible with the type signature of the source, or neither or
        both are defined (not None).
    """
    py_typecheck.check_type(source, ComputationBuildingBlock)
    source_type = source.type_signature
    # TODO: b/224484886 - Downcasting to all handled types.
    source_type = typing.cast(Union[computation_types.StructType], source_type)
    if not isinstance(source_type, computation_types.StructType):
      raise TypeError(
          'Expected the source of selection to be a TFF struct, '
          'instead found it to be of type {}.'.format(source_type)
      )
    if name is not None and index is not None:
      raise ValueError(
          'Cannot simultaneously specify a name and an index, choose one.'
      )
    if name is not None:
      py_typecheck.check_type(name, str)
      if not name:
        raise ValueError('The name of the selected element cannot be empty.')
      # Normalize, in case we are dealing with a Unicode type or some such.
      name = str(name)
      if not structure.has_field(source_type, name):
        raise ValueError(
            f'Error selecting named field `{name}` from type `{source_type}`, '
            f'whose only named fields are {structure.name_list(source_type)}.'
        )
      type_signature = source_type[name]
    elif index is not None:
      py_typecheck.check_type(index, int)
      length = len(source_type)
      if index < 0 or index >= length:
        raise ValueError(
            f'The index `{index}` does not fit into the valid range in the '
            f'struct type: 0..{length}'
        )
      type_signature = source_type[index]
    else:
      raise ValueError(
          'Must define either a name or index, and neither was specified.'
      )
    super().__init__(type_signature)
    self._source = source
    self._name = name
    self._index = index

  def _proto(self):
    selection = pb.Selection(source=self._source.proto, index=self.as_index())
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        selection=selection,
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._source

  @property
  def source(self) -> ComputationBuildingBlock:
    return self._source

  @property
  def name(self) -> Optional[str]:
    return self._name

  @property
  def index(self) -> Optional[int]:
    return self._index

  def as_index(self) -> int:
    if self._index is not None:
      return self._index
    else:
      field_to_index = structure.name_to_index_map(self.source.type_signature)  # pytype: disable=wrong-arg-types
      return field_to_index[self._name]

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Selection):
      return NotImplemented
    return (
        self._source,
        self._name,
        self._index,
    ) == (
        other._source,
        other._name,
        other._index,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((self._source, self._name, self._index))
    return self._cached_hash

  def __repr__(self):
    if self._name is not None:
      return "Selection({!r}, name='{}')".format(self._source, self._name)
    else:
      return 'Selection({!r}, index={})'.format(self._source, self._index)


class Struct(ComputationBuildingBlock, structure.Struct):
  """A struct with named or unnamed elements in TFF's internal language.

  The concise notation for structs is `<name_1=value_1, ...., name_n=value_n>`
  for structs with named elements, `<value_1, ..., value_n>` for structs with
  unnamed elements, or a mixture of these for structs with some named and some
  unnamed elements, where `name_k` are the names, and `value_k` are the value
  expressions.

  For example, a lambda expression that applies `fn` to elements of 2-structs
  pointwise could be represented as `(arg -> <fn(arg[0]),fn(arg[1])>)`.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Struct':
    _check_computation_oneof(computation_proto, 'struct')

    def _element(
        proto: pb.Struct.Element,
    ) -> tuple[Optional[str], ComputationBuildingBlock]:
      if proto.name:
        name = str(proto.name)
      else:
        name = None
      element = ComputationBuildingBlock.from_proto(proto.value)
      return (name, element)

    elements = [_element(x) for x in computation_proto.struct.element]
    return cls(elements)

  def __init__(self, elements, container_type=None):
    """Constructs a struct from the given list of elements.

    Args:
      elements: The elements of the struct, supplied as a list of (name, value)
        pairs, where 'name' can be None in case the corresponding element is not
        named and only accessible via an index (see also `structure.Struct`).
      container_type: An optional Python container type to associate with the
        struct.

    Raises:
      TypeError: if arguments are of the wrong types.
    """

    # Not using super() here and below, as the two base classes have different
    # signatures of their constructors, and the struct implementation
    # of selection interfaces should override that in the generic class 'Value'
    # to favor simplified expressions where simplification is possible.
    def _map_element(e):
      """Returns a named or unnamed element."""
      if isinstance(e, ComputationBuildingBlock):
        return (None, e)
      elif py_typecheck.is_name_value_pair(
          e, value_type=ComputationBuildingBlock
      ):
        if e[0] is not None and not e[0]:
          raise ValueError('Unexpected struct element with empty string name.')
        return (e[0], e[1])
      else:
        raise TypeError('Unexpected struct element: {}.'.format(e))

    elements = [_map_element(e) for e in elements]
    element_pairs = [
        ((e[0], e[1].type_signature) if e[0] else e[1].type_signature)
        for e in elements
    ]

    if container_type is None:
      type_signature = computation_types.StructType(element_pairs)
    else:
      type_signature = computation_types.StructWithPythonType(
          element_pairs, container_type
      )
    ComputationBuildingBlock.__init__(self, type_signature)
    structure.Struct.__init__(self, elements)
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.StructType:
    return self._type_signature

  def _proto(self):
    elements = []
    for k, v in structure.iter_elements(self):
      if k is not None:
        element = pb.Struct.Element(name=k, value=v.proto)
      else:
        element = pb.Struct.Element(value=v.proto)
      elements.append(element)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        struct=pb.Struct(element=elements),
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    return (element for _, element in structure.iter_elements(self))

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Struct):
      return NotImplemented
    if self._type_signature != other._type_signature:
      return False
    return structure.Struct.__eq__(self, other)

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((
          structure.Struct.__hash__(self),
          self._type_signature,
      ))
    return self._cached_hash

  def __repr__(self):
    def _element_repr(element):
      name, value = element
      name_repr = "'{}'".format(name) if name is not None else 'None'
      return '({}, {!r})'.format(name_repr, value)

    return 'Struct([{}])'.format(
        ', '.join(_element_repr(e) for e in structure.iter_elements(self))
    )


class Call(ComputationBuildingBlock):
  """A representation of a function invocation in TFF's internal language.

  The call construct takes an argument struct with two elements, the first being
  the function to invoke (represented as a computation with a functional result
  type), and the second being the argument to feed to that function. Typically,
  the function is either a TFF instrinsic, or a lambda expression.

  The concise notation for calls is `foo(bar)`, where `foo` is the function,
  and `bar` is the argument.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Call':
    _check_computation_oneof(computation_proto, 'call')
    fn = ComputationBuildingBlock.from_proto(computation_proto.call.function)
    arg_proto = computation_proto.call.argument
    if arg_proto.WhichOneof('computation') is not None:
      arg = ComputationBuildingBlock.from_proto(arg_proto)
    else:
      arg = None
    return cls(fn, arg)

  def __init__(
      self,
      fn: ComputationBuildingBlock,
      arg: Optional[ComputationBuildingBlock] = None,
  ):
    """Creates a call to 'fn' with argument 'arg'.

    Args:
      fn: A value of a functional type that represents the function to invoke.
      arg: The optional argument, present iff 'fn' expects one, of a type that
        matches the type of 'fn'.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(fn, ComputationBuildingBlock)
    if arg is not None:
      py_typecheck.check_type(arg, ComputationBuildingBlock)
    function_type = fn.type_signature
    # TODO: b/224484886 - Downcasting to all handled types.
    function_type = typing.cast(
        Union[computation_types.FunctionType], function_type
    )
    if not isinstance(function_type, computation_types.FunctionType):
      raise TypeError(
          f'Expected `fn` to have a `tff.FunctionType`, found {function_type}.'
      )
    parameter_type = function_type.parameter
    if parameter_type is not None:
      if arg is None:
        raise TypeError(
            f'Expected `arg` to be of type {parameter_type}, found None.'
        )
      elif not parameter_type.is_assignable_from(arg.type_signature):
        raise TypeError(
            f'Expected `arg` to be of type {parameter_type}, found an'
            f' incompatible type {arg.type_signature}.'
        )
    else:
      if arg is not None:
        raise TypeError(f'Expected `arg` to be None, found {arg}.')
    super().__init__(function_type.result)
    self._function = fn
    self._argument = arg

  def _proto(self):
    if self._argument is not None:
      call = pb.Call(
          function=self._function.proto, argument=self._argument.proto
      )
    else:
      call = pb.Call(function=self._function.proto)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature), call=call
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._function
    if self._argument is not None:
      yield self._argument

  @property
  def function(self) -> ComputationBuildingBlock:
    return self._function

  @property
  def argument(self) -> Optional[ComputationBuildingBlock]:
    return self._argument

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Call):
      return NotImplemented
    return (
        self._function,
        self._argument,
    ) == (
        other._function,
        other._argument,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((self._function, self._argument))
    return self._cached_hash

  def __repr__(self):
    if self._argument is not None:
      return 'Call({!r}, {!r})'.format(self._function, self._argument)
    else:
      return 'Call({!r})'.format(self._function)


class Lambda(ComputationBuildingBlock):
  """A representation of a lambda expression in TFF's internal language.

  A lambda expression consists of a string formal parameter name, and a result
  expression that can contain references by name to that formal parameter. A
  concise notation for lambdas is `(foo -> bar)`, where `foo` is the name of
  the formal parameter, and `bar` is the result expression.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Lambda':
    _check_computation_oneof(computation_proto, 'lambda')
    fn: pb.Lambda = getattr(computation_proto, 'lambda')
    if computation_proto.type.function.HasField('parameter'):
      parameter_type = type_serialization.deserialize_type(
          computation_proto.type.function.parameter
      )
    else:
      parameter_type = None
    result = ComputationBuildingBlock.from_proto(fn.result)
    return cls(fn.parameter_name, parameter_type, result)

  def __init__(
      self,
      parameter_name: Optional[str],
      parameter_type: Optional[object],
      result: ComputationBuildingBlock,
  ):
    """Creates a lambda expression.

    Args:
      parameter_name: The (string) name of the parameter accepted by the lambda.
        This name can be used by Reference() instances in the body of the lambda
        to refer to the parameter. Note that an empty parameter name shall be
        treated as equivalent to no parameter.
      parameter_type: The type of the parameter, an instance of types.Type or
        something convertible to it by types.to_type().
      result: The resulting value produced by the expression that forms the body
        of the lambda. Must be an instance of ComputationBuildingBlock.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    if not parameter_name:
      parameter_name = None
    if (parameter_name is None) != (parameter_type is None):
      raise TypeError(
          'A lambda expression must have either a valid parameter name and '
          'type or both parameter name and type must be `None`. '
          '`parameter_name` was {} but `parameter_type` was {}.'.format(
              parameter_name, parameter_type
          )
      )
    if parameter_name is not None:
      py_typecheck.check_type(parameter_name, str)
      parameter_type = computation_types.to_type(parameter_type)
    py_typecheck.check_type(result, ComputationBuildingBlock)
    type_signature = computation_types.FunctionType(
        parameter_type, result.type_signature
    )
    super().__init__(type_signature)
    self._parameter_name = parameter_name
    self._parameter_type = parameter_type
    self._result = result
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.FunctionType:
    return self._type_signature

  def _proto(self) -> pb.Computation:
    type_signature = type_serialization.serialize_type(self.type_signature)
    fn = pb.Lambda(
        parameter_name=self._parameter_name, result=self._result.proto
    )
    # We are unpacking the lambda argument here because `lambda` is a reserved
    # keyword in Python, but it is also the name of the parameter for a
    # `pb.Computation`.
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
    return pb.Computation(type=type_signature, **{'lambda': fn})  # pytype: disable=wrong-keyword-args

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._result

  @property
  def parameter_name(self) -> Optional[str]:
    return self._parameter_name

  @property
  def parameter_type(self) -> Optional[computation_types.Type]:
    return self._parameter_type

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Lambda):
      return NotImplemented
    return (
        self._parameter_name,
        self._parameter_type,
        self._result,
    ) == (
        other._parameter_name,
        other._parameter_type,
        other._result,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((
          self._parameter_name,
          self._parameter_type,
          self._result,
      ))
    return self._cached_hash

  def __repr__(self) -> str:
    return "Lambda('{}', {!r}, {!r})".format(
        self._parameter_name, self._parameter_type, self._result
    )


class Block(ComputationBuildingBlock):
  """A representation of a block of code in TFF's internal language.

  A block is a syntactic structure that consists of a sequence of local name
  bindings followed by a result. The bindings are interpreted sequentially,
  with bindings later in the sequence in the scope of those listed earlier,
  and the result in the scope of the entire sequence. The usual hiding rules
  apply.

  An informal concise notation for blocks is the following, with `name_k`
  representing the names defined locally for the block, `value_k` the values
  associated with them, and `result` being the expression that reprsents the
  value of the block construct.

  ```
  let name_1=value_1, name_2=value_2, ..., name_n=value_n in result
  ```

  Blocks are technically a redundant abstraction, as they can be equally well
  represented by lambda expressions. A block of the form `let x=y in z` is
  roughly equivalent to `(x -> z)(y)`. Although redundant, blocks have a use
  as a way to reduce TFF computation ASTs to a simpler, less nested and more
  readable form, and are helpful in AST transformations as a mechanism that
  prevents possible naming conflicts.

  An example use of a block expression to flatten a nested structure below:

  ```
  z = federated_sum(federated_map(x, federated_broadcast(y)))
  ```

  An equivalent form in a more sequential notation using a block expression:
  ```
  let
    v1 = federated_broadcast(y),
    v2 = federated_map(x, v1)
  in
    federated_sum(v2)
  ```
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Block':
    _check_computation_oneof(computation_proto, 'block')
    return cls(
        [
            (str(loc.name), ComputationBuildingBlock.from_proto(loc.value))
            for loc in computation_proto.block.local
        ],
        ComputationBuildingBlock.from_proto(computation_proto.block.result),
    )

  def __init__(
      self,
      local_symbols: Iterable[tuple[str, ComputationBuildingBlock]],
      result: ComputationBuildingBlock,
  ):
    """Creates a block of TFF code.

    Args:
      local_symbols: The list of one or more local declarations, each of which
        is a 2-tuple (name, value), with 'name' being the string name of a local
        symbol being defined, and 'value' being the instance of
        ComputationBuildingBlock, the output of which will be locally bound to
        that name.
      result: An instance of ComputationBuildingBlock that computes the result.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    updated_locals = []
    for index, element in enumerate(local_symbols):
      if (
          not isinstance(element, tuple)
          or (len(element) != 2)
          or not isinstance(element[0], str)
      ):
        raise TypeError(
            'Expected the locals to be a list of 2-element structs with string '
            'name as their first element, but this is not the case for the '
            'local at position {} in the sequence: {}.'.format(index, element)
        )
      name = element[0]
      value = element[1]
      py_typecheck.check_type(value, ComputationBuildingBlock)
      updated_locals.append((name, value))
    py_typecheck.check_type(result, ComputationBuildingBlock)
    super().__init__(result.type_signature)
    self._locals = updated_locals
    self._result = result

  def _proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        block=pb.Block(
            **{
                'local': [
                    pb.Block.Local(name=k, value=v.proto)
                    for k, v in self._locals
                ],
                'result': self._result.proto,
            }
        ),
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    for _, value in self._locals:
      yield value
    yield self._result

  @property
  def locals(self) -> list[tuple[str, ComputationBuildingBlock]]:
    return list(self._locals)

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Block):
      return NotImplemented
    return (self._locals, self._result) == (other._locals, other._result)

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((tuple(self._locals), self._result))
    return self._cached_hash

  def __repr__(self) -> str:
    return 'Block([{}], {!r})'.format(
        ', '.join("('{}', {!r})".format(k, v) for k, v in self._locals),
        self._result,
    )


class Intrinsic(ComputationBuildingBlock):
  """A representation of an intrinsic in TFF's internal language.

  An instrinsic is a symbol known to the TFF's compiler pipeline, represented
  as a known URI. It generally appears in expressions with a concrete type,
  although all intrinsic are defined with template types. This class does not
  deal with parsing intrinsic URIs and verifying their types, it is only a
  container. Parsing and type analysis are a responsibility of the components
  that manipulate ASTs. See intrinsic_defs.py for the list of known intrinsics.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Intrinsic':
    _check_computation_oneof(computation_proto, 'intrinsic')
    return cls(
        computation_proto.intrinsic.uri,
        type_serialization.deserialize_type(computation_proto.type),
    )

  def __init__(self, uri: str, type_signature: computation_types.Type):
    """Creates an intrinsic.

    Args:
      uri: The URI of the intrinsic.
      type_signature: A `tff.Type`, the type of the intrinsic.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(uri, str)
    py_typecheck.check_type(type_signature, computation_types.Type)
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(uri)
    if intrinsic_def is not None:
      # Note: this is really expensive.
      type_analysis.check_concrete_instance_of(
          type_signature, intrinsic_def.type_signature
      )
    super().__init__(type_signature)
    self._uri = uri

  def _proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        intrinsic=pb.Intrinsic(uri=self._uri),
    )

  def intrinsic_def(self) -> intrinsic_defs.IntrinsicDef:
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(self._uri)
    if intrinsic_def is None:
      raise ValueError(
          'Failed to retrieve definition of intrinsic with URI '
          f'`{self._uri}`. Perhaps a definition needs to be added to '
          '`intrinsic_defs.py`?'
      )
    return intrinsic_def

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def uri(self) -> str:
    return self._uri

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Intrinsic):
      return NotImplemented
    return (
        self._uri,
        self._type_signature,
    ) == (
        other._uri,
        other._type_signature,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((self._uri, self._type_signature))
    return self._cached_hash

  def __repr__(self) -> str:
    return "Intrinsic('{}', {!r})".format(self._uri, self.type_signature)


class Data(ComputationBuildingBlock):
  """A representation of data (an input pipeline).

  This class does not deal with parsing data URIs and verifying correctness,
  it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Data':
    _check_computation_oneof(computation_proto, 'data')
    return cls(
        computation_proto.data.content,
        type_serialization.deserialize_type(computation_proto.type),
    )

  def __init__(self, content: any_pb2.Any, type_spec: object):
    """Creates a representation of data.

    Args:
      content: The URI that characterizes the data.
      type_spec: Either the types.Type that represents the type of this data, or
        something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the user tries to specify an empty URI.
    """
    py_typecheck.check_type(content, any_pb2.Any)
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(content)
      )
    type_spec = computation_types.to_type(type_spec)
    super().__init__(type_spec)
    self._content = content

  def _proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        data=pb.Data(content=self._content),
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def content(self) -> any_pb2.Any:
    return self._content

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Data):
      return NotImplemented
    return (
        id(self._content),
        self._type_signature,
    ) == (
        id(other._content),
        other._type_signature,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((id(self._content), self._type_signature))
    return self._cached_hash

  def __repr__(self) -> str:
    return 'Data({}, {!r})'.format(id(self._content), self.type_signature)


class CompiledComputation(ComputationBuildingBlock):
  """A representation of a fully constructed and serialized computation.

  A compiled computation is one that has not been parsed into constituents, and
  is simply represented as an embedded `Computation` protocol buffer. Whereas
  technically, any computation can be represented and passed around this way,
  this structure is generally only used to represent TensorFlow sections, for
  which otherwise there isn't any dedicated structure.
  """

  def __init__(
      self,
      proto: pb.Computation,
      name: Optional[str] = None,
      type_signature: Optional[computation_types.Type] = None,
  ):
    """Creates a representation of a fully constructed computation.

    Args:
      proto: An instance of pb.Computation with the computation logic.
      name: An optional string name to associate with this computation, used
        only for debugging purposes. If the name is not specified (None), it is
        autogenerated as a hexadecimal string from the hash of the proto.
      type_signature: An optional type signature to associate with this
        computation rather than the serialized one.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(proto, pb.Computation)
    if name is not None:
      py_typecheck.check_type(name, str)
    if type_signature is None:
      type_signature = type_serialization.deserialize_type(proto.type)
    py_typecheck.check_type(type_signature, computation_types.Type)
    super().__init__(type_signature)
    self._proto_representation = proto
    if name is not None:
      self._name = name
    else:
      self._name = '{:x}'.format(
          zlib.adler32(self._proto_representation.SerializeToString())
      )

  def _proto(self) -> pb.Computation:
    return self._proto_representation

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def name(self) -> str:
    return self._name

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, CompiledComputation):
      return NotImplemented
    return (
        self._proto_representation,
        self._name,
        self._type_signature,
    ) == (
        other._proto_representation,
        other._name,
        other._type_signature,
    )

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((
          self._proto_representation.SerializeToString(),
          self._name,
          self._type_signature,
      ))
    return self._cached_hash

  def __repr__(self) -> str:
    return "CompiledComputation('{}', {!r})".format(
        self._name, self.type_signature
    )


class Placement(ComputationBuildingBlock):
  """A representation of a placement literal in TFF's internal language.

  Currently this can only be `tff.SERVER` or `tff.CLIENTS`.
  """

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Placement':
    _check_computation_oneof(computation_proto, 'placement')
    return cls(
        placements.uri_to_placement_literal(
            str(computation_proto.placement.uri)
        )
    )

  def __init__(self, literal: placements.PlacementLiteral):
    """Constructs a new placement instance for the given placement literal.

    Args:
      literal: The placement literal.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(literal, placements.PlacementLiteral)
    super().__init__(computation_types.PlacementType())
    self._literal = literal

  def _proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        placement=pb.Placement(uri=self._literal.uri),
    )

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def uri(self) -> str:
    return self._literal.uri

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Placement):
      return NotImplemented
    return self._literal == other._literal

  def __hash__(self):
    if self._cached_hash is None:
      self._cached_hash = hash((self._literal))
    return self._cached_hash

  def __repr__(self) -> str:
    return "Placement('{}')".format(self.uri)


class Literal(ComputationBuildingBlock):
  """A representation of a literal in TFF's internal language."""

  def __init__(
      self, value: array.Array, type_signature: computation_types.TensorType
  ):
    """Returns an initialized `tff.framework.Literal`.

    Args:
      value: The value of the literal.
      type_signature: A `tff.TensorType`.

    Raises:
      ValueError: If `value` is not compatible with `type_signature`.
    """
    if (
        isinstance(value, (np.ndarray, np.generic))
        and value.dtype.type is np.str_
    ):
      value = value.astype(np.bytes_)
    elif isinstance(value, str):
      value = value.encode()

    if not array.is_compatible_dtype(value, type_signature.dtype.type):
      raise ValueError(
          f"Expected '{value}' to be compatible with"
          f" '{type_signature.dtype.type}'."
      )

    if not array.is_compatible_shape(value, type_signature.shape):
      raise ValueError(
          f"Expected '{value}' to be compatible with '{type_signature.shape}'."
      )

    super().__init__(type_signature)
    self._value = value
    self._type_signature = type_signature
    self._cached_hash = None

  @property
  def type_signature(self) -> computation_types.TensorType:
    return self._type_signature

  @classmethod
  def from_proto(cls, computation_proto: pb.Computation) -> 'Literal':
    _check_computation_oneof(computation_proto, 'literal')
    value = array.from_proto(computation_proto.literal.value)
    type_signature = type_serialization.deserialize_type(computation_proto.type)
    if not isinstance(type_signature, computation_types.TensorType):
      raise ValueError(
          'Expected `type_signature` to be a `tff.TensorType`, found'
          f' {type(type_signature)}.'
      )
    return cls(value, type_signature)

  def _proto(self) -> pb.Computation:
    type_pb = type_serialization.serialize_type(self.type_signature)
    value_pb = array.to_proto(
        self._value, dtype_hint=self.type_signature.dtype.type
    )
    literal_pb = pb.Literal(value=value_pb)
    return pb.Computation(type=type_pb, literal=literal_pb)

  def children(self) -> Iterator[ComputationBuildingBlock]:
    return iter(())

  @property
  def value(self) -> object:
    return self._value

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Literal):
      return NotImplemented

    if self._type_signature != other._type_signature:
      return False
    if isinstance(self._value, np.ndarray) and isinstance(
        other._value, np.ndarray
    ):
      return np.array_equal(self._value, other._value)
    else:
      return self._value == other._value

  def __hash__(self):
    if self._cached_hash is None:
      if isinstance(self._value, (np.ndarray, np.generic)):
        hashable_value = tuple(self._value.flatten().tolist())
      else:
        hashable_value = self._value
      self._cached_hash = hash((hashable_value, self._type_signature))
    return self._cached_hash

  def __repr__(self) -> str:
    if isinstance(self._value, np.ndarray):
      value_repr = (
          f'np.array({self._value.tolist()},'
          f' dtype=np.{self._value.dtype.type.__name__})'
      )
    else:
      value_repr = repr(self._value)
    return f'Literal({value_repr}, {self._type_signature!r})'


def _string_representation(
    comp: ComputationBuildingBlock,
    formatted: bool,
) -> str:
  """Returns the string representation of a `ComputationBuildingBlock`.

  This functions creates a `list` of strings representing the given `comp`;
  combines the strings in either a formatted or un-formatted representation; and
  returns the resulting string represetnation.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.
    formatted: A boolean indicating if the returned string should be formatted.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """
  py_typecheck.check_type(comp, ComputationBuildingBlock)

  def _join(components: Iterable[list[str]]) -> list[str]:
    """Returns a `list` of strings by combining each component in `components`.

    >>> _join([['a'], ['b'], ['c']])
    ['abc']

    >>> _join([['a', 'b', 'c'], ['d', 'e', 'f']])
    ['abcd', 'ef']

    This function is used to help track where new-lines should be inserted into
    the string representation if the lines are formatted.

    Args:
      components: A `list` where each element is a `list` of strings
        representing a part of the string of a `ComputationBuildingBlock`.
    """
    lines = ['']
    for component in components:
      lines[-1] = '{}{}'.format(lines[-1], component[0])
      lines.extend(component[1:])
    return lines

  def _indent(lines, indent_chars='  '):
    """Returns a `list` of strings indented across a slice."""
    return ['{}{}'.format(indent_chars, e) for e in lines]

  def _lines_for_named_comps(named_comps, formatted):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named computations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    lines = []
    for index, (name, comp) in enumerate(named_comps):
      if index != 0:
        if formatted:
          lines.append([',', ''])
        else:
          lines.append([','])
      element_lines = _lines_for_comp(comp, formatted)
      if name is not None:
        element_lines = _join([
            ['{}='.format(name)],
            element_lines,
        ])
      lines.append(element_lines)
    return _join(lines)

  def _lines_for_comp(comp, formatted):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    if isinstance(comp, Block):
      lines = []
      variables_lines = _lines_for_named_comps(comp.locals, formatted)
      if formatted:
        variables_lines = _indent(variables_lines)
        lines.extend([['(let ', ''], variables_lines, ['', ' in ']])
      else:
        lines.extend([['(let '], variables_lines, [' in ']])
      result_lines = _lines_for_comp(comp.result, formatted)
      lines.append(result_lines)
      lines.append([')'])
      return _join(lines)
    elif isinstance(comp, Reference):
      if comp.context is not None:
        return ['{}@{}'.format(comp.name, comp.context)]
      else:
        return [comp.name]
    elif isinstance(comp, Selection):
      source_lines = _lines_for_comp(comp.source, formatted)
      if comp.name is not None:
        return _join([source_lines, ['.{}'.format(comp.name)]])
      else:
        return _join([source_lines, ['[{}]'.format(comp.index)]])
    elif isinstance(comp, Call):
      function_lines = _lines_for_comp(comp.function, formatted)
      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument, formatted)
        return _join([function_lines, ['('], argument_lines, [')']])
      else:
        return _join([function_lines, ['()']])
    elif isinstance(comp, CompiledComputation):
      return ['comp#{}'.format(comp.name)]
    elif isinstance(comp, Data):
      return [str(id(comp.content))]
    elif isinstance(comp, Intrinsic):
      return [comp.uri]
    elif isinstance(comp, Lambda):
      result_lines = _lines_for_comp(comp.result, formatted)
      if comp.parameter_type is None:
        param_name = ''
      else:
        param_name = comp.parameter_name
      lines = [['({} -> '.format(param_name)], result_lines, [')']]
      return _join(lines)
    elif isinstance(comp, Placement):
      placement_literal = placements.uri_to_placement_literal(comp.uri)
      return [placement_literal.name]
    elif isinstance(comp, Literal):
      return [str(comp.value)]
    elif isinstance(comp, Struct):
      if not comp:
        return ['<>']
      elements = structure.to_elements(comp)
      elements_lines = _lines_for_named_comps(elements, formatted)
      if formatted:
        elements_lines = _indent(elements_lines)
        lines = [['<', ''], elements_lines, ['', '>']]
      else:
        lines = [['<'], elements_lines, ['>']]
      return _join(lines)
    else:
      raise NotImplementedError('Unexpected type found: {}.'.format(type(comp)))

  lines = _lines_for_comp(comp, formatted)
  lines = [line.rstrip() for line in lines]
  if formatted:
    return '\n'.join(lines)
  else:
    return ''.join(lines)


def _structural_representation(comp):
  """Returns the structural string representation of the given `comp`.

  This functions creates and returns a string representing the structure of the
  abstract syntax tree for the given `comp`.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """
  py_typecheck.check_type(comp, ComputationBuildingBlock)
  padding_char = ' '

  def _get_leading_padding(string):
    """Returns the length of the leading padding for the given `string`."""
    for index, character in enumerate(string):
      if character != padding_char:
        return index
    return len(string)

  def _get_trailing_padding(string):
    """Returns the length of the trailing padding for the given `string`."""
    for index, character in enumerate(reversed(string)):
      if character != padding_char:
        return index
    return len(string)

  def _pad_left(lines, total_width):
    """Pads the beginning of each line in `lines` to the given `total_width`.

    >>>_pad_left(['aa', 'bb'], 4)
    ['  aa', '  bb',]

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_left(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(padding_char * padding, line)

    return [_pad_line_left(line, total_width) for line in lines]

  def _pad_right(lines, total_width):
    """Pads the end of each line in `lines` to the given `total_width`.

    >>>_pad_right(['aa', 'bb'], 4)
    ['aa  ', 'bb  ']

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_right(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(line, padding_char * padding)

    return [_pad_line_right(line, total_width) for line in lines]

  class Alignment(enum.Enum):
    LEFT = 1
    RIGHT = 2

  def _concatenate(lines_1, lines_2, align):
    """Concatenates two `list`s of strings.

    Concatenates two `list`s of strings by appending one list of strings to the
    other and then aligning lines of different widths by either padding the left
    or padding the right of each line to the width of the longest line.

    >>>_concatenate(['aa', 'bb'], ['ccc'], Alignment.LEFT)
    ['aa ', 'bb ', 'ccc']

    Args:
      lines_1: A `list` of strings.
      lines_2: A `list` of strings.
      align: An enum indicating how to align lines of different widths.

    Returns:
      A `list` of lines.
    """
    lines = lines_1 + lines_2
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    if align is Alignment.LEFT:
      return _pad_right(lines, longest_width)
    elif align is Alignment.RIGHT:
      return _pad_left(lines, longest_width)

  def _calculate_inset_from_padding(
      left, right, preferred_padding, minimum_content_padding
  ):
    """Calculates the inset for the given padding.

    Note: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      An integer.
    """
    assert preferred_padding >= 0
    assert minimum_content_padding >= 0

    trailing_padding = _get_trailing_padding(left[0])
    leading_padding = _get_leading_padding(right[0])
    inset = trailing_padding + leading_padding - preferred_padding
    for left_line, right_line in zip(left[1:], right[1:]):
      trailing_padding = _get_trailing_padding(left_line)
      leading_padding = _get_leading_padding(right_line)
      minimum_inset = (
          trailing_padding + leading_padding - minimum_content_padding
      )
      inset = min(inset, minimum_inset)
    return inset

  def _fit_with_inset(left, right, inset):
    r"""Concatenates the lines of two `list`s of strings.

    Note: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      inset: The amount of padding to remove or add when concatenating the
        lines.

    Returns:
      A `list` of lines.
    """
    lines = []
    for left_line, right_line in zip(left, right):
      if inset > 0:
        left_inset = 0
        trailing_padding = _get_trailing_padding(left_line)
        if trailing_padding > 0:
          left_inset = min(trailing_padding, inset)
          left_line = left_line[:-left_inset]
        if inset - left_inset > 0:
          leading_padding = _get_leading_padding(right_line)
          if leading_padding > 0:
            right_inset = min(leading_padding, inset - left_inset)
            right_line = right_line[right_inset:]
      padding = abs(inset) if inset < 0 else 0
      line = ''.join([left_line, padding_char * padding, right_line])
      lines.append(line)
    left_height = len(left)
    right_height = len(right)
    if left_height > right_height:
      lines.extend(left[right_height:])
    elif right_height > left_height:
      lines.extend(right[left_height:])
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    shortest_line = min(lines, key=len)
    shortest_width = len(shortest_line)
    if shortest_width != longest_width:
      if left_height > right_height:
        lines = _pad_right(lines, longest_width)
      else:
        lines = _pad_left(lines, longest_width)
    return lines

  def _fit_with_padding(
      left, right, preferred_padding, minimum_content_padding=4
  ):
    r"""Concatenates the lines of two `list`s of strings.

    Concatenates the lines of two `list`s of strings by appending each line
    together using a padding. The same padding is used to append each line and
    the padding is calculated starting from the `preferred_padding` without
    going below `minimum_content_padding` on any of the lines. If the two
    `list`s of strings have different lengths, padding will be applied to
    maintain the length of each string in the resulting `list` of strings.

    >>>_fit_with_padding(['aa', 'bb'], ['ccc'])
    ['aa    cccc', 'bb        ']

    >>>_fit_with_padding(['aa          ', 'bb          '], ['          ccc'])
    ['aa    cccc', 'bb        ']

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      A `list` of lines.
    """
    inset = _calculate_inset_from_padding(
        left, right, preferred_padding, minimum_content_padding
    )
    return _fit_with_inset(left, right, inset)

  def _get_node_label(comp):
    """Returns a string for node in the structure of the given `comp`."""
    if isinstance(comp, Block):
      return 'Block'
    elif isinstance(comp, Call):
      return 'Call'
    elif isinstance(comp, CompiledComputation):
      return 'Compiled({})'.format(comp.name)
    elif isinstance(comp, Data):
      return f'Data({id(comp.content)})'
    elif isinstance(comp, Intrinsic):
      return comp.uri
    elif isinstance(comp, Lambda):
      return 'Lambda({})'.format(comp.parameter_name)
    elif isinstance(comp, Reference):
      return 'Ref({})'.format(comp.name)
    elif isinstance(comp, Placement):
      return 'Placement'
    elif isinstance(comp, Selection):
      key = comp.name if comp.name is not None else comp.index
      return 'Sel({})'.format(key)
    elif isinstance(comp, Struct):
      return 'Struct'
    elif isinstance(comp, Literal):
      return f'Lit({comp.value})'
    else:
      raise TypeError('Unexpected type found: {}.'.format(type(comp)))

  def _lines_for_named_comps(named_comps):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named computations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
    """
    lines = ['[']
    for index, (name, comp) in enumerate(named_comps):
      comp_lines = _lines_for_comp(comp)
      if name is not None:
        label = '{}='.format(name)
        comp_lines = _fit_with_padding([label], comp_lines, 0, 0)
      if index == 0:
        lines = _fit_with_padding(lines, comp_lines, 0, 0)
      else:
        lines = _fit_with_padding(lines, [','], 0, 0)
        lines = _fit_with_padding(lines, comp_lines, 1)
    lines = _fit_with_padding(lines, [']'], 0, 0)
    return lines

  def _lines_for_comp(comp):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
    """
    node_label = _get_node_label(comp)

    if isinstance(
        comp,
        (
            CompiledComputation,
            Data,
            Intrinsic,
            Placement,
            Reference,
            Literal,
        ),
    ):
      return [node_label]
    elif isinstance(comp, Block):
      variables_lines = _lines_for_named_comps(comp.locals)
      variables_width = len(variables_lines[0])
      variables_trailing_padding = _get_trailing_padding(variables_lines[0])
      leading_padding = variables_width - variables_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      variables_lines = _concatenate(
          [edge_line], variables_lines, Alignment.LEFT
      )

      result_lines = _lines_for_comp(comp.result)
      result_width = len(result_lines[0])
      leading_padding = _get_leading_padding(result_lines[0]) - 1
      trailing_padding = result_width - leading_padding - 1
      edge_line = '\\{}'.format(padding_char * trailing_padding)
      result_lines = _concatenate([edge_line], result_lines, Alignment.RIGHT)

      preferred_padding = len(node_label)
      lines = _fit_with_padding(
          variables_lines, result_lines, preferred_padding
      )
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif isinstance(comp, Call):
      function_lines = _lines_for_comp(comp.function)
      function_width = len(function_lines[0])
      function_trailing_padding = _get_trailing_padding(function_lines[0])
      leading_padding = function_width - function_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      function_lines = _concatenate([edge_line], function_lines, Alignment.LEFT)

      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument)
        argument_width = len(argument_lines[0])
        leading_padding = _get_leading_padding(argument_lines[0]) - 1
        trailing_padding = argument_width - leading_padding - 1
        edge_line = '\\{}'.format(padding_char * trailing_padding)
        argument_lines = _concatenate(
            [edge_line], argument_lines, Alignment.RIGHT
        )

        preferred_padding = len(node_label)
        lines = _fit_with_padding(
            function_lines, argument_lines, preferred_padding
        )
      else:
        lines = function_lines
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif isinstance(comp, Lambda):
      result_lines = _lines_for_comp(comp.result)
      leading_padding = _get_leading_padding(result_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], result_lines, Alignment.LEFT)
    elif isinstance(comp, Selection):
      source_lines = _lines_for_comp(comp.source)
      leading_padding = _get_leading_padding(source_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], source_lines, Alignment.LEFT)
    elif isinstance(comp, Struct):
      elements = structure.to_elements(comp)
      elements_lines = _lines_for_named_comps(elements)
      leading_padding = _get_leading_padding(elements_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate(
          [node_line, edge_line], elements_lines, Alignment.LEFT
      )
    else:
      raise NotImplementedError('Unexpected type found: {}.'.format(type(comp)))

  lines = _lines_for_comp(comp)
  lines = [line.rstrip() for line in lines]
  return '\n'.join(lines)


_deserializer_dict = {
    'reference': Reference.from_proto,
    'selection': Selection.from_proto,
    'struct': Struct.from_proto,
    'call': Call.from_proto,
    'lambda': Lambda.from_proto,
    'block': Block.from_proto,
    'intrinsic': Intrinsic.from_proto,
    'data': Data.from_proto,
    'placement': Placement.from_proto,
    'literal': Literal.from_proto,
    'tensorflow': CompiledComputation,
    'xla': CompiledComputation,
}
