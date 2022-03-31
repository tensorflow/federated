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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Representation of values inside a federated computation."""

import abc
import collections
import itertools
from typing import Any, Optional, Union

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import named_containers
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import symbol_binding_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _unfederated(type_signature):
  if type_signature.is_federated():
    return type_signature.member
  return type_signature


def _is_federated_named_tuple(vimpl: 'Value') -> bool:
  comp_ty = vimpl.type_signature
  return comp_ty.is_federated() and comp_ty.member.is_struct()


def _is_named_tuple(vimpl: 'Value') -> bool:
  return vimpl.type_signature.is_struct()  # pylint: disable=protected-access


def _check_struct_or_federated_struct(
    vimpl: 'Value',
    attribute: str,
):
  if not (_is_named_tuple(vimpl) or _is_federated_named_tuple(vimpl)):
    raise AttributeError(
        f'`tff.Value` of non-structural type {vimpl.type_signature} has no '
        f'attribute {attribute}')


def _bind_computation_to_reference(comp, op: str):
  context = context_stack_impl.context_stack.current
  if not isinstance(context, symbol_binding_context.SymbolBindingContext):
    raise context_base.ContextError(
        '`tff.Value`s should only be used in contexts which can bind '
        'references, generally a `FederatedComputationContext`. Attempted '
        f'to bind the result of {op} in a context {context} of '
        f'type {type(context)}.')
  return context.bind_computation_to_reference(comp)


class Value(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """A generic base class for values that appear in TFF computations.

  If the value in this class is of `StructType` or `FederatedType` containing a
  `StructType`, the inner fields can be accessed by name
  (e.g. `y = my_value_impl.y`).
  """

  def __init__(
      self,
      comp: building_blocks.ComputationBuildingBlock,
  ):
    """Constructs a value of the given type.

    Args:
      comp: An instance of building_blocks.ComputationBuildingBlock that
        contains the logic that computes this value.
    """
    super()
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    self._comp = comp

  @property
  def type_signature(self):
    return self._comp.type_signature

  @property
  def comp(self) -> building_blocks.ComputationBuildingBlock:
    return self._comp

  def __repr__(self):
    return repr(self._comp)

  def __str__(self):
    return str(self._comp)

  def __dir__(self):
    attributes = ['type_signature', 'comp']
    type_signature = _unfederated(self.type_signature)
    if type_signature.is_struct():
      attributes.extend(dir(type_signature))
    return attributes

  def __getattr__(self, name):
    py_typecheck.check_type(name, str)
    _check_struct_or_federated_struct(self, name)
    if _is_federated_named_tuple(self):
      if name not in structure.name_list(self.type_signature.member):
        raise AttributeError(
            'There is no such attribute \'{}\' in this federated tuple. Valid '
            'attributes: ({})'.format(
                name, ', '.join(dir(self.type_signature.member))))

      return Value(
          building_block_factory.create_federated_getattr_call(
              self._comp, name))
    if name not in dir(self.type_signature):
      raise AttributeError(
          'There is no such attribute \'{}\' in this tuple. Valid attributes: ({})'
          .format(name, ', '.join(dir(self.type_signature))))
    if self._comp.is_struct():
      return Value(getattr(self._comp, name))
    return Value(building_blocks.Selection(self._comp, name=name))

  def __bool__(self):
    raise TypeError(
        'Federated computation values do not support boolean operations. '
        'If you were attempting to perform logic on tensors, consider moving '
        'this logic into a tff.tf_computation.')

  def __len__(self):
    type_signature = _unfederated(self.type_signature)
    if not type_signature.is_struct():
      raise TypeError(
          'Operator len() is only supported for (possibly federated) structure '
          'types, but the object on which it has been invoked is of type {}.'
          .format(self.type_signature))
    return len(type_signature)

  def __getitem__(self, key: Union[int, str, slice]):
    py_typecheck.check_type(key, (int, str, slice))
    if isinstance(key, str):
      return getattr(self, key)
    if _is_federated_named_tuple(self):
      return Value(
          building_block_factory.create_federated_getitem_call(self._comp,
                                                               key),)
    if not _is_named_tuple(self):
      raise TypeError(
          'Operator getitem() is only supported for structure types, but the '
          'object on which it has been invoked is of type {}.'.format(
              self.type_signature))
    elem_length = len(self.type_signature)
    if isinstance(key, int):
      if key < 0 or key >= elem_length:
        raise IndexError(
            'The index of the selected element {} is out of range.'.format(key))
      if self._comp.is_struct():
        return Value(self._comp[key])
      else:
        return Value(building_blocks.Selection(self._comp, index=key))
    elif isinstance(key, slice):
      index_range = range(*key.indices(elem_length))
      if not index_range:
        raise IndexError('Attempted to slice 0 elements, which is not '
                         'currently supported.')
      return to_value([self[k] for k in index_range], None)

  def __iter__(self):
    type_signature = _unfederated(self.type_signature)
    if not type_signature.is_struct():
      raise TypeError(
          'Operator iter() is only supported for (possibly federated) structure '
          'types, but the object on which it has been invoked is of type {}.'
          .format(self.type_signature))
    for index in range(len(type_signature)):
      yield self[index]

  def __call__(self, *args, **kwargs):
    if not self.type_signature.is_function():
      raise SyntaxError(
          'Function-like invocation is only supported for values of functional '
          'types, but the value being invoked is of type {} that does not '
          'support invocation.'.format(self.type_signature))
    if args or kwargs:
      args = [to_value(x, None) for x in args]
      kwargs = {k: to_value(v, None) for k, v in kwargs.items()}
      arg = function_utils.pack_args(self.type_signature.parameter, args,
                                     kwargs)
      arg = to_value(arg, None).comp
    else:
      arg = None
    call = building_blocks.Call(self._comp, arg)
    ref = _bind_computation_to_reference(call, 'calling a `tff.Value`')
    return structural_ref_to_py_container(ref)

  def __add__(self, other):
    other = to_value(other, None)
    if not self.type_signature.is_equivalent_to(other.type_signature):
      raise TypeError('Cannot add {} and {}.'.format(self.type_signature,
                                                     other.type_signature))
    call = building_blocks.Call(
        building_blocks.Intrinsic(
            intrinsic_defs.GENERIC_PLUS.uri,
            computation_types.FunctionType(
                [self.type_signature, self.type_signature],
                self.type_signature)),
        to_value([self, other], None).comp)
    ref = _bind_computation_to_reference(call, 'adding a tff.Value')
    return Value(ref)


def _structural_block_to_struct(
    block: building_blocks.ComputationBuildingBlock
) -> Union[Value, structure.Struct]:
  """Converts `block`s of structural type into `Struct` trees of `Value`s."""
  if not block.type_signature.is_struct():
    return Value(block)
  names = structure.name_list_with_nones(block.type_signature)
  elements = []
  for i, name in enumerate(names):
    if name is None:
      selection = building_blocks.Selection(block, index=i)
    else:
      selection = building_blocks.Selection(block, name=name)
    element = _structural_block_to_struct(selection)
    elements.append((name, element))
  return structure.Struct(elements)


def structural_ref_to_struct(
    ref: building_blocks.Reference) -> Union[Value, structure.Struct]:
  """Converts `ref`s of structural type into `Struct` trees of `Value`s."""
  ref.check_reference()
  return _structural_block_to_struct(ref)


def structural_ref_to_py_container(ref: building_blocks.Reference) -> Any:
  """Converts `ref`s of structural type into Python structures of `Value`s."""
  ref.check_reference()
  struct_ = _structural_block_to_struct(ref)
  return type_conversions.type_to_py_container(
      struct_, ref.type_signature, struct_only=True)


def _wrap_computation_as_value(proto: pb.Computation) -> Value:
  """Wraps the given computation as a `tff.Value`."""
  py_typecheck.check_type(proto, pb.Computation)
  compiled = building_blocks.CompiledComputation(proto)
  call = building_blocks.Call(compiled)
  ref = _bind_computation_to_reference(call,
                                       'wrapping a computation as a value')
  return Value(ref)


def _wrap_constant_as_value(const) -> Value:
  """Wraps the given Python constant as a `tff.Value`.

  Args:
    const: Python constant convertible to Tensor via `tf.constant`.

  Returns:
    An instance of `tff.Value`.
  """
  tf_comp, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      fn=lambda: tf.constant(const), parameter_type=None)
  return _wrap_computation_as_value(tf_comp)


def _wrap_sequence_as_value(elements, element_type) -> Value:
  """Wraps `elements` as a TFF sequence with elements of type `element_type`.

  Args:
    elements: Python object to the wrapped as a TFF sequence value.
    element_type: An instance of `Type` that determines the type of elements of
      the sequence.

  Returns:
    An instance of `tff.Value`.

  Raises:
    TypeError: If `elements` and `element_type` are of incompatible types.
  """
  # TODO(b/113116813): Add support for other representations of sequences.
  py_typecheck.check_type(elements, list)
  for element in elements:
    inferred_type = type_conversions.infer_type(element)
    if not element_type.is_assignable_from(inferred_type):
      raise TypeError(
          'Expected all sequence elements to be {}, found {}.'.format(
              element_type, inferred_type))

  def _create_dataset_from_elements():
    return tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), elements, element_type)

  proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      fn=_create_dataset_from_elements, parameter_type=None)
  return _wrap_computation_as_value(proto)


def _dictlike_items_to_value(items, type_spec, container_type) -> Value:
  elements = []
  for (i, (k, v)) in enumerate(items):
    element_type = None if type_spec is None else type_spec[i]
    element_value = to_value(v, element_type)
    elements.append((k, element_value.comp))
  return Value(building_blocks.Struct(elements, container_type))


def to_value(
    arg: Any,
    type_spec: Optional[computation_types.Type],
    *,
    parameter_type_hint=None,
    zip_if_needed: bool = False,
) -> Value:
  """Converts the argument into an instance of the abstract class `tff.Value`.

  Instances of `tff.Value` represent TFF values that appear internally in
  federated computations. This helper function can be used to wrap a variety of
  Python objects as `tff.Value` instances to allow them to be passed as
  arguments, used as functions, or otherwise manipulated within bodies of
  federated computations.

  At the moment, the supported types include:

  * Simple constants of `str`, `int`, `float`, and `bool` types, mapped to
    values of a TFF tensor type.

  * Numpy arrays (`np.ndarray` objects), also mapped to TFF tensors.

  * Dictionaries (`collections.OrderedDict` and unordered `dict`), `list`s,
    `tuple`s, `namedtuple`s, and `Struct`s, all of which are mapped to
    TFF tuple type.

  * Computations (constructed with either the `tff.tf_computation` or with the
    `tff.federated_computation` decorator), typically mapped to TFF functions.

  * Placement literals (`tff.CLIENTS`, `tff.SERVER`), mapped to values of the
    TFF placement type.

  This function is also invoked when attempting to execute a TFF computation.
  All arguments supplied in the invocation are converted into TFF values prior
  to execution. The types of Python objects that can be passed as arguments to
  computations thus matches the types listed here.

  Args:
    arg: An instance of one of the Python types that are convertible to TFF
      values (instances of `tff.Value`).
    type_spec: An optional type specifier that allows for disambiguating the
      target type (e.g., when two TFF types can be mapped to the same Python
      representations). If not specified, TFF tried to determine the type of the
      TFF value automatically.
    parameter_type_hint: An optional `tff.Type` or value convertible to it by
      `tff.to_type()` which specifies an argument type to use in the case that
      `arg` is a `function_utils.PolymorphicComputation`.
    zip_if_needed: If `True`, attempt to coerce the result of `to_value` to
      match `type_spec` by applying `intrinsics.federated_zip` to appropriate
      elements.

  Returns:
    An instance of `tff.Value` as described above.

  Raises:
    TypeError: if `arg` is of an unsupported type, or of a type that does not
      match `type_spec`. Raises explicit error message if TensorFlow constructs
      are encountered, as TensorFlow code should be sealed away from TFF
      federated context.
  """
  if type_spec is not None:
    type_spec = computation_types.to_type(type_spec)
  if isinstance(arg, Value):
    result = arg
  elif isinstance(arg, building_blocks.ComputationBuildingBlock):
    result = Value(arg)
  elif isinstance(arg, placements.PlacementLiteral):
    result = Value(building_blocks.Placement(arg))
  elif isinstance(arg, (computation_impl.ConcreteComputation,
                        function_utils.PolymorphicComputation)):
    if isinstance(arg, function_utils.PolymorphicComputation):
      if parameter_type_hint is None:
        raise TypeError(
            'Polymorphic computations cannot be converted to `tff.Value`s '
            'without a type hint. Consider explicitly specifying the '
            'argument types of a computation before passing it to a '
            'function that requires a `tff.Value` (such as a TFF intrinsic '
            'like `federated_map`). If you are a TFF developer and think '
            'this should be supported, consider providing `parameter_type_hint` '
            'as an argument to the encompassing `to_value` conversion.')
      parameter_type_hint = computation_types.to_type(parameter_type_hint)
      arg = arg.fn_for_argument_type(parameter_type_hint)
    py_typecheck.check_type(arg, computation_impl.ConcreteComputation)
    result = Value(arg.to_compiled_building_block())
  elif type_spec is not None and type_spec.is_sequence():
    result = _wrap_sequence_as_value(arg, type_spec.element)
  elif isinstance(arg, structure.Struct):
    items = structure.iter_elements(arg)
    result = _dictlike_items_to_value(items, type_spec, None)
  elif py_typecheck.is_named_tuple(arg):
    items = arg._asdict().items()
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif py_typecheck.is_attrs(arg):
    items = named_containers.attrs_class_to_odict(arg).items()
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif py_typecheck.is_dataclass(arg):
    items = named_containers.dataclass_to_odict(arg).items()
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif isinstance(arg, dict):
    if not isinstance(arg, collections.OrderedDict):
      raise TypeError(
          'Unsupported mapping type {}. Use collections.OrderedDict for '
          'mappings. Unsupported mapping: {}'.format(
              py_typecheck.type_string(type(arg)), arg))
    result = _dictlike_items_to_value(arg.items(), type_spec, type(arg))
  elif isinstance(arg, (tuple, list)):
    items = zip(itertools.repeat(None), arg)
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif isinstance(arg, tensorflow_utils.TENSOR_REPRESENTATION_TYPES):
    result = _wrap_constant_as_value(arg)
  elif isinstance(arg, (tf.Tensor, tf.Variable)):
    raise TypeError(
        'TensorFlow construct {} has been encountered in a federated '
        'context. TFF does not support mixing TF and federated orchestration '
        'code. Please wrap any TensorFlow constructs with '
        '`tff.tf_computation`.'.format(arg))
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a `tff.Value`.'.format(
            py_typecheck.type_string(type(arg))))
  py_typecheck.check_type(result, Value)
  if (type_spec is not None and
      not type_spec.is_assignable_from(result.type_signature)):
    if zip_if_needed:
      # Returns `None` if such a zip can't be performed.
      zipped_comp = building_block_factory.zip_to_match_type(
          comp_to_zip=result.comp, target_type=type_spec)
      if zipped_comp is not None:
        return Value(zipped_comp)
    raise TypeError(
        'The supplied argument maps to TFF type {}, which is incompatible with '
        'the requested type {}.'.format(result.type_signature, type_spec))
  return result
