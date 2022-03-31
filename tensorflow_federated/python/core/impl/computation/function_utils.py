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
"""Utilities for Python functions, defuns, and other types of callables."""

import functools
import inspect
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object
from tensorflow_federated.python.tensorflow_libs import function


def get_signature(
    fn: Union[types.FunctionType, types.MethodType]) -> inspect.Signature:
  """Returns the `inspect.Signature` structure for the given function or method.

  Args:
    fn: The Python function or Tensorflow function to analyze.

  Returns:
    An `inspect.Signature`.

  Raises:
    TypeError: if the argument is not of a supported type.
  """
  if isinstance(fn, (types.FunctionType, types.MethodType)):
    return inspect.signature(fn)
  elif function.is_tf_function(fn):
    return inspect.signature(fn.python_function)
  else:
    raise TypeError('Expected a Python function or a defun, found {}.'.format(
        py_typecheck.type_string(type(fn))))


def is_signature_compatible_with_types(signature: inspect.Signature, *args,
                                       **kwargs) -> bool:
  """Determines if functions matching signature accept `args` and `kwargs`.

  Args:
    signature: An instance of `inspect.Signature` to verify agains the
      arguments.
    *args: Zero or more positional arguments, all of which must be instances of
      computation_types.Type or something convertible to it by
      computation_types.to_type().
    **kwargs: Zero or more keyword arguments, all of which must be instances of
      computation_types.Type or something convertible to it by
      computation_types.to_type().

  Returns:
    `True` or `False`, depending on the outcome of the test.

  Raises:
    TypeError: if the arguments are of the wrong computation_types.
  """
  try:
    bound_args = signature.bind(*args, **kwargs)
  except TypeError:
    return False

  # If we have no defaults then `bind` will have raised `TypeError` if the
  # signature was not compatible with *args and **kwargs.
  if all(p.default is inspect.Parameter.empty
         for p in signature.parameters.values()):
    return True

  # Otherwise we need to check the defaults against the types that were given to
  # ensure they are compatible.
  for p in signature.parameters.values():
    if p.default is inspect.Parameter.empty or p.default is None:
      # No default value or optional.
      continue
    arg_value = bound_args.arguments.get(p.name, p.default)
    if arg_value is p.default:
      continue
    arg_type = computation_types.to_type(arg_value)
    default_type = type_conversions.infer_type(p.default)
    if not arg_type.is_assignable_from(default_type):
      return False
  return True


def is_argument_struct(arg) -> bool:
  """Determines if 'arg' is interpretable as an argument struct.

  Args:
    arg: A value or type to test.

  Returns:
    True iff 'arg' is either a `Struct` in which all unnamed elements
    precede named ones, or a `StructType` with this property, or something
    that can be converted into the latter by computation_types.to_type().

  Raises:
    TypeError: If the argument is neither an `structure.Struct`,
      nor a type spec.
  """
  if isinstance(arg, structure.Struct):
    elements = structure.to_elements(arg)
  elif isinstance(arg, typed_object.TypedObject):
    return is_argument_struct(arg.type_signature)
  else:
    arg = computation_types.to_type(arg)
    if arg.is_struct():
      elements = structure.to_elements(arg)
    else:
      return False
  max_unnamed = -1
  min_named = len(elements)
  for idx, element in enumerate(elements):
    if element[0]:
      min_named = min(min_named, idx)
    else:
      max_unnamed = idx
  return max_unnamed < min_named


def unpack_args_from_struct(
    struct_with_args) -> Tuple[List[Any], Dict[str, Any]]:
  """Extracts argument types from a struct.

  Args:
    struct_with_args: An instance of either an `struct.Struct` or
      computation_types.StructType` (or something convertible to it by
      computation_types.to_type()), on which is_argument_struct() is True.

  Returns:
    A pair (args, kwargs) containing tuple elements from 'struct_with_args'.

  Raises:
    TypeError: if 'struct_with_args' is of a wrong type.
  """
  if not is_argument_struct(struct_with_args):
    raise TypeError('Not an argument struct: {}.'.format(struct_with_args))
  if isinstance(struct_with_args, structure.Struct):
    elements = structure.to_elements(struct_with_args)
  elif isinstance(struct_with_args, typed_object.TypedObject):
    elements = []
    for index, (name, _) in enumerate(
        structure.to_elements(struct_with_args.type_signature)):
      if name is not None:
        elements.append((name, getattr(struct_with_args, name)))
      else:
        elements.append((None, struct_with_args[index]))
  else:
    struct_with_args = computation_types.to_type(struct_with_args)
    struct_with_args.check_struct()
    elements = structure.to_elements(struct_with_args)
  args = []
  kwargs = {}
  for name, value in elements:
    if name is not None:
      kwargs[name] = value
    else:
      args.append(value)
  return args, kwargs


def pack_args_into_struct(args: Sequence[Any],
                          kwargs: Mapping[str, Any],
                          type_spec=None) -> structure.Struct:
  """Packs positional and keyword arguments into a `Struct`.

  If 'type_spec' is not None, it must be a `StructType` or something that's
  convertible to it by computation_types.to_type(). The assignment of arguments
  to fields of the struct follows the same rule as during function calls. If
  'type_spec' is None, the positional arguments precede any of the keyword
  arguments, and the ordering of the keyword arguments matches the ordering in
  which they appear in kwargs. If the latter is an OrderedDict, the ordering
  will be preserved. On the other hand, if the latter is an ordinary unordered
  dict, the ordering is arbitrary.

  Args:
    args: Positional arguments.
    kwargs: Keyword arguments.
    type_spec: The optional type specification (either an instance of
      `computation_types.StructType` or something convertible to it), or None if
      there's no type. Used to drive the arrangements of args into fields of the
      constructed struct, as noted in the description.

  Returns:
    An struct containing all the arguments.

  Raises:
    TypeError: if the arguments are of the wrong computation_types.
  """
  type_spec = computation_types.to_type(type_spec)
  if not type_spec:
    return structure.Struct([(None, arg) for arg in args] +
                            list(kwargs.items()))
  else:
    py_typecheck.check_type(type_spec, computation_types.StructType)
    if not is_argument_struct(type_spec):  # pylint: disable=attribute-error
      raise TypeError(
          'Parameter type {} does not have a structure of an argument struct, '
          'and cannot be populated from multiple positional and keyword '
          'arguments'.format(type_spec))
    else:
      result_elements = []
      positions_used = set()
      keywords_used = set()
      for index, (name,
                  elem_type) in enumerate(structure.to_elements(type_spec)):
        if index < len(args):
          # This argument is present in `args`.
          if name is not None and name in kwargs:
            raise TypeError('Argument `{}` specified twice.'.format(name))
          else:
            arg_value = args[index]
            result_elements.append((name, arg_value))
            positions_used.add(index)
        elif name is not None and name in kwargs:
          # This argument is present in `kwargs`.
          arg_value = kwargs[name]
          result_elements.append((name, arg_value))
          keywords_used.add(name)
        elif name:
          raise TypeError(f'Missing argument `{name}` of type {elem_type}.')
        else:
          raise TypeError(
              f'Missing argument of type {elem_type} at position {index}.')
      positions_missing = set(range(len(args))).difference(positions_used)
      if positions_missing:
        raise TypeError(
            f'Positional arguments at {positions_missing} not used.')
      keywords_missing = set(kwargs.keys()).difference(keywords_used)
      if keywords_missing:
        raise TypeError(f'Keyword arguments at {keywords_missing} not used.')
      return structure.Struct(result_elements)


def pack_args(parameter_type, args: Sequence[Any], kwargs: Mapping[str, Any]):
  """Pack arguments into a single one that matches the given parameter type.

  The arguments may or may not be packed into a `Struct`, depending on the type
  of the parameter, and how many arguments are present.

  Args:
    parameter_type: The type of the single parameter expected by a computation,
      an instance of computation_types.Type or something convertible to it, or
      None if the computation is not expecting a parameter.
    args: Positional arguments of a call.
    kwargs: Keyword arguments of a call.

  Returns:
    A single value object of type that matches 'parameter_type' that contains
    all the arguments, or None if the 'parameter_type' is None.

  Raises:
    TypeError: if the args/kwargs do not match the given parameter type.
  """
  if parameter_type is None:
    # If there's no parameter type, there should be no args of any kind.
    if args or kwargs:
      raise TypeError('Was not expecting any arguments.')
    else:
      return None
  parameter_type = computation_types.to_type(parameter_type)
  if not args and not kwargs:
    raise TypeError(
        'Declared a parameter of type {}, but got no arguments.'.format(
            parameter_type))
  single_positional_arg = (len(args) == 1 and not kwargs)
  if single_positional_arg:
    return args[0]
  if not parameter_type.is_struct():
    # If not a `StructType`, a single positional argument is the only
    # supported call style.
    raise TypeError(
        'Parameter type {} is compatible only with a single positional '
        'argument, but found {} positional and {} keyword args.'.format(
            parameter_type, len(args), len(kwargs)))
  if not is_argument_struct(parameter_type):
    raise TypeError(
        'Parameter type {} does not have a structure of an argument '
        'struct, and cannot be populated from multiple positional and '
        'keyword arguments; please construct a struct before the '
        'call.'.format(parameter_type))
  return pack_args_into_struct(args, kwargs, parameter_type)


def _infer_unpack_needed(fn: types.FunctionType,
                         parameter_type: computation_types.Type,
                         should_unpack: Optional[bool] = None) -> bool:
  """Returns whether parameter_type must be unpacked when calling fn.

  Args:
    fn: The function to be invoked.
    parameter_type: The TFF type of the parameter bundle to be accepted by the
      returned callable.
    should_unpack: Default or expected return value; None implies the inferred
      value should be returned. If either unpacking or packing could work, and
      should_unpack is not None, then should_unpack is returned.

  Returns:
    A `bool` indicating whether or not to unpack.
  """
  # TODO(b/113112885): Revisit whether the 3-way 'unpack' knob is sufficient
  # for our needs, or more options are needed.
  if should_unpack not in [True, False, None]:
    raise TypeError('The unpack argument has an unexpected value {!r}.'.format(
        should_unpack))
  py_typecheck.check_type(parameter_type, computation_types.Type)
  unpack = should_unpack  # Default return value.
  signature = get_signature(fn)
  unpack_required = not is_signature_compatible_with_types(
      signature, parameter_type)
  # Boolean identity comparison becaue unpack can have a non-boolean value.
  if unpack_required and should_unpack is False:  # pylint: disable=g-bool-id-comparison
    raise TypeError(
        'The supplied function \'{}\' with signature {} cannot accept a '
        'value of type \'{}\' as a single argument.'.format(
            fn.__name__, signature, parameter_type))
  if is_argument_struct(parameter_type):
    arg_types, kwarg_types = unpack_args_from_struct(parameter_type)
    unpack_possible = is_signature_compatible_with_types(
        signature, *arg_types, **kwarg_types)
  else:
    unpack_possible = False
  # Boolean identity comparison becaue unpack can have a non-boolean value.
  if not unpack_possible and should_unpack is True:  # pylint: disable=g-bool-id-comparison
    raise TypeError(
        'The supplied function with signature {} cannot accept a value of type '
        '{} as multiple positional and/or keyword arguments. That is, the '
        'argument cannot be unpacked, but unpacking was requested.'.format(
            signature, parameter_type))
  if unpack_required and not unpack_possible:
    raise TypeError(
        'The supplied function "{}" with signature {} cannot accept a value of '
        'type {} as either a single argument or multiple positional and/or '
        'keyword arguments.'.format(fn.__name__, signature, parameter_type))
  if not unpack_required and unpack_possible and should_unpack is None:
    # The supplied function could accept a value as either a single argument,
    # or as multiple positional and/or keyword arguments, and the caller did
    # not specify any preference, leaving ambiguity in how to handle the
    # mapping. We resolve the ambiguity by defaulting to capturing the entire
    # argument, as that's the behavior suggested as expected by the users.
    unpack = False

  if unpack is None:
    # Any ambiguity at this point has been resolved, so the following
    # condition holds and need only be verified in tests.
    assert unpack_required == unpack_possible, (unpack_required,
                                                unpack_possible)
    unpack = unpack_possible

  return unpack


_Arguments = Tuple[List[Any], Dict[str, Any]]


def _unpack_arg(arg_types, kwarg_types, arg) -> _Arguments:
  """Unpacks 'arg' into an argument list based on types."""
  args = []
  for idx, expected_type in enumerate(arg_types):
    element_value = arg[idx]
    actual_type = type_conversions.infer_type(element_value)
    if not expected_type.is_assignable_from(actual_type):
      raise TypeError(
          'Expected element at position {} to be of type {}, found {}.'.format(
              idx, expected_type, actual_type))
    if isinstance(element_value, structure.Struct):
      element_value = type_conversions.type_to_py_container(
          element_value, expected_type)
    args.append(element_value)
  kwargs = {}
  for name, expected_type in kwarg_types.items():
    element_value = getattr(arg, name)
    actual_type = type_conversions.infer_type(element_value)
    if not expected_type.is_assignable_from(actual_type):
      raise TypeError(
          'Expected element named {} to be of type {}, found {}.'.format(
              name, expected_type, actual_type))
    if type_analysis.is_struct_with_py_container(element_value, expected_type):
      element_value = type_conversions.type_to_py_container(
          element_value, expected_type, struct_only=True)
    kwargs[name] = element_value
  return args, kwargs


def _ensure_arg_type(parameter_type, arg) -> _Arguments:
  """Ensures that `arg` matches `parameter_type` before returning it."""
  arg_type = type_conversions.infer_type(arg)
  if not parameter_type.is_assignable_from(arg_type):
    raise TypeError('Expected an argument of type {}, found {}.'.format(
        parameter_type, arg_type))
  arg = type_conversions.type_to_py_container(
      arg, parameter_type, struct_only=True)
  return [arg], {}


def create_argument_unpacking_fn(
    fn: types.FunctionType,
    parameter_type: Optional[computation_types.Type],
    unpack: Optional[bool] = None) -> Callable[[Any], _Arguments]:
  """Returns a function which converts TFF values into arguments to `fn`.

  This function helps to simplify dealing with functions and defuns that might
  have diverse and complex signatures, but that represent computations and as
  such, conceptually only accept a single parameter.

  The argument provided to the returned callable is expected to contain all
  arguments required by `fn` and matching the supplied parameter type signature.
  If `fn` takes multiple parameters, those should be represented by packing
  the arguments to the returned callable into a `Struct`.

  The callable unpacks that structure and returns its elements as an `Arguments`
  structure containing both positional and keyword arguments.

  Example usage:

    @tf.function
    def my_fn(x, y, z=10, name='bar', *p, **q):
      return x + y

    type_spec = (tf.int32, tf.int32)
    argument_converter = create_argument_unpacking_fn(my_fn, type_spec)
    arg = Struct([('x', 10), ('y', 20)])
    args, kwargs = argument_converter(arg)
    ... = my_fn(*args, **kwargs)

  Args:
    fn: The function to unpack arguments for.
    parameter_type: The TFF type of the parameter bundle to be accepted by the
      returned callable.
    unpack: Whether to break the parameter down into constituent parts (`True`),
      leave the parameter as a single unit (False), or allow it to be inferred
      from the signature of `fn` (None). In the latter case (None), if any
      ambiguity arises, an exception is thrown.

  Returns:
    A callable accepting one argument to unpack.

  Raises:
    TypeError: if arguments to this call are of the wrong types, or if the
      supplied 'parameter_type' is not compatible with `fn`.
  """
  if parameter_type is None:

    def _none_arg(arg):
      if arg is not None:
        raise RuntimeError(
            'Unexpected non-`None` argument to no-arg function with '
            f'parameter type `None`: {arg}')
      return [], {}

    return _none_arg
  py_typecheck.check_type(parameter_type, computation_types.Type)
  if _infer_unpack_needed(fn, parameter_type, unpack):
    arg_types, kwarg_types = unpack_args_from_struct(parameter_type)
    return functools.partial(_unpack_arg, arg_types, kwarg_types)
  else:
    return functools.partial(_ensure_arg_type, parameter_type)


class PolymorphicComputation(object):
  """A generic polymorphic function that accepts arguments of diverse types."""

  def __init__(self, concrete_function_factory: Callable[
      [computation_types.Type, Optional[bool]], computation_base.Computation]):
    """Crates a polymorphic function with a given function factory.

    Args:
      concrete_function_factory: A callable that accepts a (non-None) TFF type
        as an argument, as well as an optional boolean `unpack` argument which
        should be treated as documented in `create_argument_unpacking_fn` above.
        The callable must return a `Computation` instance that's been created to
        accept a single positional argument of this TFF type (to be reused for
        future calls with parameters of a matching type).
    """
    self._concrete_function_factory = concrete_function_factory
    self._concrete_function_cache = {}

  def fn_for_argument_type(
      self,
      arg_type: computation_types.Type,
      unpack: Optional[bool] = None) -> computation_base.Computation:
    """Concretizes this function with the provided `arg_type`.

    The first time this function is called with a particular type on a
    given `PolymorphicComputation` (or this `PolymorphicComputation` is called
    with an argument of the given type), the underlying function will be
    traced using the provided argument type as input. Later calls will
    return the cached computed concrete function.

    Args:
      arg_type: The argument type to use when concretizing this function.
      unpack: Whether to force unpacking the arguments (`True`), never unpack
        the arguments (`False`), or infer whether or not to unpack the arguments
        (`None`).

    Returns:
      The `computation_base.Computation` that results from tracing this
      `PolymorphicComputation` with `arg_type.
    """
    key = repr(arg_type) + str(unpack)
    concrete_fn = self._concrete_function_cache.get(key)
    if not concrete_fn:
      concrete_fn = (self._concrete_function_factory)(arg_type, unpack)
      py_typecheck.check_type(concrete_fn, computation_base.Computation,
                              'computation')
      if concrete_fn.type_signature.parameter != arg_type:
        raise TypeError(
            'Expected a concrete function that takes parameter {}, got one '
            'that takes {}.'.format(arg_type,
                                    concrete_fn.type_signature.parameter))
      self._concrete_function_cache[key] = concrete_fn
    return concrete_fn

  def __call__(self, *args, **kwargs):
    """Invokes this polymorphic function with a given set of arguments.

    Args:
      *args: Positional args.
      **kwargs: Keyword args.

    Returns:
      The result of calling a concrete function, instantiated on demand based
      on the argument types (and cached for future calls).

    Raises:
      TypeError: if the concrete functions created by the factory are of the
        wrong computation_types.
    """
    # TODO(b/113112885): We may need to normalize individuals args, such that
    # the type is more predictable and uniform (e.g., if someone supplies an
    # unordered dictionary), possibly by converting dict-like and tuple-like
    # containers into `Struct`s.
    packed_arg = pack_args_into_struct(args, kwargs)
    arg_type = type_conversions.infer_type(packed_arg)
    # We know the argument types have been packed, so force unpacking.
    concrete_fn = self.fn_for_argument_type(arg_type, unpack=True)
    return concrete_fn(packed_arg)
