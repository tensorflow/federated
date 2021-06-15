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
"""A factory of intrinsics for use in composing federated computations."""

import warnings

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import symbol_binding_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.federated_context import value_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


def _bind_comp_as_reference(comp):
  fc_context = context_stack_impl.context_stack.current
  if not isinstance(fc_context, symbol_binding_context.SymbolBindingContext):
    raise context_base.ContextError(
        f'Attempted to construct an intrinsic in context {fc_context} which '
        ' does not support binding references.')
  return fc_context.bind_computation_to_reference(comp)


def federated_aggregate(value, zero, accumulate, merge,
                        report) -> value_impl.Value:
  """Aggregates `value` from `tff.CLIENTS` to `tff.SERVER`.

  This generalized aggregation function admits multi-layered architectures that
  involve one or more intermediate stages to handle scalable aggregation across
  a very large number of participants.

  The multi-stage aggregation process is defined as follows:

  * Clients are organized into groups. Within each group, a set of all the
    member constituents of `value` contributed by clients in the group are first
    reduced using reduction operator `accumulate` with `zero` as the zero in the
    algebra. If members of `value` are of type `T`, and `zero` (the result of
    reducing an empty set) is of type `U`, the reduction operator `accumulate`
    used at this stage should be of type `(<U,T> -> U)`. The result of this
    stage is a set of items of type `U`, one item for each group of clients.

  * Next, the `U`-typed items generated by the preceding stage are merged using
    the binary commutative associative operator `merge` of type `(<U,U> -> U)`.
    The result of this stage is a single top-level `U` that emerges at the root
    of the hierarchy at the `tff.SERVER`. Actual implementations may structure
    this step as a cascade of multiple layers.

  * Finally, the `U`-typed result of the reduction performed in the preceding
    stage is projected into the result value using `report` as the mapping
    function (for example, if the structures being merged consist of counters,
    this final step might include computing their ratios).

  Args:
    value: A value of a TFF federated type placed at `tff.CLIENTS` to aggregate.
    zero: The zero of type `U` in the algebra of reduction operators, as
      described above.
    accumulate: The reduction operator to use in the first stage of the process.
      If `value` is of type `{T}@CLIENTS`, and `zero` is of type `U`, this
      operator should be of type `(<U,T> -> U)`.
    merge: The reduction operator to employ in the second stage of the process.
      Must be of type `(<U,U> -> U)`, where `U` is as defined above.
    report: The projection operator to use at the final stage of the process to
      compute the final result of aggregation. If the intended result to be
      returned by `tff.federated_aggregate` is of type `R@SERVER`, this operator
      must be of type `(U -> R)`.

  Returns:
    A representation on the `tff.SERVER` of the result of aggregating `value`
    using the multi-stage process described above.

  Raises:
    TypeError: If the arguments are not of the types specified above.
  """
  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                             'value to be aggregated')

  zero = value_impl.to_value(zero, None)
  py_typecheck.check_type(zero, value_impl.Value)
  accumulate = value_impl.to_value(
      accumulate,
      None,
      parameter_type_hint=computation_types.StructType(
          [zero.type_signature, value.type_signature.member]))
  merge = value_impl.to_value(
      merge,
      None,
      parameter_type_hint=computation_types.StructType(
          [zero.type_signature, zero.type_signature]))
  report = value_impl.to_value(
      report, None, parameter_type_hint=zero.type_signature)
  for op in [accumulate, merge, report]:
    py_typecheck.check_type(op, value_impl.Value)
    py_typecheck.check_type(op.type_signature, computation_types.FunctionType)

  if not accumulate.type_signature.parameter[0].is_assignable_from(
      zero.type_signature):
    raise TypeError('Expected `zero` to be assignable to type {}, '
                    'but was of incompatible type {}.'.format(
                        accumulate.type_signature.parameter[0],
                        zero.type_signature))

  accumulate_type_expected = type_factory.reduction_op(
      accumulate.type_signature.result, value.type_signature.member)
  merge_type_expected = type_factory.reduction_op(
      accumulate.type_signature.result, accumulate.type_signature.result)
  report_type_expected = computation_types.FunctionType(
      merge.type_signature.result, report.type_signature.result)
  for op_name, op, type_expected in [('accumulate', accumulate,
                                      accumulate_type_expected),
                                     ('merge', merge, merge_type_expected),
                                     ('report', report, report_type_expected)]:
    if not type_expected.is_assignable_from(op.type_signature):
      raise TypeError(
          'Expected parameter `{}` to be of type {}, but received {} instead.'
          .format(op_name, type_expected, op.type_signature))

  comp = building_block_factory.create_federated_aggregate(
      value.comp, zero.comp, accumulate.comp, merge.comp, report.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_broadcast(value):
  """Broadcasts a federated value from the `tff.SERVER` to the `tff.CLIENTS`.

  Args:
    value: A value of a TFF federated type placed at the `tff.SERVER`, all
      members of which are equal (the `tff.FederatedType.all_equal` property of
      `value` is `True`).

  Returns:
    A representation of the result of broadcasting: a value of a TFF federated
    type placed at the `tff.CLIENTS`, all members of which are equal.

  Raises:
    TypeError: If the argument is not a federated TFF value placed at the
      `tff.SERVER`.
  """
  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.SERVER,
                                             'value to be broadcasted')

  if not value.type_signature.all_equal:
    raise TypeError('The broadcasted value should be equal at all locations.')

  comp = building_block_factory.create_federated_broadcast(value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_collect(value):
  """Returns a federated value from `tff.CLIENTS` as a `tff.SERVER` sequence.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.

  Returns:
    A stream of the same type as the member constituents of `value` placed at
    the `tff.SERVER`.

  Raises:
    TypeError: If the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                             'value to be collected')

  comp = building_block_factory.create_federated_collect(value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_eval(fn, placement):
  """Evaluates a federated computation at `placement`, returning the result.

  Args:
    fn: A no-arg TFF computation.
    placement: The desired result placement (either `tff.SERVER` or
      `tff.CLIENTS`).

  Returns:
    A federated value with the given placement `placement`.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  # TODO(b/113112108): Verify that neither the value, nor any of its parts
  # are of a federated type.

  fn = value_impl.to_value(fn, None)
  py_typecheck.check_type(fn, value_impl.Value)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)

  if fn.type_signature.parameter is not None:
    raise TypeError(
        '`federated_eval` expects a `fn` that accepts no arguments, but '
        'the `fn` provided has a parameter of type {}.'.format(
            fn.type_signature.parameter))

  comp = building_block_factory.create_federated_eval(fn.comp, placement)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_map(fn, arg):
  """Maps a federated value pointwise using a mapping function.

  The function `fn` is applied separately across the group of devices
  represented by the placement type of `arg`. For example, if `value` has
  placement type `tff.CLIENTS`, then `fn` is applied to each client
  individually. In particular, this operation does not alter the placement of
  the federated value.

  Args:
    fn: A mapping function to apply pointwise to member constituents of `arg`.
      The parameter of this function must be of the same type as the member
      constituents of `arg`.
    arg: A value of a TFF federated type (or a value that can be implicitly
      converted into a TFF federated type, e.g., by zipping) placed at
      `tff.CLIENTS` or `tff.SERVER`.

  Returns:
    A federated value with the same placement as `arg` that represents the
    result of `fn` on the member constituent of `arg`.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  # TODO(b/113112108): Possibly lift the restriction that the mapped value
  # must be placed at the server or clients. Would occur after adding support
  # for placement labels in the federated types, and expanding the type
  # specification of the intrinsic this is based on to work with federated
  # values of arbitrary placement.

  arg = value_impl.to_value(arg, None)
  arg = value_utils.ensure_federated_value(arg, label='value to be mapped')

  fn = value_impl.to_value(
      fn, None, parameter_type_hint=arg.type_signature.member)

  py_typecheck.check_type(fn, value_impl.Value)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  if not fn.type_signature.parameter.is_assignable_from(
      arg.type_signature.member):
    raise TypeError(
        'The mapping function expects a parameter of type {}, but member '
        'constituents of the mapped value are of incompatible type {}.'.format(
            fn.type_signature.parameter, arg.type_signature.member))

  # TODO(b/144384398): Change structure to one that maps the placement type
  # to the building_block function that fits it, in a way that allows the
  # appropriate type checks.
  if arg.type_signature.placement is placements.SERVER:
    if not arg.type_signature.all_equal:
      raise TypeError(
          'Arguments placed at {} should be equal at all locations.'.format(
              placements.SERVER))
    comp = building_block_factory.create_federated_apply(fn.comp, arg.comp)
  elif arg.type_signature.placement is placements.CLIENTS:
    comp = building_block_factory.create_federated_map(fn.comp, arg.comp)
  else:
    raise TypeError('Expected `arg` to have a type with a supported placement, '
                    'found {}.'.format(arg.type_signature.placement))

  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_map_all_equal(fn, arg):
  """`federated_map` with the `all_equal` bit set in the `arg` and return."""
  # TODO(b/113112108): Possibly lift the restriction that the mapped value
  # must be placed at the clients after adding support for placement labels
  # in the federated types, and expanding the type specification of the
  # intrinsic this is based on to work with federated values of arbitrary
  # placement.
  arg = value_impl.to_value(arg, None)
  arg = value_utils.ensure_federated_value(arg, placements.CLIENTS,
                                           'value to be mapped')

  fn = value_impl.to_value(
      fn, None, parameter_type_hint=arg.type_signature.member)

  py_typecheck.check_type(fn, value_impl.Value)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  if not fn.type_signature.parameter.is_assignable_from(
      arg.type_signature.member):
    raise TypeError(
        'The mapping function expects a parameter of type {}, but member '
        'constituents of the mapped value are of incompatible type {}.'.format(
            fn.type_signature.parameter, arg.type_signature.member))

  comp = building_block_factory.create_federated_map_all_equal(
      fn.comp, arg.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_mean(value, weight=None):
  """Computes a `tff.SERVER` mean of `value` placed on `tff.CLIENTS`.

  For values `v_1, ..., v_k`, and weights `w_1, ..., w_k`, this means
  `sum_{i=1}^k (w_i * v_i) / sum_{i=1}^k w_i`.

  Args:
    value: The value of which the mean is to be computed. Must be of a TFF
      federated type placed at `tff.CLIENTS`. The value may be structured, e.g.,
      its member constituents can be named tuples. The tensor types that the
      value is composed of must be floating-point or complex.
    weight: An optional weight, a TFF federated integer or floating-point tensor
      value, also placed at `tff.CLIENTS`.

  Returns:
    A representation at the `tff.SERVER` of the mean of the member constituents
    of `value`, optionally weighted with `weight` if specified (otherwise, the
    member constituents contributed by all clients are equally weighted).

  Raises:
    TypeError: If `value` is not a federated TFF value placed at `tff.CLIENTS`,
      or if `weight` is not a federated integer or a floating-point tensor with
      the matching placement.
  """
  # TODO(b/113112108): Possibly relax the constraints on numeric types, and
  # inject implicit casts where appropriate. For instance, we might want to
  # allow `tf.int32` values as the input, and automatically cast them to
  # `tf.float321 before invoking the average, thus producing a floating-point
  # result.

  # TODO(b/120439632): Possibly allow the weight to be either structured or
  # non-scalar, e.g., for the case of averaging a convolutional layer, when
  # we would want to use a different weight for every filter, and where it
  # might be cumbersome for users to have to manually slice and assemble a
  # variable.

  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                             'value to be averaged')
  if not type_analysis.is_average_compatible(value.type_signature):
    raise TypeError(
        'The value type {} is not compatible with the average operator.'.format(
            value.type_signature))

  if weight is not None:
    weight = value_impl.to_value(weight, None)
    weight = value_utils.ensure_federated_value(weight, placements.CLIENTS,
                                                'weight to use in averaging')
    py_typecheck.check_type(weight.type_signature.member,
                            computation_types.TensorType)
    if weight.type_signature.member.shape.ndims != 0:
      raise TypeError('The weight type {} is not a federated scalar.'.format(
          weight.type_signature))
    if not (weight.type_signature.member.dtype.is_integer or
            weight.type_signature.member.dtype.is_floating):
      raise TypeError(
          'The weight type {} is not a federated integer or floating-point '
          'tensor.'.format(weight.type_signature))

  weight_comp = None if weight is None else weight.comp
  comp = building_block_factory.create_federated_mean(value.comp, weight_comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_sum(value):
  """Computes a sum at `tff.SERVER` of a `value` placed on the `tff.CLIENTS`.

  To sum integer values with stronger privacy properties, consider using
  `tff.federated_secure_sum_bitwidth`.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.

  Returns:
    A representation of the sum of the member constituents of `value` placed
    on the `tff.SERVER`.

  Raises:
    TypeError: If the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                             'value to be summed')
  type_analysis.check_is_sum_compatible(value.type_signature)
  comp = building_block_factory.create_federated_sum(value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_value(value, placement):
  """Returns a federated value at `placement`, with `value` as the constituent.

  Deprecation warning: Using `tff.federated_value` with arguments other than
  simple Python constants is deprecated. When placing the result of a
  `tf_computation`, prefer `tff.federated_eval`.

  Args:
    value: A value of a non-federated TFF type to be placed.
    placement: The desired result placement (either `tff.SERVER` or
      `tff.CLIENTS`).

  Returns:
    A federated value with the given placement `placement`, and the member
    constituent `value` equal at all locations.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  if isinstance(value, value_impl.Value):
    warnings.warn(
        'Deprecation warning: Using `tff.federated_value` with arguments '
        'other than simple Python constants is deprecated. When placing the '
        'result of a `tf_computation`, prefer `tff.federated_eval`.',
        DeprecationWarning)
  value = value_impl.to_value(value, None)
  if type_analysis.contains(value.type_signature, lambda t: t.is_federated()):
    raise TypeError('Cannot place value {} containing federated types at '
                    'another placement; requested to be placed at {}.'.format(
                        value, placement))

  comp = building_block_factory.create_federated_value(value.comp, placement)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_zip(value):
  """Converts an N-tuple of federated values into a federated N-tuple value.

  Args:
    value: A value of a TFF named tuple type, the elements of which are
      federated values with the same placement.

  Returns:
    A federated value placed at the same location as the members of `value`, in
    which every member component is a named tuple that consists of the
    corresponding member components of the elements of `value`.

  Raises:
    TypeError: If the argument is not a named tuple of federated values with the
      same placement.
  """
  # TODO(b/113112108): We use the iterate/unwrap approach below because
  # our type system is not powerful enough to express the concept of
  # "an operation that takes tuples of T of arbitrary length", and therefore
  # the intrinsic federated_zip must only take a fixed number of arguments,
  # here fixed at 2. There are other potential approaches to getting around
  # this problem (e.g. having the operator act on sequences and thereby
  # sidestepping the issue) which we may want to explore.
  value = value_impl.to_value(value, None)
  py_typecheck.check_type(value, value_impl.Value)
  py_typecheck.check_type(value.type_signature, computation_types.StructType)

  comp = building_block_factory.create_federated_zip(value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def _select_parameter_mismatch(
    param_type,
    type_desc,
    name,
    secure,
    expected_type=None,
):
  """Throws a `TypeError` indicating a mismatched `select` parameter type."""
  secure_string = '_secure' if secure else ''
  intrinsic_name = f'federated{secure_string}_select'
  message = (
      f'Expected `{intrinsic_name}` parameter `{name}` to be {type_desc}')
  if expected_type is None:
    raise TypeError(f'{message}, found value of type {param_type}')
  else:
    raise TypeError(f'{message}:\n' +
                    computation_types.type_mismatch_error_message(
                        param_type,
                        expected_type,
                        computation_types.TypeRelation.ASSIGNABLE,
                        second_is_expected=True))


def _check_select_keys_type(keys_type, secure):
  if not (keys_type.is_federated and keys_type.placement.is_clients()):
    _select_parameter_mismatch(keys_type, 'a federated value placed at clients',
                               'client_keys', secure)
  if not (keys_type.member.is_tensor() and keys_type.member.dtype == tf.int32
          and keys_type.member.shape.rank == 1 and
          keys_type.member.shape.dims[0].value is not None):
    _select_parameter_mismatch(
        keys_type.member, 'a one-dimensional fixed-length tf.int32 tensor',
        'client_keys.type_signature.member', secure)


def federated_select(client_keys, max_key, server_val, select_fn):
  """Sends selected values from a server database to clients.

  Args:
    client_keys: `tff.CLIENTS`-placed one-dimensional fixed-size `int32` keys
      used to select values from `database` to load for each client.
    max_key: A `tff.SERVER`-placed `int32` which is guaranteed to be greater
      than any of `client_keys`. Lower values may permit more optimizations.
    server_val: `tff.SERVER`-placed value used as an input to `select_fn`.
    select_fn: A function which accepts `server_val` and a `int32` client key
      and returns a value to be sent to the client. `select_fn` should be
      deterministic (nonrandom).

  Returns:
    `tff.CLIENTS`-placed sequences of values returned from `select_fn`. In each
    sequence, the order of values will match the order of keys in the
    corresponding `client_keys` tensor. For example, a client with keys
    `[1, 2, ...]` will receive a sequence of values
    `[select_fn(server_val, 1), select_fn(server_val, 2), ...]`.

  Raises:
    TypeError: If `client_keys` is not of type `{int32[N]}@CLIENTS`, if
      `max_key` is not of type `int32@SERVER`, if `server_val` is not a
      server-placed value (`S@SERVER`), or if `select_fn` is not a function
      of type `<S, int32> -> RESULT`.
  """
  return _federated_select(
      client_keys, max_key, server_val, select_fn, secure=False)


def federated_secure_select(client_keys, max_key, server_val, select_fn):
  """Sends privately-selected values from a server database to  clients.

  Args:
    client_keys: `tff.CLIENTS`-placed one-dimensional fixed-size `int32` keys
      used to select values from `database` to load for each client.
    max_key: A `tff.SERVER`-placed `int32` which is guaranteed to be greater
      than any of `client_keys`. Lower values may permit more optimizations.
    server_val: `tff.SERVER`-placed value used as an input to `select_fn`.
    select_fn: A function which accepts `server_val` and a `int32` client key
      and returns a value to be sent to the client. `select_fn` should be
      deterministic (nonrandom).

  Returns:
    `tff.CLIENTS`-placed sequences of values returned from `select_fn`. In each
    sequence, the order of values will match the order of keys in the
    corresponding `client_keys` tensor. For example, a client with keys
    `[1, 2, ...]` will receive a sequence of values
    `[select_fn(server_val, 1), select_fn(server_val, 2), ...]`.

  Raises:
    TypeError: If `client_keys` is not of type `{int32[N]}@CLIENTS`, if
      `max_key` is not of type `int32@SERVER`, if `server_val` is not a
      server-placed value (`S@SERVER`), or if `select_fn` is not a function
      of type `<S, int32> -> RESULT`.
  """
  return _federated_select(
      client_keys, max_key, server_val, select_fn, secure=True)


def _federated_select(client_keys, max_key, server_val, select_fn, secure):
  """Internal helper for `federated_select` and `federated_secure_select`."""
  client_keys = value_impl.to_value(client_keys, None)
  _check_select_keys_type(client_keys.type_signature, secure)
  max_key = value_impl.to_value(max_key, None)
  expected_max_key_type = computation_types.at_server(tf.int32)
  if not expected_max_key_type.is_assignable_from(max_key.type_signature):
    _select_parameter_mismatch(
        max_key.type_signature,
        'a 32-bit unsigned integer placed at server',
        'max_key',
        secure,
        expected_type=expected_max_key_type)
  server_val = value_impl.to_value(server_val, None)
  expected_server_val_type = computation_types.at_server(
      computation_types.AbstractType('T'))
  if (not server_val.type_signature.is_federated() or
      not server_val.type_signature.placement.is_server()):
    _select_parameter_mismatch(
        server_val.type_signature,
        'a value placed at server',
        'server_val',
        secure,
        expected_type=expected_server_val_type)
  select_fn_param_type = computation_types.to_type(
      [server_val.type_signature.member, tf.int32])
  select_fn = value_impl.to_value(
      select_fn, None, parameter_type_hint=select_fn_param_type)
  expected_select_fn_type = computation_types.FunctionType(
      select_fn_param_type, computation_types.AbstractType('U'))
  if (not select_fn.type_signature.is_function() or
      not select_fn.type_signature.parameter.is_assignable_from(
          select_fn_param_type)):
    _select_parameter_mismatch(
        select_fn.type_signature,
        'a function from state and key to result',
        'select_fn',
        secure,
        expected_type=expected_select_fn_type)
  comp = building_block_factory.create_federated_select(client_keys.comp,
                                                        max_key.comp,
                                                        server_val.comp,
                                                        select_fn.comp, secure)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def federated_secure_sum_bitwidth(value, bitwidth):
  """Computes a sum at `tff.SERVER` of a `value` placed on the `tff.CLIENTS`.

  This function computes a sum such that it should not be possible for the
  server to learn any clients individual value. The specific algorithm and
  mechanism used to compute the secure sum may vary depending on the target
  runtime environment the computation is compiled for or executed on. See
  https://research.google/pubs/pub47246/ for more information.

  Not all executors support `tff.federated_secure_sum_bitwidth()`; consult the
  documentation for the specific executor or executor stack you plan on using
  for the specific of how it's handled by that executor.

  The `bitwidth` argument represents the bitwidth of the aggregand, that is the
  bitwidth of the input `value`. The federated secure sum bitwidth (i.e., the
  bitwidth of the *sum* of the input `value`s over all clients) will be a
  function of this bitwidth and the number of participating clients.

  Example:

  ```python
  value = tff.federated_value(1, tff.CLIENTS)
  result = tff.federated_secure_sum_bitwidth(value, 2)

  value = tff.federated_value([1, 1], tff.CLIENTS)
  result = tff.federated_secure_sum_bitwidth(value, [2, 4])

  value = tff.federated_value([1, [1, 1]], tff.CLIENTS)
  result = tff.federated_secure_sum_bitwidth(value, [2, [4, 8]])
  ```

  Note: To sum non-integer values or to sum integers with fewer constraints and
  weaker privacy properties, consider using `federated_sum`.

  Args:
    value: An integer value of a TFF federated type placed at the `tff.CLIENTS`,
      in the range [0, 2^bitwidth - 1].
    bitwidth: An integer or nested structure of integers matching the structure
      of `value`. If integer `bitwidth` is used with a nested `value`, the same
      integer is used for each tensor in `value`.

  Returns:
    A representation of the sum of the member constituents of `value` placed
    on the `tff.SERVER`.

  Raises:
    TypeError: If the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  value = value_impl.to_value(value, None)
  value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                             'value to be summed')
  type_analysis.check_is_structure_of_integers(value.type_signature)
  bitwidth_value = value_impl.to_value(bitwidth, None)
  value_member_type = value.type_signature.member
  bitwidth_type = bitwidth_value.type_signature
  if not type_analysis.is_valid_bitwidth_type_for_value_type(
      bitwidth_type, value_member_type):
    raise TypeError(
        'Expected `federated_secure_sum_bitwidth` parameter `bitwidth` to match '
        'the structure of `value`, with one integer bitwidth per tensor in '
        '`value`. Found `value` of `{}` and `bitwidth` of `{}`.'.format(
            value_member_type, bitwidth_type))
  if bitwidth_type.is_tensor() and value_member_type.is_struct():
    bitwidth_value = value_impl.to_value(
        structure.map_structure(lambda _: bitwidth, value_member_type), None)
  comp = building_block_factory.create_federated_secure_sum_bitwidth(
      value.comp, bitwidth_value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)


def sequence_map(fn, arg):
  """Maps a TFF sequence `value` pointwise using a given function `fn`.

  This function supports two modes of usage:

  * When applied to a non-federated sequence, it maps individual elements of
    the sequence pointwise. If the supplied `fn` is of type `T->U` and
    the sequence `arg` is of type `T*` (a sequence of `T`-typed elements),
    the result is a sequence of type `U*` (a sequence of `U`-typed elements),
    with each element of the input sequence individually mapped by `fn`.
    In this mode of usage, `sequence_map` behaves like a compuatation with type
    signature `<T->U,T*> -> U*`.

  * When applied to a federated sequence, `sequence_map` behaves as if it were
    individually applied to each member constituent. In this mode of usage, one
    can think of `sequence_map` as a specialized variant of `federated_map` that
    is designed to work with sequences and allows one to
    specify a `fn` that operates at the level of individual elements.
    Indeed, under the hood, when `sequence_map` is invoked on a federated type,
    it injects `federated_map`, thus
    emitting expressions like
    `federated_map(a -> sequence_map(fn, x), arg)`.

  Args:
    fn: A mapping function to apply pointwise to elements of `arg`.
    arg: A value of a TFF type that is either a sequence, or a federated
      sequence.

  Returns:
    A sequence with the result of applying `fn` pointwise to each
    element of `arg`, or if `arg` was federated, a federated sequence
    with the result of invoking `sequence_map` on member sequences locally
    and independently at each location.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  fn = value_impl.to_value(fn, None)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  arg = value_impl.to_value(arg, None)

  if arg.type_signature.is_sequence():
    comp = building_block_factory.create_sequence_map(fn.comp, arg.comp)
    comp = _bind_comp_as_reference(comp)
    return value_impl.Value(comp)
  elif arg.type_signature.is_federated():
    parameter_type = computation_types.SequenceType(fn.type_signature.parameter)
    result_type = computation_types.SequenceType(fn.type_signature.result)
    intrinsic_type = computation_types.FunctionType(
        (fn.type_signature, parameter_type), result_type)
    intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_MAP.uri,
                                          intrinsic_type)
    intrinsic_impl = value_impl.Value(intrinsic)
    local_fn = value_utils.get_curried(intrinsic_impl)(fn)
    return federated_map(local_fn, arg)
  else:
    raise TypeError(
        'Cannot apply `tff.sequence_map()` to a value of type {}.'.format(
            arg.type_signature))


def sequence_reduce(value, zero, op):
  """Reduces a TFF sequence `value` given a `zero` and reduction operator `op`.

  This method reduces a set of elements of a TFF sequence `value`, using a given
  `zero` in the algebra (i.e., the result of reducing an empty sequence) of some
  type `U`, and a reduction operator `op` with type signature `(<U,T> -> U)`
  that incorporates a single `T`-typed element of `value` into the `U`-typed
  result of partial reduction. In the special case of `T` equal to `U`, this
  corresponds to the classical notion of reduction of a set using a commutative
  associative binary operator. The generalized reduction (with `T` not equal to
  `U`) requires that repeated application of `op` to reduce a set of `T` always
  yields the same `U`-typed result, regardless of the order in which elements
  of `T` are processed in the course of the reduction.

  One can also invoke `sequence_reduce` on a federated sequence, in which case
  the reductions are performed pointwise; under the hood, we construct an
  expression  of the form
  `federated_map(x -> sequence_reduce(x, zero, op), value)`. See also the
  discussion on `sequence_map`.

  Note: When applied to a federated value this function does the reduce
  point-wise.

  Args:
    value: A value that is either a TFF sequence, or a federated sequence.
    zero: The result of reducing a sequence with no elements.
    op: An operator with type signature `(<U,T> -> U)`, where `T` is the type of
      the elements of the sequence, and `U` is the type of `zero` to be used in
      performing the reduction.

  Returns:
    The `U`-typed result of reducing elements in the sequence, or if the `value`
    is federated, a federated `U` that represents the result of locally
    reducing each member constituent of `value`.

  Raises:
    TypeError: If the arguments are not of the types specified above.
  """
  value = value_impl.to_value(value, None)
  zero = value_impl.to_value(zero, None)
  op = value_impl.to_value(op, None)
  # Check if the value is a federated sequence that should be reduced
  # under a `federated_map`.
  if value.type_signature.is_federated():
    is_federated_sequence = True
    value_member_type = value.type_signature.member
    value_member_type.check_sequence()
    zero_member_type = zero.type_signature.member
  else:
    is_federated_sequence = False
    value.type_signature.check_sequence()
  if not is_federated_sequence:
    comp = building_block_factory.create_sequence_reduce(
        value.comp, zero.comp, op.comp)
    comp = _bind_comp_as_reference(comp)
    return value_impl.Value(comp)
  else:
    ref_type = computation_types.StructType(
        [value_member_type, zero_member_type])
    ref = building_blocks.Reference('arg', ref_type)
    arg1 = building_blocks.Selection(ref, index=0)
    arg2 = building_blocks.Selection(ref, index=1)
    call = building_block_factory.create_sequence_reduce(arg1, arg2, op.comp)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, call)
    fn_value_impl = value_impl.Value(fn)
    args = building_blocks.Struct([value.comp, zero.comp])
    return federated_map(fn_value_impl, args)


def sequence_sum(value):
  """Computes a sum of elements in a sequence.

  Args:
    value: A value of a TFF type that is either a sequence, or a federated
      sequence.

  Returns:
    The sum of elements in the sequence. If the argument `value` is of a
    federated type, the result is also of a federated type, with the sum
    computed locally and independently at each location (see also a discussion
    on `sequence_map` and `sequence_reduce`).

  Raises:
    TypeError: If the arguments are of wrong or unsupported types.
  """
  value = value_impl.to_value(value, None)
  if value.type_signature.is_sequence():
    element_type = value.type_signature.element
  else:
    py_typecheck.check_type(value.type_signature,
                            computation_types.FederatedType)
    py_typecheck.check_type(value.type_signature.member,
                            computation_types.SequenceType)
    element_type = value.type_signature.member.element
  type_analysis.check_is_sum_compatible(element_type)

  if value.type_signature.is_sequence():
    comp = building_block_factory.create_sequence_sum(value.comp)
    comp = _bind_comp_as_reference(comp)
    return value_impl.Value(comp)
  elif value.type_signature.is_federated():
    intrinsic_type = computation_types.FunctionType(
        value.type_signature.member, value.type_signature.member.element)
    intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_SUM.uri,
                                          intrinsic_type)
    intrinsic_impl = value_impl.Value(intrinsic)
    return federated_map(intrinsic_impl, value)
  else:
    raise TypeError(
        'Cannot apply `tff.sequence_sum()` to a value of type {}.'.format(
            value.type_signature))
