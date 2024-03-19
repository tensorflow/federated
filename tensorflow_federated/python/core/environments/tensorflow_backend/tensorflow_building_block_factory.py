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
"""A library of construction functions for building block structures."""

from collections.abc import Callable
import functools
from typing import Optional, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions


@functools.lru_cache()
def _create_tensorflow_constant(
    type_spec: computation_types.Type,
    scalar_value: Union[int, float, str],
    name=None,
) -> building_blocks.Call:
  """Creates called graph returning constant `scalar_value` of type `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    type_spec: A `computation_types.Type` whose resulting type tree can only
      contain named tuples and tensors.
    scalar_value: Scalar value to place in all the tensor leaves of `type_spec`.
    name: An optional string name to use as the name of the computation.

  Returns:
    An instance of `building_blocks.Call`, whose argument is `None`
    and whose function is a noarg
    `building_blocks.CompiledComputation` which returns the
    specified `scalar_value` packed into a TFF structure of type `type_spec.

  Raises:
    TypeError: If the type assumptions above are violated.
  """
  proto, function_type = tensorflow_computation_factory.create_constant(
      scalar_value, type_spec
  )
  compiled = building_blocks.CompiledComputation(
      proto, name, type_signature=function_type
  )
  return building_blocks.Call(compiled, None)


def create_null_federated_aggregate() -> building_blocks.Call:
  """Creates an aggregate over an empty struct and returns an empty struct."""
  unit = building_blocks.Struct([])
  unit_type = unit.type_signature
  value = building_block_factory.create_federated_value(
      unit, placements.CLIENTS
  )
  zero = unit
  accumulate_proto, accumulate_type = (
      tensorflow_computation_factory.create_binary_operator(
          lambda a, b: a, unit_type
      )
  )
  accumulate = building_blocks.CompiledComputation(
      accumulate_proto, type_signature=accumulate_type
  )
  merge = accumulate
  report_proto, report_type = tensorflow_computation_factory.create_identity(
      computation_types.StructType([])
  )
  report = building_blocks.CompiledComputation(
      report_proto, type_signature=report_type
  )
  return building_block_factory.create_federated_aggregate(
      value, zero, accumulate, merge, report
  )


def create_null_federated_broadcast():
  return building_block_factory.create_federated_broadcast(
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.SERVER
      )
  )


def create_null_federated_map() -> building_blocks.Call:
  fn_proto, fn_type = tensorflow_computation_factory.create_identity(
      computation_types.StructType([])
  )
  fn = building_blocks.CompiledComputation(fn_proto, type_signature=fn_type)
  return building_block_factory.create_federated_map(
      fn,
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.CLIENTS
      ),
  )


def create_null_federated_secure_modular_sum():
  return building_block_factory.create_federated_secure_modular_sum(
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.CLIENTS
      ),
      building_blocks.Struct([]),
  )


def create_null_federated_secure_sum():
  return building_block_factory.create_federated_secure_sum(
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.CLIENTS
      ),
      building_blocks.Struct([]),
  )


def create_null_federated_secure_sum_bitwidth():
  return building_block_factory.create_federated_secure_sum_bitwidth(
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.CLIENTS
      ),
      building_blocks.Struct([]),
  )


@functools.lru_cache()
def create_generic_constant(
    type_spec: Optional[computation_types.Type], scalar_value: Union[int, float]
) -> building_blocks.ComputationBuildingBlock:
  """Creates constant for a combination of federated, tuple and tensor types.

  Args:
    type_spec: A `computation_types.Type` containing only federated, tuple or
      tensor types, or `None` to use to construct a generic constant.
    scalar_value: The scalar value we wish this constant to have.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock`
    representing `scalar_value` packed into `type_spec`.

  Raises:
    TypeError: If types don't match their specification in the args section.
      Notice validation of consistency of `type_spec` with `scalar_value` is not
      the rsponsibility of this function.
  """
  if type_spec is None:
    return _create_tensorflow_constant(type_spec, scalar_value)
  py_typecheck.check_type(type_spec, computation_types.Type)
  inferred_scalar_value_type = type_conversions.tensorflow_infer_type(
      scalar_value
  )
  if not isinstance(
      inferred_scalar_value_type, computation_types.TensorType
  ) or not array_shape.is_shape_scalar(inferred_scalar_value_type.shape):
    raise TypeError(
        'Must pass a scalar value to `create_generic_constant`; encountered a '
        'value {}'.format(scalar_value)
    )

  def _check_parameters(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.FederatedType,
            computation_types.StructType,
            computation_types.TensorType,
        ),
    )

  if not type_analysis.contains_only(type_spec, _check_parameters):
    raise TypeError

  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.StructType,
            computation_types.TensorType,
        ),
    )

  if type_analysis.contains_only(type_spec, _predicate):
    return _create_tensorflow_constant(type_spec, scalar_value)
  elif isinstance(type_spec, computation_types.FederatedType):
    unplaced_zero = _create_tensorflow_constant(type_spec.member, scalar_value)
    if type_spec.placement is placements.CLIENTS:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True
      )
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type
      )
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri, placement_fn_type
      )
    elif type_spec.placement is placements.SERVER:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True
      )
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type
      )
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri, placement_fn_type
      )
    else:
      raise NotImplementedError(
          f'Unexpected placement found: {type_spec.placement}.'
      )
    return building_blocks.Call(placement_function, unplaced_zero)
  elif isinstance(type_spec, computation_types.StructType):
    elements = []
    for i, _ in enumerate(type_spec):
      elements.append(create_generic_constant(type_spec[i], scalar_value))
    names = [name for name, _ in structure.iter_elements(type_spec)]
    packed_elements = building_blocks.Struct(elements)
    named_tuple = building_block_factory.create_named_tuple(
        packed_elements,
        names,
        type_spec.python_container,
    )
    return named_tuple
  else:
    raise ValueError(
        'The type_spec {} has slipped through all our '
        'generic constant cases, and failed to raise.'.format(type_spec)
    )


def apply_binary_operator_with_upcast(
    arg: building_blocks.ComputationBuildingBlock,
    operator: Callable[[object, object], object],
) -> building_blocks.Call:
  """Constructs result of applying `operator` to `arg` upcasting if appropriate.

  Notice `arg` here must be of federated type, with a named tuple member of
  length 2, or a named tuple type of length 2. If the named tuple type of `arg`
  satisfies certain conditions (that is, there is only a single tensor dtype in
  the first element of `arg`, and the second element represents a scalar of
  this dtype), the second element will be upcast to match the first. Here this
  means it will be pushed into a nested structure matching the structure of the
  first element of `arg`. For example, it makes perfect sense to divide a model
  of type `<a=float32[784],b=float32[10]>` by a scalar of type `float32`, but
  the binary operator constructors we have implemented only take arguments of
  type `<T, T>`. Therefore in this case we would broadcast the `float` argument
  to the `tuple` type, before constructing a binary operator which divides
  pointwise.

  Args:
    arg: `building_blocks.ComputationBuildingBlock` of federated type whose
      `member` attribute is a named tuple type of length 2, or named tuple type
      of length 2.
    operator: Callable representing binary operator to apply to the 2-tuple
      represented by the federated `arg`.

  Returns:
    Instance of `building_blocks.Call`
    encapsulating the result of formally applying `operator` to
    `arg[0], `arg[1]`, upcasting `arg[1]` in the condition described above.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  if isinstance(arg.type_signature, computation_types.FederatedType):
    tuple_type = arg.type_signature.member
    assert isinstance(tuple_type, computation_types.StructType)
  elif isinstance(arg.type_signature, computation_types.StructType):
    tuple_type = arg.type_signature
  else:
    raise TypeError(
        'Generic binary operators are only implemented for federated tuple and '
        'unplaced tuples; you have passed {}.'.format(arg.type_signature)
    )

  tf_representing_proto, tf_representing_type = (
      tensorflow_computation_factory.create_binary_operator_with_upcast(
          operator, tuple_type
      )
  )
  tf_representing_op = building_blocks.CompiledComputation(
      tf_representing_proto, type_signature=tf_representing_type
  )

  if isinstance(arg.type_signature, computation_types.FederatedType):
    called = building_block_factory.create_federated_map_or_apply(
        tf_representing_op, arg
    )
  else:
    called = building_blocks.Call(tf_representing_op, arg)

  return called
