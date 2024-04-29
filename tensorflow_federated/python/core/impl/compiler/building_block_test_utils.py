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
"""Utilities for testing building blocks."""

import numpy as np

from google.protobuf import any_pb2
from tensorflow_federated.python.core.impl.compiler import array
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def create_test_any_proto_for_array_value(value: np.ndarray):
  """Creates an Any proto for the given value."""
  test_proto = array.to_proto(value)
  any_proto = any_pb2.Any()
  any_proto.Pack(test_proto)
  return any_proto


def create_chained_calls(functions, arg):
  r"""Creates a chain of `n` calls.

       Call
      /    \
  Comp      ...
               \
                Call
               /    \
           Comp      Comp

  The first functional computation in `functions` must have a parameter type
  that is assignable from the type of `arg`, each other functional computation
  in `functions` must have a parameter type that is assignable from the previous
  functional computations result type.

  Args:
    functions: A Python list of functional computations.
    arg: A `building_blocks.ComputationBuildingBlock`.

  Returns:
    A `building_blocks.Call`.
  """
  for fn in functions:
    if not fn.parameter_type.is_assignable_from(arg.type_signature):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature)
          )
      )
    call = building_blocks.Call(fn, arg)
    arg = call
  return call


def create_whimsy_block(comp, variable_name, variable_type=np.int32):
  r"""Returns an identity block.

           Block
          /     \
     [x=1]       Comp

  Args:
    comp: The computation to use as the result.
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  ref = building_blocks.Literal(1, computation_types.TensorType(variable_type))
  return building_blocks.Block([(variable_name, ref)], comp)


def create_whimsy_called_intrinsic(parameter_name, parameter_type=np.int32):
  r"""Returns a whimsy called intrinsic.

            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  intrinsic_type = computation_types.FunctionType(
      parameter_type, parameter_type
  )
  intrinsic = building_blocks.Intrinsic('intrinsic', intrinsic_type)
  ref = building_blocks.Reference(parameter_name, parameter_type)
  return building_blocks.Call(intrinsic, ref)


def create_whimsy_called_federated_aggregate(
    accumulate_parameter_name='acc_param',
    merge_parameter_name='merge_param',
    report_parameter_name='report_param',
    value_type=np.int32,
):
  r"""Returns a whimsy called federated aggregate.

                      Call
                     /    \
  federated_aggregate      Tuple
                           |
                           [Lit(1), Lit(1), Lambda(x),   Lambda(x),   Lambda(x)]
                                            |            |            |
                                            Lit(1)       Lit(1)       Lit(1)

  Args:
    accumulate_parameter_name: The name of the accumulate parameter.
    merge_parameter_name: The name of the merge parameter.
    report_parameter_name: The name of the report parameter.
    value_type: The TFF type of the value to be aggregated, placed at CLIENTS.
  """
  tensor_type = computation_types.TensorType(value_type)
  value = building_block_factory.create_federated_value(
      building_blocks.Literal(1, tensor_type), placements.CLIENTS
  )
  literal_block = building_blocks.Literal(1, tensor_type)
  zero = literal_block
  accumulate_type = computation_types.StructType((value_type, value_type))
  accumulate_result = literal_block
  accumulate = building_blocks.Lambda(
      accumulate_parameter_name, accumulate_type, accumulate_result
  )
  merge_type = computation_types.StructType((value_type, value_type))
  merge_result = literal_block
  merge = building_blocks.Lambda(merge_parameter_name, merge_type, merge_result)
  report_result = literal_block
  report = building_blocks.Lambda(
      report_parameter_name, value_type, report_result
  )
  return building_block_factory.create_federated_aggregate(
      value, zero, accumulate, merge, report
  )


def create_whimsy_called_federated_apply(
    parameter_name, parameter_type: computation_types._DtypeLike = np.int32
):
  r"""Returns a whimsy called federated apply.

                  Call
                 /    \
  federated_apply      Tuple
                       |
                       [Lambda(x), Lit(1)]
                        |
                        Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  val = parameter_type(1)
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.TensorType(parameter_type)
  arg = building_block_factory.create_federated_value(
      building_blocks.Literal(val, arg_type), placement=placements.SERVER
  )
  return building_block_factory.create_federated_apply(fn, arg)


def create_whimsy_called_federated_broadcast(
    value_type: computation_types._DtypeLike = np.int32,
):
  r"""Returns a whimsy called federated broadcast.

                      Call
                     /    \
  federated_broadcast      Lit(1)

  Args:
    value_type: The type of the value.
  """
  val = value_type(1)
  tensor_type = computation_types.TensorType(value_type)
  value = building_block_factory.create_federated_value(
      building_blocks.Literal(val, tensor_type), placement=placements.SERVER
  )
  return building_block_factory.create_federated_broadcast(value)


def create_whimsy_called_federated_map(
    parameter_name, parameter_type: computation_types._DtypeLike = np.int32
):
  r"""Returns a whimsy called federated map.

                Call
               /    \
  federated_map      Tuple
                     |
                     [Lambda(x), Lit(1)]
                      |
                      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  val = parameter_type(1)
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.TensorType(parameter_type)
  arg = building_block_factory.create_federated_value(
      building_blocks.Literal(val, arg_type), placement=placements.CLIENTS
  )
  # pylint: disable=protected-access
  arg._type_signature = computation_types.FederatedType(
      arg_type, placements.CLIENTS, all_equal=False
  )
  # pylint: enable=protected-access
  return building_block_factory.create_federated_map(fn, arg)


def create_whimsy_called_federated_map_all_equal(
    parameter_name, parameter_type: computation_types._DtypeLike = np.int32
):
  r"""Returns a whimsy called federated map.

                          Call
                         /    \
  federated_map_all_equal      Tuple
                               |
                               [Lambda(x), Lit(1)]
                                |
                                Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  val = parameter_type(1)
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.TensorType(parameter_type)
  arg = building_block_factory.create_federated_value(
      building_blocks.Literal(val, arg_type), placement=placements.CLIENTS
  )
  return building_block_factory.create_federated_map_all_equal(fn, arg)


def create_whimsy_called_federated_mean(
    value_type: computation_types._DtypeLike = np.float32,
    weights_type: computation_types._DtypeLike | None = None,
):
  """Returns a called federated mean."""
  val = value_type(1)
  value_type = computation_types.TensorType(value_type)
  values = building_block_factory.create_federated_value(
      building_blocks.Literal(val, value_type),
      placement=placements.CLIENTS,
  )
  if weights_type is not None:
    weights_val = weights_type(1)
    weights_type = computation_types.TensorType(weights_type)
    weights = building_block_factory.create_federated_value(
        building_blocks.Literal(weights_val, weights_type),
        placement=placements.CLIENTS,
    )
  else:
    weights = None
  return building_block_factory.create_federated_mean(values, weights)


def create_whimsy_called_federated_secure_sum_bitwidth(
    value_type: computation_types._DtypeLike = np.int32,
):
  r"""Returns a whimsy called secure sum.

                       Call
                      /    \
  federated_secure_sum_bitwidth      [Lit(1), Lit(1)]

  Args:
    value_type: The type of the value.
  """
  val = value_type(1)
  tensor_type = computation_types.TensorType(value_type)
  value = building_block_factory.create_federated_value(
      building_blocks.Literal(val, tensor_type), placement=placements.CLIENTS
  )
  bitwidth = building_blocks.Literal(val, tensor_type)
  return building_block_factory.create_federated_secure_sum_bitwidth(
      value, bitwidth
  )


def create_whimsy_called_federated_sum(
    value_type: computation_types._DtypeLike = np.int32,
):
  r"""Returns a whimsy called federated sum.

                Call
               /    \
  federated_sum      Lit(a)

  Args:
    value_type: The type of the value.
  """
  val = value_type(1)
  tensor_type = computation_types.TensorType(value_type)
  value = building_block_factory.create_federated_value(
      building_blocks.Literal(val, tensor_type), placement=placements.CLIENTS
  )
  return building_block_factory.create_federated_sum(value)


def create_whimsy_called_sequence_map(
    parameter_name, parameter_type=np.int32, any_proto=any_pb2.Any()
):
  r"""Returns a whimsy called sequence map.

               Call
              /    \
  sequence_map      Data(id)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
    any_proto: The any proto to use for the data block.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.SequenceType(parameter_type)
  arg = building_blocks.Data(any_proto, arg_type)
  return building_block_factory.create_sequence_map(fn, arg)


def create_whimsy_called_federated_value(
    placement: placements.PlacementLiteral,
    value_type: computation_types._DtypeLike = np.int32,
):
  val = value_type(1)
  value = building_blocks.Literal(val, computation_types.TensorType(value_type))
  return building_block_factory.create_federated_value(value, placement)


def create_identity_block(variable_name, comp):
  r"""Returns an identity block.

           Block
          /     \
  [x=comp]       Ref(x)

  Args:
    variable_name: The name of the variable.
    comp: The computation to use as the variable.
  """
  ref = building_blocks.Reference(variable_name, comp.type_signature)
  return building_blocks.Block([(variable_name, comp)], ref)


def create_identity_block_with_whimsy_ref(
    variable_name, variable_type: computation_types._DtypeLike = np.int32
):
  r"""Returns an identity block with a whimsy `ref` computation.

             Block
            /     \
  [x=Lit(1)]       Ref(x)

  Args:
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  val = variable_type(1)
  literal = building_blocks.Literal(
      val, computation_types.TensorType(variable_type)
  )
  return create_identity_block(variable_name, literal)


def create_identity_function(parameter_name, parameter_type=np.int32):
  r"""Returns an identity function.

  Lambda(x)
  |
  Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  ref = building_blocks.Reference(parameter_name, parameter_type)
  return building_blocks.Lambda(ref.name, ref.type_signature, ref)


def create_lambda_to_whimsy_called_intrinsic(
    parameter_name, parameter_type=np.int32
):
  r"""Returns a lambda to call a whimsy intrinsic.

            Lambda(x)
            |
            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  call = create_whimsy_called_intrinsic(
      parameter_name=parameter_name, parameter_type=parameter_type
  )
  return building_blocks.Lambda(parameter_name, parameter_type, call)


def create_nested_syntax_tree():
  r"""Constructs computation with explicit ordering for testing traversals.

  The goal of this computation is to exercise each switch
  in transform_postorder_with_symbol_bindings, at least all those that recurse.

  The computation this function constructs can be represented as below.

  Notice that the body of the Lambda *does not depend on the Lambda's
  parameter*, so that if we were actually executing this call the argument will
  be thrown away.

  All leaf nodes are instances of `building_blocks.Lit`.

                            Call
                           /    \
                 Lambda('arg')   Lit(11)
                     |
                   Block('y','z')-------------
                  /                          |
  ['y'=Lit(1),'z'=Lit(2)]                    |
                                           Tuple
                                         /       \
                                   Block('v')     Block('x')-------
                                     / \              |            |
                       ['v'=Selection]  Lit(7)    ['x'=Lit(8)]     |
                             |                                     |
                             |                                     |
                             |                                 Block('w')
                             |                                   /   \
                           Tuple ------              ['w'=Lit(9)]     Lit(10)
                         /              \
                 Block('t')             Block('u')
                  /     \              /          \
        ['t'=L(3)]       Lit(4) ['u'=Lit(5)]       Lit(6)


  Postorder traversals:
  If we are reading Literal values, results of a postorder traversal should be:
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

  If we are reading locals declarations, results of a postorder traversal should
  be:
  [t, u, v, w, x, y, z]

  And if we are reading both in an interleaved fashion, results of a postorder
  traversal should be:
  [1, 2, 3, 4, t, 5, 6, u, 7, v, 8, 9, 10, w, x, y, z, 11]

  Preorder traversals:
  If we are reading Literal values, results of a preorder traversal should be:
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

  If we are reading locals declarations, results of a preorder traversal should
  be:
  [y, z, v, t, u, x, w]

  And if we are reading both in an interleaved fashion, results of a preorder
  traversal should be:
  [y, z, 1, 2, v, t, 3, 4, u, 5, 6, 7, x, 8, w, 9, 10, 11]

  Since we are also exposing the ability to hook into variable declarations,
  it is worthwhile considering the order in which variables are assigned in
  this tree. Notice that this order maps neither to preorder nor to postorder
  when purely considering the nodes of the tree above. This would be:
  [arg, y, z, t, u, v, x, w]

  Returns:
    An instance of `building_blocks.ComputationBuildingBlock`
    satisfying the description above.
  """
  tensor_type = computation_types.TensorType(np.int32)
  lit_c = building_blocks.Literal(3, tensor_type)
  lit_d = building_blocks.Literal(4, tensor_type)
  left_most_leaf = building_blocks.Block([('t', lit_c)], lit_d)

  lit_e = building_blocks.Literal(5, tensor_type)
  lit_f = building_blocks.Literal(6, tensor_type)
  center_leaf = building_blocks.Block([('u', lit_e)], lit_f)
  inner_tuple = building_blocks.Struct([left_most_leaf, center_leaf])

  selected = building_blocks.Selection(inner_tuple, index=0)
  lit_g = building_blocks.Literal(7, tensor_type)
  middle_block = building_blocks.Block([('v', selected)], lit_g)

  lit_i = building_blocks.Literal(8, tensor_type)
  lit_j = building_blocks.Literal(9, tensor_type)
  right_most_endpoint = building_blocks.Block([('w', lit_i)], lit_j)

  lit_h = building_blocks.Literal(10, tensor_type)
  right_child = building_blocks.Block([('x', lit_h)], right_most_endpoint)

  result = building_blocks.Struct([middle_block, right_child])
  lit_a = building_blocks.Literal(1, tensor_type)
  lit_b = building_blocks.Literal(2, tensor_type)
  whimsy_outer_block = building_blocks.Block(
      [('y', lit_a), ('z', lit_b)], result
  )
  whimsy_lambda = building_blocks.Lambda('arg', tensor_type, whimsy_outer_block)
  whimsy_arg = building_blocks.Literal(11, tensor_type)
  called_lambda = building_blocks.Call(whimsy_lambda, whimsy_arg)

  return called_lambda
