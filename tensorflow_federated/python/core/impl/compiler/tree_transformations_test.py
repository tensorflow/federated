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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils


class TransformTestBase(absltest.TestCase):

  def assert_transforms(self, comp, file, changes_type=False, unmodified=False):
    # NOTE: A `transform` method must be present on inheritors.
    after, modified = self.transform(comp)
    golden.check_string(
        file,
        (
            f'Before transformation:\n\n{comp.formatted_representation()}\n\n'
            f'After transformation:\n\n{after.formatted_representation()}'
        ),
    )
    if not changes_type:
      type_test_utils.assert_types_identical(
          comp.type_signature, after.type_signature
      )
    if unmodified:
      self.assertFalse(modified)
    else:
      self.assertTrue(modified)
    return after


def _create_chained_whimsy_federated_maps(functions, arg):
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
    if not fn.parameter_type.is_assignable_from(arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)
          )
      )
    call = building_block_factory.create_federated_map_all_equal(fn, arg)
    arg = call
  return call


class RemoveMappedOrAppliedIdentityTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       building_block_test_utils.create_whimsy_called_federated_map),
      ('federated_map_all_equal',
       intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
       building_block_test_utils.create_whimsy_called_federated_map_all_equal),
  )
  # pyformat: enable
  def test_removes_intrinsic(self, uri, factory):
    call = factory(parameter_name='a')
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        comp.compact_representation(),
        '{}(<(a -> a),federated_value_at_clients(1)>)'.format(uri),
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_value_at_clients(1)',
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_apply(self):
    call = building_block_test_utils.create_whimsy_called_federated_apply(
        parameter_name='a'
    )
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        comp.compact_representation(),
        'federated_apply(<(a -> a),federated_value_at_server(1)>)',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_value_at_server(1)',
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_sequence_map(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array(1, np.int32)
    )
    call = building_block_test_utils.create_whimsy_called_sequence_map(
        parameter_name='a', any_proto=any_proto
    )
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )
    data_str = str(id(any_proto))
    self.assertEqual(
        comp.compact_representation(),
        f'sequence_map(<(a -> a),{data_str}>)',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        data_str,
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_map_with_named_result(self):
    parameter_type = [('a', np.int32), ('b', np.int32)]
    fn = building_block_test_utils.create_identity_function('c', parameter_type)
    arg_type = computation_types.FederatedType(
        parameter_type, placements.CLIENTS
    )
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array(1, np.int32)
    )
    arg = building_blocks.Data(any_proto, arg_type)
    call = building_block_factory.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )
    str_data = str(id(any_proto))
    self.assertEqual(
        comp.compact_representation(), f'federated_map(<(c -> c),{str_data}>)'
    )
    self.assertEqual(transformed_comp.compact_representation(), str_data)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_nested_federated_map(self):
    called_intrinsic = (
        building_block_test_utils.create_whimsy_called_federated_map(
            parameter_name='a'
        )
    )
    block = building_block_test_utils.create_whimsy_block(
        called_intrinsic, variable_name='b'
    )
    comp = block

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        comp.compact_representation(),
        '(let b=1 in federated_map(<(a -> a),federated_value_at_clients(1)>))',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=1 in federated_value_at_clients(1))',
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_chained_federated_maps(self):
    fn = building_block_test_utils.create_identity_function('a', np.int32)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        comp.compact_representation(),
        'federated_map_all_equal(<(a -> a),federated_map_all_equal(<(a ->'
        ' a),federated_value_at_clients(1)>)>)',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_value_at_clients(1)',
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_remove_whimsy_intrinsic(self):
    comp = building_block_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a'
    )

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        transformed_comp.compact_representation(), comp.compact_representation()
    )
    self.assertEqual(transformed_comp.compact_representation(), 'intrinsic(a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_remove_called_lambda(self):
    fn = building_block_test_utils.create_identity_function('a', np.int32)
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    call = building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = (
        tree_transformations.remove_mapped_or_applied_identity(comp)
    )

    self.assertEqual(
        transformed_comp.compact_representation(), comp.compact_representation()
    )
    self.assertEqual(transformed_comp.compact_representation(), '(a -> a)(1)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class RemoveUnusedBlockLocalsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._unused_block_remover = tree_transformations.RemoveUnusedBlockLocals()

  def test_should_transform_block(self):
    blk = building_blocks.Block(
        [(
            'x',
            building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        )],
        building_blocks.Literal(2, computation_types.TensorType(np.int32)),
    )
    self.assertTrue(self._unused_block_remover.should_transform(blk))

  def test_should_not_transform_data(self):
    data = building_blocks.Literal(2, computation_types.TensorType(np.int32))
    self.assertFalse(self._unused_block_remover.should_transform(data))

  def test_removes_block_with_unused_reference(self):
    input_data = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    blk = building_blocks.Block(
        [(
            'x',
            building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        )],
        input_data,
    )
    data, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertTrue(modified)
    self.assertEqual(
        data.compact_representation(), input_data.compact_representation()
    )

  def test_unwraps_block_with_empty_locals(self):
    input_data = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    blk = building_blocks.Block([], input_data)
    data, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertTrue(modified)
    self.assertEqual(
        data.compact_representation(), input_data.compact_representation()
    )

  def test_removes_nested_blocks_with_unused_reference(self):
    input_data = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    blk = building_blocks.Block(
        [(
            'x',
            building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        )],
        input_data,
    )
    higher_level_blk = building_blocks.Block([('y', input_data)], blk)
    data, modified = transformation_utils.transform_postorder(
        higher_level_blk, self._unused_block_remover.transform
    )
    self.assertTrue(modified)
    self.assertEqual(
        data.compact_representation(), input_data.compact_representation()
    )

  def test_leaves_single_used_reference(self):
    blk = building_blocks.Block(
        [(
            'x',
            building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        )],
        building_blocks.Reference('x', np.int32),
    )
    transformed_blk, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertFalse(modified)
    self.assertEqual(
        transformed_blk.compact_representation(), blk.compact_representation()
    )

  def test_leaves_chained_used_references(self):
    blk = building_blocks.Block(
        [
            (
                'x',
                building_blocks.Literal(
                    1, computation_types.TensorType(np.int32)
                ),
            ),
            ('y', building_blocks.Reference('x', np.int32)),
        ],
        building_blocks.Reference('y', np.int32),
    )
    transformed_blk, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertFalse(modified)
    self.assertEqual(
        transformed_blk.compact_representation(), blk.compact_representation()
    )

  def test_removes_locals_referencing_each_other_but_unreferenced_in_result(
      self,
  ):
    input_data = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    blk = building_blocks.Block(
        [
            (
                'x',
                building_blocks.Literal(
                    1, computation_types.TensorType(np.int32)
                ),
            ),
            ('y', building_blocks.Reference('x', np.int32)),
        ],
        input_data,
    )
    transformed_blk, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertTrue(modified)
    self.assertEqual(
        transformed_blk.compact_representation(),
        input_data.compact_representation(),
    )

  def test_leaves_lone_referenced_local(self):
    ref = building_blocks.Reference('y', np.int32)
    blk = building_blocks.Block(
        [
            (
                'x',
                building_blocks.Literal(
                    1, computation_types.TensorType(np.int32)
                ),
            ),
            (
                'y',
                building_blocks.Literal(
                    2, computation_types.TensorType(np.int32)
                ),
            ),
        ],
        ref,
    )
    transformed_blk, modified = transformation_utils.transform_postorder(
        blk, self._unused_block_remover.transform
    )
    self.assertTrue(modified)
    self.assertEqual(transformed_blk.compact_representation(), '(let y=2 in y)')


class UniquifyReferenceNamesTest(TransformTestBase):

  def transform(self, comp):
    return tree_transformations.uniquify_reference_names(comp)

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.uniquify_reference_names(None)

  def test_renames_lambda_but_not_unbound_reference_when_given_name_generator(
      self,
  ):
    ref = building_blocks.Reference('x', np.int32)
    lambda_binding_y = building_blocks.Lambda('y', np.float32, ref)

    name_generator = building_block_factory.unique_name_generator(
        lambda_binding_y
    )
    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        lambda_binding_y, name_generator
    )

    self.assertEqual(lambda_binding_y.compact_representation(), '(y -> x)')
    self.assertEqual(transformed_comp.compact_representation(), '(_var1 -> x)')
    self.assertEqual(
        transformed_comp.type_signature, lambda_binding_y.type_signature
    )
    self.assertTrue(modified)

  def test_single_level_block(self):
    ref = building_blocks.Reference('a', np.int32)
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block((('a', lit), ('a', ref), ('a', ref)), ref)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        block
    )

    self.assertEqual(block.compact_representation(), '(let a=1,a=a,a=a in a)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let a=1,_var1=a,_var2=_var1 in _var2)',
    )
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_nested_blocks(self):
    x_ref = building_blocks.Reference('a', np.int32)
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block1 = building_blocks.Block([('a', lit), ('a', x_ref)], x_ref)
    block2 = building_blocks.Block([('a', lit), ('a', x_ref)], block1)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        block2
    )

    self.assertEqual(
        block2.compact_representation(),
        '(let a=1,a=a in (let a=1,a=a in a))',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let a=1,_var1=a in (let _var2=1,_var3=_var2 in _var3))',
    )
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_nested_lambdas(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    input1 = building_blocks.Reference('a', lit.type_signature)
    first_level_call = building_blocks.Call(
        building_blocks.Lambda('a', input1.type_signature, input1), lit
    )
    input2 = building_blocks.Reference('b', first_level_call.type_signature)
    second_level_call = building_blocks.Call(
        building_blocks.Lambda('b', input2.type_signature, input2),
        first_level_call,
    )

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        second_level_call
    )

    self.assertEqual(
        transformed_comp.compact_representation(), '(b -> b)((a -> a)(1))'
    )
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertFalse(modified)

  def test_block_lambda_block_lambda(self):
    x_ref = building_blocks.Reference('a', np.int32)
    inner_lambda = building_blocks.Lambda('a', np.int32, x_ref)
    called_lambda = building_blocks.Call(inner_lambda, x_ref)
    lower_block = building_blocks.Block(
        [('a', x_ref), ('a', x_ref)], called_lambda
    )
    second_lambda = building_blocks.Lambda('a', np.int32, lower_block)
    second_call = building_blocks.Call(second_lambda, x_ref)
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    last_block = building_blocks.Block([('a', lit), ('a', x_ref)], second_call)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        last_block
    )

    self.assertEqual(
        last_block.compact_representation(),
        '(let a=1,a=a in (a -> (let a=a,a=a in (a -> a)(a)))(a))',
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        (
            '(let a=1,_var1=a in (_var2 -> (let _var3=_var2,_var4=_var3 in'
            ' (_var5 -> _var5)(_var4)))(_var1))'
        ),
    )
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_blocks_nested_inside_of_locals(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    lower_block = building_blocks.Block([('a', lit)], lit)
    middle_block = building_blocks.Block([('a', lower_block)], lit)
    higher_block = building_blocks.Block([('a', middle_block)], lit)
    y_ref = building_blocks.Reference('a', np.int32)
    lower_block_with_y_ref = building_blocks.Block([('a', y_ref)], lit)
    middle_block_with_y_ref = building_blocks.Block(
        [('a', lower_block_with_y_ref)], lit
    )
    higher_block_with_y_ref = building_blocks.Block(
        [('a', middle_block_with_y_ref)], lit
    )
    multiple_bindings_highest_block = building_blocks.Block(
        [('a', higher_block), ('a', higher_block_with_y_ref)],
        higher_block_with_y_ref,
    )

    transformed_comp = self.assert_transforms(
        multiple_bindings_highest_block,
        'uniquify_names_blocks_nested_inside_of_locals.expected',
    )
    tree_analysis.check_has_unique_names(transformed_comp)

  def test_keeps_existing_nonoverlapping_names(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block([('a', lit), ('b', lit)], lit)
    comp = block

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        comp
    )

    self.assertEqual(block.compact_representation(), '(let a=1,b=1 in 1)')
    self.assertEqual(
        transformed_comp.compact_representation(), '(let a=1,b=1 in 1)'
    )
    self.assertFalse(modified)


class NormalizeTypesTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.normalize_types(None)

  def test_ignore_unnormalized_all_equal(self):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    unnormalized_comp = tree_transformations.normalize_types(
        building_blocks.Reference('x', fed_type_all_equal),
        normalize_all_equal_bit=False,
    )
    self.assertEqual(unnormalized_comp.type_signature, fed_type_all_equal)
    self.assertIsInstance(unnormalized_comp, building_blocks.Reference)
    self.assertEqual(str(unnormalized_comp), 'x')

  def test_converts_all_equal_at_clients_reference_to_not_equal(self):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    normalized_comp = tree_transformations.normalize_types(
        building_blocks.Reference('x', fed_type_all_equal)
    )
    self.assertEqual(
        normalized_comp.type_signature,
        computation_types.FederatedType(
            np.int32, placements.CLIENTS, all_equal=False
        ),
    )
    self.assertIsInstance(normalized_comp, building_blocks.Reference)
    self.assertEqual(str(normalized_comp), 'x')

  def test_converts_not_all_equal_at_server_reference_to_equal(self):
    fed_type_not_all_equal = computation_types.FederatedType(
        np.int32, placements.SERVER, all_equal=False
    )
    normalized_comp = tree_transformations.normalize_types(
        building_blocks.Reference('x', fed_type_not_all_equal)
    )
    self.assertEqual(
        normalized_comp.type_signature,
        computation_types.FederatedType(
            np.int32, placements.SERVER, all_equal=True
        ),
    )
    self.assertIsInstance(normalized_comp, building_blocks.Reference)
    self.assertEqual(str(normalized_comp), 'x')

  def test_converts_all_equal_at_clients_lambda_parameter_to_not_equal(self):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    normalized_fed_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    ref = building_blocks.Reference('x', fed_type_all_equal)
    lam = building_blocks.Lambda('x', fed_type_all_equal, ref)
    normalized_lambda = tree_transformations.normalize_types(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(fed_type_all_equal, fed_type_all_equal),
    )
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(
            normalized_fed_type, normalized_fed_type
        ),
    )

  def test_converts_all_equal_at_clients_lambda_struct_parameter_to_not_equal(
      self,
  ):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    normalized_fed_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    lam = building_blocks.Lambda(
        'x',
        computation_types.StructType([fed_type_all_equal, fed_type_all_equal]),
        building_blocks.Reference(
            'x',
            computation_types.StructType(
                [fed_type_all_equal, fed_type_all_equal]
            ),
        ),
    )
    normalized_lambda = tree_transformations.normalize_types(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(
            computation_types.StructType(
                [fed_type_all_equal, fed_type_all_equal]
            ),
            computation_types.StructType(
                [fed_type_all_equal, fed_type_all_equal]
            ),
        ),
    )
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(
            computation_types.StructType(
                [normalized_fed_type, normalized_fed_type]
            ),
            computation_types.StructType(
                [normalized_fed_type, normalized_fed_type]
            ),
        ),
    )

  def test_converts_all_equal_at_clients_lambda_nested_struct_parameter_to_not_equal(
      self,
  ):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    normalized_fed_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    lam = building_blocks.Lambda(
        'x',
        computation_types.StructType([
            fed_type_all_equal,
            computation_types.StructType([fed_type_all_equal]),
        ]),
        building_blocks.Reference(
            'x',
            computation_types.StructType([
                fed_type_all_equal,
                computation_types.StructType([fed_type_all_equal]),
            ]),
        ),
    )
    normalized_lambda = tree_transformations.normalize_types(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(
            computation_types.StructType([
                fed_type_all_equal,
                computation_types.StructType([fed_type_all_equal]),
            ]),
            computation_types.StructType([
                fed_type_all_equal,
                computation_types.StructType([fed_type_all_equal]),
            ]),
        ),
    )
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(
            computation_types.StructType([
                normalized_fed_type,
                computation_types.StructType([normalized_fed_type]),
            ]),
            computation_types.StructType([
                normalized_fed_type,
                computation_types.StructType([normalized_fed_type]),
            ]),
        ),
    )

  def test_converts_not_all_equal_at_server_lambda_parameter_to_equal(self):
    fed_type_not_all_equal = computation_types.FederatedType(
        np.int32, placements.SERVER, all_equal=False
    )
    normalized_fed_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    ref = building_blocks.Reference('x', fed_type_not_all_equal)
    lam = building_blocks.Lambda('x', fed_type_not_all_equal, ref)
    normalized_lambda = tree_transformations.normalize_types(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(
            fed_type_not_all_equal, fed_type_not_all_equal
        ),
    )
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(
            normalized_fed_type, normalized_fed_type
        ),
    )

  def test_converts_federated_map_all_equal_to_federated_map(self):
    fed_type_all_equal = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    normalized_fed_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    int_ref = building_blocks.Reference('x', np.int32)
    int_identity = building_blocks.Lambda('x', np.int32, int_ref)
    federated_int_ref = building_blocks.Reference('y', fed_type_all_equal)
    called_federated_map_all_equal = (
        building_block_factory.create_federated_map_all_equal(
            int_identity, federated_int_ref
        )
    )
    normalized_federated_map = tree_transformations.normalize_types(
        called_federated_map_all_equal
    )
    self.assertEqual(
        called_federated_map_all_equal.function.uri,
        intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
    )
    self.assertIsInstance(normalized_federated_map, building_blocks.Call)
    self.assertIsInstance(
        normalized_federated_map.function, building_blocks.Intrinsic
    )
    self.assertEqual(
        normalized_federated_map.function.uri, intrinsic_defs.FEDERATED_MAP.uri
    )
    self.assertEqual(
        normalized_federated_map.type_signature, normalized_fed_type
    )


class ReplaceSelectionsTest(absltest.TestCase):

  def test_replace_selection(self):
    comp = building_blocks.Selection(
        building_blocks.Reference('x', [np.int32, np.int32]), index=1
    )
    y = building_blocks.Reference('y', np.int32)
    path_to_replacement = {
        (1,): y,
    }
    new_comp = tree_transformations.replace_selections(
        comp, 'x', path_to_replacement
    )
    self.assertEqual(new_comp.proto, y.proto)

  def test_replace_multiple_instances_of_selection(self):
    comp = building_blocks.Struct([
        building_blocks.Selection(
            building_blocks.Reference('x', [np.int32, [np.int32]]), index=1
        ),
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', [np.int32, [np.int32]]),
                index=1,
            ),
            index=0,
        ),
    ])
    y = building_blocks.Reference('y', [np.int32])
    path_to_replacement = {
        (1,): y,
    }
    new_comp = tree_transformations.replace_selections(
        comp, 'x', path_to_replacement
    )
    self.assertEqual(
        new_comp.proto,
        building_blocks.Struct(
            [y, building_blocks.Selection(y, index=0)]
        ).proto,
    )

  def test_replace_selection_mismatching_ref_name(self):
    comp = building_blocks.Selection(
        building_blocks.Reference('x', [np.int32, np.int32]), index=1
    )
    y = building_blocks.Reference('y', np.int32)
    path_to_replacement = {
        (1,): y,
    }
    new_comp = tree_transformations.replace_selections(
        comp, 'z', path_to_replacement
    )
    self.assertEqual(new_comp.proto, comp.proto)

  def test_fail_replace_compiled_comp(self):
    arg_type = computation_types.StructType([np.int32])
    fn_type = computation_types.FunctionType(arg_type, arg_type)
    mock_fn = mock.create_autospec(
        building_blocks.CompiledComputation, spec_set=True, instance=True
    )
    type(mock_fn).type_signature = mock.PropertyMock(
        spec=computation_types.FunctionType, return_value=fn_type, spec_set=True
    )
    comp = building_blocks.Call(
        mock_fn,
        building_blocks.Reference('x', arg_type),
    )
    y = building_blocks.Reference('y', np.int32)
    path_to_replacement = {
        (0,): y,
    }
    with self.assertRaisesRegex(ValueError, 'Encountered called graph'):
      tree_transformations.replace_selections(comp, 'x', path_to_replacement)

  def test_no_subsequent_replacement(self):
    comp = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', [[np.int32]]), index=0
        ),
        index=0,
    )
    replacement = building_blocks.Reference('x', [np.int32])
    path_to_replacement = {
        (0,): replacement,
    }
    new_comp = tree_transformations.replace_selections(
        comp, 'x', path_to_replacement
    )
    # The inner x[0] portion of x[0][0] should be replaced by x, yielding x[0].
    # This resulting x[0] should not in turn be replaced by x because the
    # type signatures would not be accurate.
    self.assertEqual(
        new_comp.proto,
        building_blocks.Selection(
            building_blocks.Reference('x', [np.int32]), index=0
        ).proto,
    )


class AsFunctionOfSomeParametersTest(absltest.TestCase):

  def test_empty_path(self):
    comp = building_blocks.Lambda(
        'x', np.int32, building_blocks.Reference('x', np.int32)
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(comp, [])
    self.assertEqual(new_comp.parameter_type, computation_types.StructType([]))
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEqual(unbound_references, set(['x']))

  def test_all_path(self):
    comp = building_blocks.Lambda(
        'x', np.int32, building_blocks.Reference('x', np.int32)
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [()]
    )
    self.assertEqual(
        new_comp.parameter_type, computation_types.StructType([np.int32])
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)

  def test_selection_path(self):
    arg_type = [[np.int32]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', arg_type), index=0
            ),
            index=0,
        ),
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [(0, 0)]
    )
    self.assertEqual(
        new_comp.parameter_type, computation_types.StructType([np.int32])
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)

  def test_partial_selection_path(self):
    arg_type = [[np.int32]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Selection(
            building_blocks.Reference('x', arg_type), index=0
        ),
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [(0,)]
    )
    self.assertEqual(
        new_comp.parameter_type, computation_types.StructType([[np.int32]])
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)

  def test_invalid_selection_path(self):
    arg_type = [[np.int32]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', arg_type), index=0
            ),
            index=0,
        ),
    )
    with self.assertRaises(tree_transformations.ParameterSelectionError):
      tree_transformations.as_function_of_some_subparameters(comp, [(0, 1)])

  def test_multiple_selection_path(self):
    arg_type = [np.int32, np.float32, [np.int32, np.str_]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Struct([
            building_blocks.Selection(
                building_blocks.Reference('x', arg_type), index=1
            ),
            building_blocks.Selection(
                building_blocks.Selection(
                    building_blocks.Reference('x', arg_type), index=2
                ),
                index=0,
            ),
            building_blocks.Selection(
                building_blocks.Reference('x', arg_type), index=2
            ),
        ]),
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [(1,), (2,)]
    )
    self.assertEqual(
        new_comp.parameter_type,
        computation_types.StructType([np.float32, [np.int32, np.str_]]),
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)

  def test_unused_selection_path(self):
    arg_type = [np.int32, np.float32, [np.int32, np.str_]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Selection(
            building_blocks.Reference('x', arg_type), index=1
        ),
    )
    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [(1,), (2,)]
    )
    self.assertEqual(
        new_comp.parameter_type,
        computation_types.StructType([np.float32, [np.int32, np.str_]]),
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)

  def test_paths_not_applied_sequentially(self):
    arg_type = [np.int32, np.float32, [np.int32, np.str_]]
    comp = building_blocks.Lambda(
        'x',
        arg_type,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', arg_type), index=2
            ),
            index=1,
        ),
    )

    new_comp = tree_transformations.as_function_of_some_subparameters(
        comp, [(2,), (1,)]
    )
    self.assertEqual(
        new_comp.parameter_type,
        computation_types.StructType([[np.int32, np.str_], np.float32]),
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_comp
    )[new_comp]
    self.assertEmpty(unbound_references)
    self.assertIsInstance(new_comp.result.result, building_blocks.Selection)
    self.assertIsInstance(
        new_comp.result.result.source, building_blocks.Selection
    )


class StripPlacementTest(parameterized.TestCase):

  def assert_has_no_intrinsics_nor_federated_types(self, comp):
    def _check(x):
      if isinstance(x.type_signature, computation_types.FederatedType):
        raise AssertionError(f'Unexpected federated type: {x.type_signature}')
      if isinstance(x, building_blocks.Intrinsic):
        raise AssertionError(f'Unexpected intrinsic: {x}')

    tree_analysis.visit_postorder(comp, _check)

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.strip_placement(None)

  def test_computation_non_federated_type(self):
    before = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    after, modified = tree_transformations.strip_placement(before)
    self.assertEqual(before, after)
    self.assertFalse(modified)

  def test_raises_disallowed_intrinsic(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(np.int32, placements.SERVER)
    )
    broadcaster = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_BROADCAST.uri,
        computation_types.FunctionType(
            fed_ref.type_signature,
            computation_types.FederatedType(
                fed_ref.type_signature.member,
                placements.CLIENTS,
                all_equal=True,
            ),
        ),
    )
    called_broadcast = building_blocks.Call(broadcaster, fed_ref)
    with self.assertRaises(ValueError):
      tree_transformations.strip_placement(called_broadcast)

  def test_raises_multiple_placements(self):
    server_placed_data = building_blocks.Reference(
        'x', computation_types.FederatedType(np.int32, placements.SERVER)
    )
    clients_placed_data = building_blocks.Reference(
        'y', computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    block_holding_both = building_blocks.Block(
        [('x', server_placed_data)], clients_placed_data
    )
    with self.assertRaisesRegex(ValueError, 'multiple different placements'):
      tree_transformations.strip_placement(block_holding_both)

  def test_passes_unbound_type_signature_obscured_under_block(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(np.int32, placements.SERVER)
    )
    block = building_blocks.Block(
        [
            ('y', fed_ref),
            (
                'x',
                building_blocks.Literal(
                    1, computation_types.TensorType(np.int32)
                ),
            ),
            ('z', building_blocks.Reference('x', np.int32)),
        ],
        building_blocks.Reference('y', fed_ref.type_signature),
    )
    tree_transformations.strip_placement(block)

  def test_passes_noarg_lambda(self):
    lam = building_blocks.Lambda(
        None,
        None,
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
    )
    fed_int_type = computation_types.FederatedType(np.int32, placements.SERVER)
    fed_eval = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
        computation_types.FunctionType(lam.type_signature, fed_int_type),
    )
    called_eval = building_blocks.Call(fed_eval, lam)
    tree_transformations.strip_placement(called_eval)

  def test_removes_federated_types_under_function(self):
    int_type = np.int32
    server_int_type = computation_types.FederatedType(
        int_type, placements.SERVER
    )
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', server_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref
    )
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id
    )
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)

  def test_strip_placement_removes_federated_applys(self):
    int_type = computation_types.TensorType(np.int32)
    server_int_type = computation_types.FederatedType(
        int_type, placements.SERVER
    )
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', server_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref
    )
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id
    )
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, server_int_type
    )
    type_test_utils.assert_types_identical(after.type_signature, int_type)
    self.assertEqual(
        before.compact_representation(),
        'federated_apply(<(x -> x),federated_apply(<(x -> x),x>)>)',
    )
    self.assertEqual(after.compact_representation(), '(x -> x)((x -> x)(x))')

  def test_strip_placement_removes_federated_maps(self):
    int_type = computation_types.TensorType(np.int32)
    clients_int_type = computation_types.FederatedType(
        int_type, placements.CLIENTS
    )
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', clients_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref
    )
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id
    )
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, clients_int_type
    )
    type_test_utils.assert_types_identical(after.type_signature, int_type)
    self.assertEqual(
        before.compact_representation(),
        'federated_map(<(x -> x),federated_map(<(x -> x),x>)>)',
    )
    self.assertEqual(after.compact_representation(), '(x -> x)((x -> x)(x))')

  def test_unwrap_removes_federated_zips_at_server(self):
    list_type = computation_types.to_type([np.int32, np.float32] * 2)
    server_list_type = computation_types.FederatedType(
        list_type, placements.SERVER
    )
    fed_tuple = building_blocks.Reference('tup', server_list_type)
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    before = building_block_factory.create_federated_zip(unzipped)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, server_list_type
    )
    type_test_utils.assert_types_identical(after.type_signature, list_type)

  def test_unwrap_removes_federated_zips_at_clients(self):
    list_type = computation_types.to_type([np.int32, np.float32] * 2)
    clients_list_type = computation_types.FederatedType(
        list_type, placements.SERVER
    )
    fed_tuple = building_blocks.Reference('tup', clients_list_type)
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    before = building_block_factory.create_federated_zip(unzipped)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, clients_list_type
    )
    type_test_utils.assert_types_identical(after.type_signature, list_type)

  def test_strip_placement_removes_federated_value_at_server(self):
    int_data = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    float_data = building_blocks.Literal(
        2.0, computation_types.TensorType(np.float32)
    )
    fed_int = building_block_factory.create_federated_value(
        int_data, placements.SERVER
    )
    fed_float = building_block_factory.create_federated_value(
        float_data, placements.SERVER
    )
    tup = building_blocks.Struct([fed_int, fed_float], container_type=tuple)
    before = building_block_factory.create_federated_zip(tup)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    tuple_type = computation_types.StructWithPythonType(
        [(None, np.int32), (None, np.float32)], tuple
    )
    type_test_utils.assert_types_identical(
        before.type_signature,
        computation_types.FederatedType(tuple_type, placements.SERVER),
    )
    type_test_utils.assert_types_identical(after.type_signature, tuple_type)

  def test_strip_placement_federated_value_at_clients(self):
    int_data = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    float_data = building_blocks.Literal(
        2.0, computation_types.TensorType(np.float32)
    )
    fed_int = building_block_factory.create_federated_value(
        int_data, placements.CLIENTS
    )
    fed_float = building_block_factory.create_federated_value(
        float_data, placements.CLIENTS
    )
    tup = building_blocks.Struct([fed_int, fed_float], container_type=tuple)
    before = building_block_factory.create_federated_zip(tup)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    tuple_type = computation_types.StructWithPythonType(
        [(None, np.int32), (None, np.float32)], tuple
    )
    type_test_utils.assert_types_identical(
        before.type_signature,
        computation_types.FederatedType(tuple_type, placements.CLIENTS),
    )
    type_test_utils.assert_types_identical(after.type_signature, tuple_type)

  def test_strip_placement_with_called_lambda(self):
    int_type = computation_types.TensorType(np.int32)
    server_int_type = computation_types.FederatedType(
        int_type, placements.SERVER
    )
    federated_ref = building_blocks.Reference('outer', server_int_type)
    inner_federated_ref = building_blocks.Reference('inner', server_int_type)
    identity_lambda = building_blocks.Lambda(
        'inner', server_int_type, inner_federated_ref
    )
    before = building_blocks.Call(identity_lambda, federated_ref)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, server_int_type
    )
    type_test_utils.assert_types_identical(after.type_signature, int_type)

  def test_strip_placement_nested_federated_type(self):
    int_type = computation_types.TensorType(np.int32)
    server_int_type = computation_types.FederatedType(
        int_type, placements.SERVER
    )
    tupled_int_type = computation_types.to_type((int_type, int_type))
    tupled_server_int_type = computation_types.to_type(
        (server_int_type, server_int_type)
    )
    fed_ref = building_blocks.Reference('x', server_int_type)
    before = building_blocks.Struct([fed_ref, fed_ref], container_type=tuple)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    type_test_utils.assert_types_identical(
        before.type_signature, tupled_server_int_type
    )
    type_test_utils.assert_types_identical(
        after.type_signature, tupled_int_type
    )


if __name__ == '__main__':
  absltest.main()
