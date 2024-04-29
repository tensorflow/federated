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

from unittest import mock

from absl.testing import absltest
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_test_utils


class ToCallDominantTest(absltest.TestCase):

  def assert_compact_representations_equal(
      self,
      actual: building_blocks.ComputationBuildingBlock,
      expected: building_blocks.ComputationBuildingBlock,
  ):
    """Asserts that two building blocks have the same compact representation."""
    self.assertEqual(
        actual.compact_representation(), expected.compact_representation()
    )

  def test_inlines_references(self):
    int_type = computation_types.to_type(np.int32)
    int_ref = lambda name: building_blocks.Reference(name, int_type)
    int_fn = lambda name, result: building_blocks.Lambda(name, int_type, result)
    before = int_fn(
        'x',
        building_blocks.Block(
            [
                ('y', int_ref('x')),
                ('z', int_ref('y')),
            ],
            int_ref('z'),
        ),
    )
    after = transformations.to_call_dominant(before)
    expected = int_fn('x', int_ref('x'))
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_selections(self):
    int_type = computation_types.to_type(np.int32)
    structed = computation_types.StructType([int_type])
    double = computation_types.StructType([structed])
    bb = building_blocks
    before = bb.Lambda(
        'x',
        double,
        bb.Block(
            [
                ('y', bb.Selection(bb.Reference('x', double), index=0)),
                ('z', bb.Selection(bb.Reference('y', structed), index=0)),
            ],
            bb.Reference('z', int_type),
        ),
    )
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x',
        double,
        bb.Selection(bb.Selection(bb.Reference('x', double), index=0), index=0),
    )
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_structs(self):
    int_type = computation_types.to_type(np.int32)
    structed = computation_types.StructType([int_type])
    double = computation_types.StructType([structed])
    bb = building_blocks
    before = bb.Lambda(
        'x',
        int_type,
        bb.Block(
            [
                ('y', bb.Struct([building_blocks.Reference('x', int_type)])),
                ('z', bb.Struct([building_blocks.Reference('y', structed)])),
            ],
            bb.Reference('z', double),
        ),
    )
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x', int_type, bb.Struct([bb.Struct([bb.Reference('x', int_type)])])
    )
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_selection_from_struct(self):
    int_type = computation_types.to_type(np.int32)
    bb = building_blocks
    before = bb.Lambda(
        'x',
        int_type,
        bb.Selection(bb.Struct([bb.Reference('x', int_type)]), index=0),
    )
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda('x', int_type, bb.Reference('x', int_type))
    self.assert_compact_representations_equal(after, expected)

  def test_creates_binding_for_each_call(self):
    int_type = computation_types.to_type(np.int32)
    int_to_int_type = computation_types.FunctionType(int_type, int_type)
    bb = building_blocks
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3])
    )
    int_to_int_fn = bb.Data(any_proto, int_to_int_type)
    before = bb.Lambda(
        'x',
        int_type,
        bb.Call(
            int_to_int_fn, bb.Call(int_to_int_fn, bb.Reference('x', int_type))
        ),
    )
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x',
        int_type,
        bb.Block(
            [
                ('_var1', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
                (
                    '_var2',
                    bb.Call(int_to_int_fn, bb.Reference('_var1', int_type)),
                ),
            ],
            bb.Reference('_var2', int_type),
        ),
    )
    self.assert_compact_representations_equal(after, expected)

  def test_evaluates_called_lambdas(self):
    int_type = computation_types.to_type(np.int32)
    int_to_int_type = computation_types.FunctionType(int_type, int_type)
    int_thunk_type = computation_types.FunctionType(None, int_type)
    bb = building_blocks
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3])
    )
    int_to_int_fn = bb.Data(any_proto, int_to_int_type)

    # -> (let result = ext(x) in (-> result))
    # Each call of the outer lambda should create a single binding, with
    # calls to the inner lambda repeatedly returning references to the binding.
    higher_fn = bb.Lambda(
        None,
        None,
        bb.Block(
            [('result', bb.Call(int_to_int_fn, bb.Reference('x', int_type)))],
            bb.Lambda(None, None, bb.Reference('result', int_type)),
        ),
    )
    block_locals = [
        ('fn', higher_fn),
        # fn = -> (let result = ext(x) in (-> result))
        ('get_val1', bb.Call(bb.Reference('fn', higher_fn.type_signature))),
        # _var2 = ext(x)
        # get_val1 = -> _var2
        ('get_val2', bb.Call(bb.Reference('fn', higher_fn.type_signature))),
        # _var3 = ext(x)
        # get_val2 = -> _var3
        ('val11', bb.Call(bb.Reference('get_val1', int_thunk_type))),
        # val11 = _var2
        ('val12', bb.Call(bb.Reference('get_val1', int_thunk_type))),
        # val12 = _var2
        ('val2', bb.Call(bb.Reference('get_val2', int_thunk_type))),
        # val2 = _var3
    ]
    before = bb.Lambda(
        'x',
        int_type,
        bb.Block(
            block_locals,
            # <_var2, _var2, _var3>
            bb.Struct([
                bb.Reference('val11', int_type),
                bb.Reference('val12', int_type),
                bb.Reference('val2', int_type),
            ]),
        ),
    )
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x',
        int_type,
        bb.Block(
            [
                ('_var2', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
                ('_var3', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
            ],
            bb.Struct([
                bb.Reference('_var2', int_type),
                bb.Reference('_var2', int_type),
                bb.Reference('_var3', int_type),
            ]),
        ),
    )
    self.assert_compact_representations_equal(after, expected)

  def test_creates_block_for_non_lambda(self):
    bb = building_blocks
    int_type = computation_types.TensorType(np.int32)
    two_int_type = computation_types.StructType(
        [(None, int_type), (None, int_type)]
    )
    get_two_int_type = computation_types.FunctionType(None, two_int_type)
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3])
    )
    call_ext = bb.Call(bb.Data(any_proto, get_two_int_type))
    before = bb.Selection(call_ext, index=0)
    after = transformations.to_call_dominant(before)
    expected = bb.Block(
        [
            ('_var1', call_ext),
        ],
        bb.Selection(bb.Reference('_var1', two_int_type), index=0),
    )
    self.assert_compact_representations_equal(after, expected)

  def test_call_to_higher_order_external_allowed(self):
    int_type = computation_types.TensorType(np.int32)
    int_to_int_type = computation_types.FunctionType(int_type, int_type)
    int_to_int_to_int_type = computation_types.FunctionType(
        int_to_int_type, int_type
    )
    call_ext = building_blocks.Call(
        building_blocks.Reference('call_with_one', int_to_int_to_int_type),
        building_blocks.Lambda(
            'x', int_type, building_blocks.Reference('num', int_type)
        ),
    )
    after = transformations.to_call_dominant(call_ext)
    self.assertIsInstance(after, building_blocks.Block)
    self.assertLen(after.locals, 1)
    (ref_name, bound_call) = after.locals[0]
    self.assertEqual(
        bound_call.compact_representation(), call_ext.compact_representation()
    )
    expected_result = building_blocks.Reference(
        ref_name, call_ext.type_signature
    )
    self.assert_compact_representations_equal(after.result, expected_result)


class ForceAlignAndSplitByIntrinsicTest(absltest.TestCase):

  def assert_splits_on(self, comp, calls):
    """Asserts that `force_align_and_split_by_intrinsics` removes intrinsics."""
    if not isinstance(calls, list):
      calls = [calls]
    uris = [call.function.uri for call in calls]
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, calls
    )

    # Ensure that the resulting computations no longer contain the split
    # intrinsics.
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uris))
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uris))
    # Removal isn't interesting to test for if it wasn't there to begin with.
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp, uris))

    if comp.parameter_type is not None:
      type_test_utils.assert_types_equivalent(
          comp.parameter_type, before.parameter_type
      )
    else:
      self.assertIsNone(before.parameter_type)
    # There must be one parameter for each intrinsic in `calls`.
    self.assertIsInstance(
        before.type_signature.result, computation_types.StructType
    )
    self.assertLen(before.type_signature.result, len(calls))

    # Check that `after`'s parameter is a structure like:
    # {
    #   'original_arg': comp.parameter_type, (if present)
    #   'intrinsic_results': [...],
    # }
    self.assertIsInstance(after.parameter_type, computation_types.StructType)
    if comp.parameter_type is not None:
      self.assertLen(after.parameter_type, 2)
      type_test_utils.assert_types_equivalent(
          comp.parameter_type, after.parameter_type.original_arg
      )
    else:
      self.assertLen(after.parameter_type, 1)
    # There must be one result for each intrinsic in `calls`.
    self.assertLen(after.parameter_type.intrinsic_results, len(calls))

    # Check that each pair of (param, result) is a valid type substitution
    # for the intrinsic in question.
    for i in range(len(calls)):
      concrete_signature = computation_types.FunctionType(
          before.type_signature.result[i],
          after.parameter_type.intrinsic_results[i],
      )
      abstract_signature = calls[i].function.intrinsic_def().type_signature
      type_analysis.check_concrete_instance_of(
          concrete_signature, abstract_signature
      )

  def test_cannot_split_on_chained_intrinsic(self):
    int_type = computation_types.TensorType(np.int32)
    client_int_type = computation_types.FederatedType(
        int_type, placements.CLIENTS
    )
    int_ref = lambda name: building_blocks.Reference(name, int_type)

    def client_int_ref(name):
      return building_blocks.Reference(name, client_int_type)

    body = building_blocks.Block(
        [
            (
                'a',
                building_block_factory.create_federated_map(
                    building_blocks.Lambda('p1', int_type, int_ref('p1')),
                    client_int_ref('param'),
                ),
            ),
            (
                'b',
                building_block_factory.create_federated_map(
                    building_blocks.Lambda('p2', int_type, int_ref('p2')),
                    client_int_ref('a'),
                ),
            ),
        ],
        client_int_ref('b'),
    )
    comp = building_blocks.Lambda('param', client_int_type, body)
    intrinsic_defaults = [
        building_block_test_utils.create_whimsy_called_federated_map('test'),
    ]
    with self.assertRaises(transformations.NonAlignableAlongIntrinsicError):
      transformations.force_align_and_split_by_intrinsics(
          comp, intrinsic_defaults
      )

  def test_splits_on_intrinsic_noarg_function(self):
    federated_broadcast = (
        building_block_test_utils.create_whimsy_called_federated_broadcast()
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda(None, None, called_intrinsics)
    call = building_block_test_utils.create_whimsy_called_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_broadcast(self):
    federated_broadcast = (
        building_block_test_utils.create_whimsy_called_federated_broadcast()
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda('a', np.int32, called_intrinsics)
    call = building_block_test_utils.create_whimsy_called_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_nested_in_tuple_broadcast(self):
    first_broadcast = (
        building_block_test_utils.create_whimsy_called_federated_broadcast()
    )
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3])
    )
    packed_broadcast = building_blocks.Struct([
        building_blocks.Data(
            any_proto,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ),
        first_broadcast,
    ])
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    result = transformations.to_call_dominant(second_broadcast)
    comp = building_blocks.Lambda('a', np.int32, result)
    call = building_block_test_utils.create_whimsy_called_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_broadcast(self):
    federated_broadcast = (
        building_block_test_utils.create_whimsy_called_federated_broadcast()
    )
    called_intrinsics = building_blocks.Struct([
        federated_broadcast,
        federated_broadcast,
    ])
    comp = building_blocks.Lambda('a', np.int32, called_intrinsics)
    call = building_block_test_utils.create_whimsy_called_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_aggregate(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    call = building_block_test_utils.create_whimsy_called_federated_aggregate(
        value_type=np.int32
    )
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_aggregate(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    call = building_block_test_utils.create_whimsy_called_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_secure_sum_bitwidth(self):
    federated_secure_sum_bitwidth = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    called_intrinsics = building_blocks.Struct([federated_secure_sum_bitwidth])
    comp = building_blocks.Lambda('a', np.int32, called_intrinsics)
    call = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_secure_sum_bitwidths(self):
    federated_secure_sum_bitwidth = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('a', np.int32, called_intrinsics)
    call = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    self.assert_splits_on(comp, call)

  def test_removes_selected_intrinsic_leaving_remaining_intrinsic(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    federated_secure_sum_bitwidth = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    null_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate()
    )
    secure_sum_bitwidth_uri = federated_secure_sum_bitwidth.function.uri
    aggregate_uri = null_aggregate.function.uri
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, [null_aggregate]
    )
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, secure_sum_bitwidth_uri)
    )
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, aggregate_uri)
    )
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(before, aggregate_uri)
    )
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(after, aggregate_uri)
    )
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(before, secure_sum_bitwidth_uri)
        or tree_analysis.contains_called_intrinsic(
            after, secure_sum_bitwidth_uri
        )
    )

  def test_splits_on_two_intrinsics(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    federated_secure_sum_bitwidth = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    self.assert_splits_on(
        comp,
        [
            building_block_test_utils.create_whimsy_called_federated_aggregate(),
            building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(),
        ],
    )

  def test_splits_on_multiple_instances_of_two_intrinsics(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    federated_secure_sum_bitwidth = (
        building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth()
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    self.assert_splits_on(
        comp,
        [
            building_block_test_utils.create_whimsy_called_federated_aggregate(),
            building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(),
        ],
    )

  def test_splits_even_when_selected_intrinsic_is_not_present(self):
    federated_aggregate = (
        building_block_test_utils.create_whimsy_called_federated_aggregate(
            accumulate_parameter_name='a',
            merge_parameter_name='b',
            report_parameter_name='c',
        )
    )
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', np.int32, called_intrinsics)
    transformations.force_align_and_split_by_intrinsics(
        comp,
        [
            building_block_test_utils.create_whimsy_called_federated_aggregate(),
            building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(),
        ],
    )


class AugmentLambdaWithParameterForUnboundReferences(absltest.TestCase):

  def _check_transformed_comp_validity(
      self,
      original_comp: building_blocks.Lambda,
      transformed_comp: building_blocks.ComputationBuildingBlock,
      lambda_parameter_extension_name: str,
  ):
    self.assertIsInstance(transformed_comp, building_blocks.Lambda)

    # The transformed lambda comp should have an additional element in the input
    # parameter named `lambda_parameter_extension_name`.
    self.assertLen(
        transformed_comp.parameter_type, len(original_comp.parameter_type) + 1
    )
    self.assertEqual(
        structure.to_elements(transformed_comp.parameter_type)[
            len(transformed_comp.parameter_type) - 1
        ][0],
        lambda_parameter_extension_name,
    )

    # The transformed lambda comp should have no unbound references.
    self.assertEmpty(
        transformation_utils.get_map_of_unbound_references(transformed_comp)[
            transformed_comp
        ]
    )

  def test_identifies_unbound_refs(self):
    original_arg_type = computation_types.StructType([np.int32])
    int_at_clients_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    comp = building_blocks.Lambda(
        'arg',
        original_arg_type,
        building_blocks.Block(
            [(
                'a',
                building_block_factory.create_federated_sum(
                    building_blocks.Reference('x', int_at_clients_type)
                ),
            )],
            building_blocks.Struct([
                building_blocks.Reference(
                    'a',
                    computation_types.FederatedType(
                        np.int32, placements.SERVER
                    ),
                ),
                building_blocks.Reference('y', int_at_clients_type),
                building_blocks.Reference('x', int_at_clients_type),
            ]),
        ),
    )
    lambda_parameter_extension_name = 'intermediate_state'
    transformed_comp, new_input_comps = (
        transformations._augment_lambda_with_parameter_for_unbound_references(
            comp, lambda_parameter_extension_name
        )
    )

    self._check_transformed_comp_validity(
        comp, transformed_comp, lambda_parameter_extension_name
    )
    self.assertLen(new_input_comps, 3)
    for new_input_comp, expected_new_input_comp in zip(
        new_input_comps,
        [
            building_blocks.Reference('x', int_at_clients_type),
            building_blocks.Reference('y', int_at_clients_type),
            building_blocks.Reference('x', int_at_clients_type),
        ],
    ):
      self.assertEqual(new_input_comp.proto, expected_new_input_comp.proto)

  def test_identifies_unbound_selections(self):
    original_arg_type = computation_types.StructType([np.int32])
    int_at_clients_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    federated_sum_param = building_blocks.Selection(
        building_blocks.Reference('x', [int_at_clients_type]), index=0
    )
    other_result_param = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference(
                'y', [[int_at_clients_type, int_at_clients_type]]
            ),
            index=0,
        ),
        index=1,
    )
    comp = building_blocks.Lambda(
        'arg',
        original_arg_type,
        building_blocks.Block(
            [(
                'a',
                building_block_factory.create_federated_sum(
                    federated_sum_param
                ),
            )],
            building_blocks.Struct([
                building_blocks.Reference(
                    'a',
                    computation_types.FederatedType(
                        np.int32, placements.SERVER
                    ),
                ),
                other_result_param,
            ]),
        ),
    )
    lambda_parameter_extension_name = 'intermediate_state'
    transformed_comp, new_input_comps = (
        transformations._augment_lambda_with_parameter_for_unbound_references(
            comp, lambda_parameter_extension_name
        )
    )

    self._check_transformed_comp_validity(
        comp, transformed_comp, lambda_parameter_extension_name
    )
    self.assertLen(new_input_comps, 2)
    for new_input_comp, expected_new_input_comp in zip(
        new_input_comps, [federated_sum_param, other_result_param]
    ):
      self.assertEqual(new_input_comp.proto, expected_new_input_comp.proto)

  def test_identifies_unbound_refs_in_struct(self):
    original_arg_type = computation_types.StructType(
        [computation_types.FederatedType(np.int32, placements.CLIENTS)]
    )
    int_at_clients_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    comp = building_blocks.Lambda(
        'arg',
        original_arg_type,
        building_blocks.Block(
            [(
                'a',
                building_block_factory.create_federated_sum(
                    building_block_factory.create_federated_zip(
                        building_blocks.Struct([
                            building_blocks.Selection(
                                building_blocks.Reference(
                                    'arg', original_arg_type
                                ),
                                index=0,
                            ),
                            building_blocks.Reference('b', int_at_clients_type),
                            building_blocks.Struct([
                                building_blocks.Reference(
                                    'c', int_at_clients_type
                                )
                            ]),
                        ])
                    )
                ),
            )],
            building_blocks.Reference(
                'a',
                computation_types.FederatedType(
                    [np.int32, np.int32, [np.int32]], placements.SERVER
                ),
            ),
        ),
    )
    lambda_parameter_extension_name = 'intermediate_state'
    transformed_comp, new_input_comps = (
        transformations._augment_lambda_with_parameter_for_unbound_references(
            comp, lambda_parameter_extension_name
        )
    )

    self._check_transformed_comp_validity(
        comp, transformed_comp, lambda_parameter_extension_name
    )
    self.assertLen(new_input_comps, 2)
    for new_input_comp, expected_new_input_comp in zip(
        new_input_comps,
        [
            building_blocks.Reference('b', int_at_clients_type),
            building_blocks.Reference('c', int_at_clients_type),
        ],
    ):
      self.assertEqual(new_input_comp.proto, expected_new_input_comp.proto)

  def test_no_unbound_refs(self):
    original_arg_type = computation_types.StructType([np.int32])
    comp = building_blocks.Lambda(
        'arg',
        original_arg_type,
        building_blocks.Selection(
            building_blocks.Reference('arg', original_arg_type), index=0
        ),
    )
    lambda_parameter_extension_name = 'intermediate_state'
    transformed_comp, new_input_comps = (
        transformations._augment_lambda_with_parameter_for_unbound_references(
            comp, lambda_parameter_extension_name
        )
    )

    self._check_transformed_comp_validity(
        comp, transformed_comp, lambda_parameter_extension_name
    )
    self.assertEmpty(new_input_comps)

  def test_parameter_usage_without_selection(self):
    original_arg_type = computation_types.StructType([np.int32])
    comp = building_blocks.Lambda(
        'arg',
        original_arg_type,
        building_blocks.Reference('arg', original_arg_type),
    )
    lambda_parameter_extension_name = 'intermediate_state'
    with self.assertRaises(ValueError):
      transformations._augment_lambda_with_parameter_for_unbound_references(
          comp, lambda_parameter_extension_name
      )


class DivisiveForceAlignAndSplitByIntrinsicsTest(absltest.TestCase):

  def find_intrinsics_in_comp(self, comp):
    found_intrinsics = []

    def _find_intrinsics(building_block):
      nonlocal found_intrinsics
      if isinstance(building_block, building_blocks.Call) and isinstance(
          building_block.function, building_blocks.Intrinsic
      ):
        found_intrinsics.append(building_block.function.uri)

    tree_analysis.visit_postorder(comp, _find_intrinsics)
    return found_intrinsics

  def check_split_signatures(self, original_comp, before, intrinsic, after):
    for comp in (before, intrinsic, after):
      self.assertIsInstance(comp, building_blocks.Lambda)
      self.assertIsInstance(
          comp.type_signature.parameter, computation_types.StructType
      )
      self.assertIsInstance(comp.result, building_blocks.Block)

    original_comp = transformations.to_call_dominant(original_comp)
    original_comp = tree_transformations.normalize_types(
        original_comp, normalize_all_equal_bit=False
    )

    self.assertIsInstance(
        before.type_signature.result, computation_types.StructType
    )
    self.assertEqual(
        [x for x, _ in structure.to_elements(before.type_signature.result)],
        ['intrinsic_args_from_before_comp', 'intermediate_state'],
    )
    self.assertIsInstance(before.result.result[0], building_blocks.Struct)
    self.assertIsInstance(before.result.result[1], building_blocks.Struct)

    intrinsic_args_from_before_comp_index = (
        len(intrinsic.type_signature.parameter) - 1
    )
    intrinsic_arg_names = [
        x for x, _ in structure.to_elements(intrinsic.type_signature.parameter)
    ]
    self.assertEqual(
        intrinsic_arg_names[intrinsic_args_from_before_comp_index],
        'intrinsic_args_from_before_comp',
    )
    self.assertEqual(
        intrinsic.type_signature.parameter[
            intrinsic_args_from_before_comp_index
        ],
        before.type_signature.result[0],
    )
    self.assertIsInstance(
        intrinsic.type_signature.result, computation_types.StructType
    )
    self.assertLen(
        intrinsic.result.locals, len(intrinsic.type_signature.result)
    )

    intrinsic_results_index = len(after.type_signature.parameter) - 2
    intermediate_state_index = len(after.type_signature.parameter) - 1
    after_comp_parameter_names = [
        x for x, _ in structure.to_elements(after.type_signature.parameter)
    ]
    self.assertEqual(
        after_comp_parameter_names[intrinsic_results_index], 'intrinsic_results'
    )
    self.assertEqual(
        after_comp_parameter_names[intermediate_state_index],
        'intermediate_state',
    )
    self.assertEqual(
        after.type_signature.parameter[intrinsic_results_index],
        intrinsic.type_signature.result,
    )
    self.assertEqual(
        after.type_signature.parameter[intermediate_state_index],
        before.type_signature.result[1],
    )
    self.assertEqual(
        after.type_signature.result, original_comp.type_signature.result
    )

  def test_splits_on_intrinsic(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0
    intrinsic_call = building_block_factory.create_federated_broadcast(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=server_data_index
        )
    )
    comp = building_blocks.Lambda('arg', arg_type, intrinsic_call)

    # Allow the before and after comps to depend on the entire original comp
    # input. Do not allow the intrinsic comp to depend on the original comp
    # input at all.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_broadcast intrinsic is only present in the
    # intrinsic comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri],
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))

    # Check that one intrinsic arg is passed from the before comp.
    self.assertLen(before.result.result[0], 1)

  def test_fails_split_with_unavailable_subparameters_to_before_comp(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0
    intrinsic_call = building_block_factory.create_federated_broadcast(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=server_data_index
        )
    )
    comp = building_blocks.Lambda('arg', arg_type, intrinsic_call)

    # Do not allow the before or intrinsic comps to depend on the original comp
    # input at all. This should fail when the before comp attempts to produce
    # an output containing the args needed by the intrinsic comp.
    with self.assertRaises(transformations.UnavailableRequiredInputsError):
      transformations.divisive_force_align_and_split_by_intrinsics(
          comp,
          [intrinsic_defs.FEDERATED_BROADCAST],
          before_comp_allowed_original_arg_subparameters=[],
          intrinsic_comp_allowed_original_arg_subparameters=[],
          after_comp_allowed_original_arg_subparameters=[()],
      )

  def test_splits_on_intrinsic_with_multiple_args(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    client_data_index = 1
    intrinsic_call = building_block_factory.create_federated_mean(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=client_data_index
        ),
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=client_data_index
        ),
    )
    comp = building_blocks.Lambda('arg', arg_type, intrinsic_call)

    # Allow the before comp to depend on the client portion of the original
    # comp input. Don't allow the intrinsic comp to depend on the original comp
    # input at all.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_WEIGHTED_MEAN],
            before_comp_allowed_original_arg_subparameters=[
                (client_data_index,)
            ],
            intrinsic_comp_allowed_original_arg_subparameters=[],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_weighted_mean intrinsic is only present in the
    # intrinsic comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri],
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))

    # Check that both intrinsic args are passed from the before comp.
    self.assertLen(before.result.result[0], 2)
    # The intrinsic comp should consist of exactly one intrinsic call with
    # two args.
    self.assertLen(intrinsic.result.locals, 1)
    self.assertIsInstance(
        intrinsic.result.locals[0][1].argument, building_blocks.Struct
    )
    self.assertLen(intrinsic.result.locals[0][1].argument, 2)

  def test_splits_on_intrinsic_with_args_from_original_arg(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    intermediate_state_type = [np.int32]
    server_data_index = 0
    client_data_index = 1
    intermediate_state_index = 2
    arg_type = [
        (None, server_val_type),
        (None, client_val_type),
        ('intermediate_state', intermediate_state_type),
    ]
    intrinsic_call = building_block_factory.create_federated_secure_modular_sum(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=client_data_index
        ),
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('arg', arg_type),
                index=intermediate_state_index,
            ),
            index=0,
        ),
        preapply_modulus=False,
    )
    comp = building_blocks.Lambda('arg', arg_type, intrinsic_call)

    # Allow the before comp to depend on the client portion of the original
    # comp input and the intrinsic comp to depend on the intermediate
    # state portion of the original comp input. Allow the after comp to depend
    # on the server and intermediate state portions of the original comp input.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM],
            before_comp_allowed_original_arg_subparameters=[
                (client_data_index,)
            ],
            intrinsic_comp_allowed_original_arg_subparameters=[
                (intermediate_state_index,)
            ],
            after_comp_allowed_original_arg_subparameters=[
                (server_data_index,),
                (intermediate_state_index,),
            ],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_secure_modular_sum intrinsic is only present in
    # the intrinsic comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM.uri],
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))

    # Check that one intrinsic arg is passed from the before comp and the other
    # comes from the original arg.
    self.assertLen(before.result.result[0], 1)
    self.assertLen(intrinsic.result.locals, 1)
    self.assertIsInstance(
        intrinsic.result.locals[0][1].argument, building_blocks.Struct
    )
    self.assertLen(intrinsic.result.locals[0][1].argument, 2)

  def test_splits_with_non_empty_before_and_after_block_comps(self):
    server_val_type = computation_types.FederatedType(
        [np.int32, np.int32], placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    inner_server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    server_data_index = 0

    selecting_function = building_blocks.Lambda(
        'inner_arg',
        server_val_type.member,
        building_blocks.Selection(
            building_blocks.Reference('inner_arg', server_val_type.member),
            index=0,
        ),
    )
    block_locals = [
        (
            'inner_server_data_selection',
            building_block_factory.create_federated_map_or_apply(
                selecting_function,
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=server_data_index,
                ),
            ),
        ),
        (
            'broadcast_result',
            building_block_factory.create_federated_broadcast(
                building_blocks.Reference(
                    'inner_server_data_selection', inner_server_val_type
                )
            ),
        ),
        (
            'another_inner_server_data_selection',
            building_block_factory.create_federated_map_or_apply(
                selecting_function,
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=server_data_index,
                ),
            ),
        ),
    ]
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Struct([
                building_blocks.Reference(
                    'broadcast_result',
                    computation_types.FederatedType(
                        np.int32, placements.CLIENTS
                    ),
                ),
                building_blocks.Reference(
                    'another_inner_server_data_selection',
                    inner_server_val_type,
                ),
            ]),
        ),
    )

    # Allow the before comp to depend on the server portion of the original comp
    # input, the intrinsic comp to depend on none of it, and the after comp to
    # depend on all of it.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[
                (server_data_index,)
            ],
            intrinsic_comp_allowed_original_arg_subparameters=[],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_broadcast intrinsic is only present in the
    # intrinsic comp and that the before and after comps have a federated_apply
    # intrinsic.
    self.assertEqual(
        self.find_intrinsics_in_comp(before),
        [intrinsic_defs.FEDERATED_APPLY.uri],
    )
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri],
    )
    self.assertEqual(
        self.find_intrinsics_in_comp(after),
        [intrinsic_defs.FEDERATED_APPLY.uri],
    )

    # Check that the before and after comps have blocks with at least one local.
    self.assertNotEmpty(before.result.locals)
    self.assertNotEmpty(after.result.locals)

  def test_splits_with_no_matching_intrinsics(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    client_data_index = 1

    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_block_factory.create_federated_sum(
            building_blocks.Selection(
                building_blocks.Reference('arg', arg_type),
                index=client_data_index,
            )
        ),
    )

    # Allow the output comps to depend on all portions of the original comp
    # input.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_sum intrinsic is only present in the
    # after comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEmpty(self.find_intrinsics_in_comp(intrinsic))
    self.assertEqual(
        self.find_intrinsics_in_comp(after),
        [intrinsic_defs.FEDERATED_SUM.uri],
    )

    # Check that the intermediate state is empty.
    self.assertEmpty(before.result.result[1])

  def test_splits_with_intermediate_state_for_unbound_refs(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0

    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=server_data_index
        ),
    )

    # Allow the before and intrinsic comps to depend on all portions of the
    # original comp input. Don't allow the after comp to depend on any of it
    # (this will force the server_state portion of the return value to be
    # passed through the intermediate state).
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the intermediate state is non-empty and contains the expected
    # value (note that there is an extra layer of indirection to access the
    # original arg).
    self.assertNotEmpty(before.result.result[1])
    self.assertEqual(
        before.result.result[1].proto,
        building_blocks.Struct(
            [
                building_blocks.Selection(
                    building_blocks.Selection(
                        building_blocks.Reference(
                            before.parameter_name, before.parameter_type
                        ),
                        index=0,
                    ),
                    index=server_data_index,
                )
            ]
        ).proto,
    )

    # Allow all the output comps to depend on all portions of the original comp
    # input. Intermediate state should no longer be needed.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the intermediate state is empty.
    self.assertEmpty(before.result.result[1])

  def test_splits_with_intermediate_state_for_duplication(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]

    block_locals = []
    # Create a local that will be needed in both the before and after comps.
    proto = mock.create_autospec(
        computation_pb2.Computation, spec_set=True, instance=True
    )
    function_type = computation_types.FunctionType(None, np.int32)
    compiled = building_blocks.CompiledComputation(
        proto, name='state_1', type_signature=function_type
    )
    state_1 = building_blocks.Call(compiled, None)
    server_state_val_call_1 = building_block_factory.create_federated_value(
        state_1,
        placements.SERVER,
    )
    block_locals.append(('server_state_val_1', server_state_val_call_1))
    proto = mock.create_autospec(
        computation_pb2.Computation, spec_set=True, instance=True
    )
    function_type = computation_types.FunctionType(None, np.int32)
    compiled = building_blocks.CompiledComputation(
        proto, name='state_2', type_signature=function_type
    )
    state_2 = building_blocks.Call(compiled, None)
    server_state_val_call_2 = building_block_factory.create_federated_value(
        state_2,
        placements.SERVER,
    )
    block_locals.append(('server_state_val_2', server_state_val_call_2))
    broadcast_call_1 = building_block_factory.create_federated_broadcast(
        building_blocks.Reference(
            'server_state_val_1', server_state_val_call_1.type_signature
        )
    )
    block_locals.append(('broadcast_result_1', broadcast_call_1))
    broadcast_call_2 = building_block_factory.create_federated_broadcast(
        building_blocks.Reference(
            'server_state_val_2', server_state_val_call_2.type_signature
        )
    )
    block_locals.append(('broadcast_result_2', broadcast_call_2))
    federated_zip_call = building_block_factory.create_federated_zip(
        building_blocks.Struct([
            building_blocks.Reference(
                'broadcast_result_1', broadcast_call_1.type_signature
            ),
            building_blocks.Reference(
                'broadcast_result_2', broadcast_call_2.type_signature
            ),
        ])
    )
    block_locals.append(('federated_zip_result', federated_zip_call))
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Struct([
                building_blocks.Reference(
                    'federated_zip_result',
                    federated_zip_call.type_signature,
                ),
                building_blocks.Reference(
                    'server_state_val_1',
                    server_state_val_call_1.type_signature,
                ),
            ]),
        ),
    )

    # Allow all the output comps to depend on all portions of the original comp
    # input. Intermediate state should no longer be needed.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the intermediate state is non-empty and contains the expected
    # references (a renamed version of server_state_val_1 but nothing
    # corresponding to server_state_val_2).
    self.assertNotEmpty(before.result.result[1])

    def _predicate(
        building_block: building_blocks.ComputationBuildingBlock,
    ) -> bool:
      return isinstance(
          building_block, building_blocks.Reference
      ) and isinstance(
          building_block.type_signature, computation_types.FederatedType
      )

    self.assertEqual(
        tree_analysis.count(before.result.result[1], _predicate), 1
    )

    # Check that the before comp has two federated_value_at_server intrinsics.
    self.assertEqual(
        self.find_intrinsics_in_comp(before),
        [intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri] * 2,
    )
    # Check that the intrinsic comp has two broadcast intrinsics.
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri] * 2,
    )
    # Check that the after comp only has a federated_zip_at_clients intrinsic.
    self.assertEqual(
        self.find_intrinsics_in_comp(after),
        [intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri],
    )

  def test_splits_with_intermediate_state_for_duplication_and_unbound_refs(
      self,
  ):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0

    block_locals = []
    # Create a local that will be needed in both the before and after comps.
    proto = mock.create_autospec(
        computation_pb2.Computation, spec_set=True, instance=True
    )
    function_type = computation_types.FunctionType(None, np.int32)
    compiled = building_blocks.CompiledComputation(
        proto, name='state', type_signature=function_type
    )
    state = building_blocks.Call(compiled, None)
    server_state_val_call = building_block_factory.create_federated_value(
        state,
        placements.SERVER,
    )
    block_locals.append(('server_state_val', server_state_val_call))
    broadcast_call = building_block_factory.create_federated_broadcast(
        building_blocks.Reference(
            'server_state_val', server_state_val_call.type_signature
        )
    )
    block_locals.append(('broadcast_result', broadcast_call))
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Struct([
                building_blocks.Reference(
                    'broadcast_result', broadcast_call.type_signature
                ),
                building_blocks.Reference(
                    'server_state_val', server_state_val_call.type_signature
                ),
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=server_data_index,
                ),
            ]),
        ),
    )

    # Allow only the before and intrinsic comps to depend on all portions of the
    # original comp. Force the intermediate state to be used to make the server
    # data available to the after comp.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the intermediate state is non-empty and contains the expected
    # values (the server data selection and a reference that is a renamed
    # version of server_state_val).
    self.assertNotEmpty(before.result.result[1])

    def _server_data_selection_predicate(bb):
      return (
          isinstance(bb, building_blocks.Selection)
          and bb.source.name == before.parameter_name
      )

    def _server_state_val_predicate(bb):
      return isinstance(bb, building_blocks.Reference) and isinstance(
          bb.type_signature, computation_types.FederatedType
      )

    self.assertEqual(
        tree_analysis.count(
            before.result.result[1], _server_data_selection_predicate
        ),
        1,
    )
    self.assertEqual(
        tree_analysis.count(
            before.result.result[1], _server_state_val_predicate
        ),
        1,
    )

    # Check that the before comp has only a federated_value_at_server intrinsic.
    self.assertEqual(
        self.find_intrinsics_in_comp(before),
        [intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri],
    )
    # Check that the intrinsic comp has only a broadcast intrinsics.
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri],
    )
    # Check that the after comp has no intrinsics.
    self.assertEmpty(self.find_intrinsics_in_comp(after))

  def test_splits_on_multiple_instances_of_intrinsic(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0
    broadcast_result_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )

    block_locals = []
    block_locals.append((
        'broadcast_result_1',
        building_block_factory.create_federated_broadcast(
            building_blocks.Selection(
                building_blocks.Reference('arg', arg_type),
                index=server_data_index,
            )
        ),
    ))
    block_locals.append((
        'broadcast_result_2',
        building_block_factory.create_federated_broadcast(
            building_blocks.Selection(
                building_blocks.Reference('arg', arg_type),
                index=server_data_index,
            )
        ),
    ))
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Struct([
                building_blocks.Reference(
                    'broadcast_result_1', broadcast_result_type
                ),
                building_blocks.Reference(
                    'broadcast_result_2', broadcast_result_type
                ),
            ]),
        ),
    )

    # Allow the before and after comps to depend on the entire original comp
    # input, but do not allow the intrinsic comp to depend on the original comp
    # input at all.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_broadcast intrinsic is only present in the
    # intrinsic comp, and that there are two instances of it.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri] * 2,
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))

    # Check that the two intrinsic args (one for each broadcast call) are
    # passed from the before comp.
    self.assertLen(before.result.result[0], 2)
    self.assertLen(intrinsic.result.locals, 2)

  def test_splits_on_multiple_intrinsics(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    intermediate_state_type = [np.int32]
    server_data_index = 0
    client_data_index = 1
    intermediate_state_index = 2
    arg_type = [
        (None, server_val_type),
        (None, client_val_type),
        ('intermediate_state', intermediate_state_type),
    ]

    federated_sum_call = building_block_factory.create_federated_sum(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=client_data_index
        )
    )
    federated_secure_modular_sum_call = (
        building_block_factory.create_federated_secure_modular_sum(
            building_blocks.Selection(
                building_blocks.Reference('arg', arg_type),
                index=client_data_index,
            ),
            building_blocks.Selection(
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=intermediate_state_index,
                ),
                index=0,
            ),
            preapply_modulus=False,
        )
    )
    block_locals = [
        ('federated_sum_result', federated_sum_call),
        (
            'federated_secure_modular_sum_result',
            federated_secure_modular_sum_call,
        ),
    ]
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Struct([
                building_blocks.Reference(
                    'federated_sum_result',
                    federated_sum_call.type_signature,
                ),
                building_blocks.Reference(
                    'federated_secure_modular_sum_result',
                    federated_secure_modular_sum_call.type_signature,
                ),
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=server_data_index,
                ),
            ]),
        ),
    )

    # Allow the before comp to depend on the client portion of the original
    # comp input, the intrinsic comp to depend on the intermediate state
    # portion of the original comp input, and the after comp to depend on the
    # server and intermediate state portions of the original comp input.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [
                intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM,
                intrinsic_defs.FEDERATED_SUM,
                intrinsic_defs.FEDERATED_MEAN,
            ],
            before_comp_allowed_original_arg_subparameters=[
                (client_data_index,)
            ],
            intrinsic_comp_allowed_original_arg_subparameters=[
                (intermediate_state_index,)
            ],
            after_comp_allowed_original_arg_subparameters=[
                (server_data_index,),
                (intermediate_state_index,),
            ],
        )
    )

    self.check_split_signatures(comp, before, intrinsic, after)

    # Check that the federated_sum and federated_secure_modular_sum intrinsics
    # are only present in the intrinsic comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        set(self.find_intrinsics_in_comp(intrinsic)),
        set([
            intrinsic_defs.FEDERATED_SUM.uri,
            intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM.uri,
        ]),
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))

    # Check that the intrinsic args are obtained correctly. There should be
    # two intrinsic args passed directly from the before comp and one intrinsic
    # arg (the modulus) that is retrieved from the intermediate state from the
    # original comp input.
    self.assertLen(before.result.result[0], 2)
    # There should be two intrinsic calls.
    self.assertLen(intrinsic.result.locals, 2)
    # The federated_sum call takes one arg.
    self.assertNotIsInstance(
        intrinsic.result.locals[0][1].argument, building_blocks.Struct
    )
    # The federated_secure_modular_sum call takes two args.
    self.assertIsInstance(
        intrinsic.result.locals[1][1].argument, building_blocks.Struct
    )
    self.assertLen(intrinsic.result.locals[1][1].argument, 2)

  def test_cannot_split_on_chained_intrinsic(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0

    block_locals = [
        (
            'broadcast_result_at_clients',
            building_block_factory.create_federated_broadcast(
                building_blocks.Selection(
                    building_blocks.Reference('arg', arg_type),
                    index=server_data_index,
                )
            ),
        ),
        (
            'federated_sum_result',
            building_block_factory.create_federated_sum(
                building_blocks.Reference(
                    'broadcast_result_at_clients',
                    computation_types.FederatedType(
                        np.int32, placements.CLIENTS
                    ),
                )
            ),
        ),
    ]
    comp = building_blocks.Lambda(
        'arg',
        arg_type,
        building_blocks.Block(
            block_locals,
            building_blocks.Reference('federated_sum_result', server_val_type),
        ),
    )

    # Allow all parts of the split to depend on the entire original comp input.
    with self.assertRaises(transformations.NonAlignableAlongIntrinsicError):
      transformations.divisive_force_align_and_split_by_intrinsics(
          comp,
          [
              intrinsic_defs.FEDERATED_BROADCAST,
              intrinsic_defs.FEDERATED_SUM,
          ],
          before_comp_allowed_original_arg_subparameters=[()],
          intrinsic_comp_allowed_original_arg_subparameters=[()],
          after_comp_allowed_original_arg_subparameters=[()],
      )

  def test_splits_on_nested_in_tuple_broadcast(self):
    server_val_type = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    client_val_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    arg_type = [server_val_type, client_val_type]
    server_data_index = 0

    first_broadcast = building_block_factory.create_federated_broadcast(
        building_blocks.Selection(
            building_blocks.Reference('arg', arg_type), index=server_data_index
        )
    )
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3])
    )
    packed_broadcast = building_blocks.Struct(
        [building_blocks.Data(any_proto, server_val_type), first_broadcast]
    )
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    comp = building_blocks.Lambda('arg', arg_type, second_broadcast)

    # Allow all parts of the split to depend on the entire original comp input.
    before, intrinsic, after = (
        transformations.divisive_force_align_and_split_by_intrinsics(
            comp,
            [intrinsic_defs.FEDERATED_BROADCAST],
            before_comp_allowed_original_arg_subparameters=[()],
            intrinsic_comp_allowed_original_arg_subparameters=[()],
            after_comp_allowed_original_arg_subparameters=[()],
        )
    )

    # Check that there is only one federated_broadcast intrinsic present in the
    # intrinsic comp.
    self.assertEmpty(self.find_intrinsics_in_comp(before))
    self.assertEqual(
        self.find_intrinsics_in_comp(intrinsic),
        [intrinsic_defs.FEDERATED_BROADCAST.uri],
    )
    self.assertEmpty(self.find_intrinsics_in_comp(after))


if __name__ == '__main__':
  absltest.main()
