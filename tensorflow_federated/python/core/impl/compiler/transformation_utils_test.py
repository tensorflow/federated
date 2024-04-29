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

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _construct_complex_symbol_tree():
  """Constructs complex context tree for mutation testing."""
  symbol_tree = transformation_utils.SymbolTree(FakeTracker)
  for _ in range(2):
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree.walk_down_one_variable_binding()
  symbol_tree.drop_scope_down(0)
  for _ in range(2):
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree.walk_down_one_variable_binding()
  return symbol_tree


def _construct_simple_block(type_signature):
  """Constructs minimal example of LET construct in TFF."""
  test_arg = building_blocks.Literal(1, type_signature)
  result = building_blocks.Reference('x', test_arg.type_signature)
  simple_block = building_blocks.Block([('x', test_arg)], result)
  return simple_block


class UpdatableTracker(transformation_utils.BoundVariableTracker):

  def __init__(self, name, value):
    super().__init__(name, value)
    self.count = 0

  def update(self, comp):
    self.count += 1

  def __str__(self):
    return '{Count: ' + str(self.count) + '}'

  def __eq__(self, other):
    return id(self) == id(other)


class FakeTracker(transformation_utils.BoundVariableTracker):

  def update(self, comp=None):
    pass

  def __str__(self):
    return self.name

  def __eq__(self, other):
    return isinstance(other, FakeTracker)


def fake_tracker_node_factory():
  return transformation_utils.SequentialBindingNode(
      FakeTracker('FakeTracker', None)
  )


class TrivialBoundVariableTracker(transformation_utils.BoundVariableTracker):

  def update(self, comp):
    pass

  def __str__(self):
    return 'TrivialBoundVariableTracker'


def _construct_trivial_instance_of_all_computation_building_blocks():
  cbb_list = []
  any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
      np.array(1, np.int32)
  )
  ref_to_x = building_blocks.Reference('x', np.int32)
  cbb_list.append(('reference', ref_to_x))
  lam = building_blocks.Lambda('x', np.int32, ref_to_x)
  cbb_list.append(('lambda', lam))
  block = building_blocks.Block([('x', ref_to_x)], lam)
  cbb_list.append(('block', block))
  data = building_blocks.Data(any_proto, np.int32)
  cbb_list.append(('data', data))
  function_type = computation_types.FunctionType(np.int32, np.int32)
  intrinsic = building_blocks.Intrinsic('whimsy_intrinsic', function_type)
  cbb_list.append(('intrinsic', intrinsic))
  tff_struct = building_blocks.Struct([ref_to_x])
  cbb_list.append(('struct', tff_struct))
  selection = building_blocks.Selection(tff_struct, index=0)
  cbb_list.append(('selection', selection))
  call = building_blocks.Call(lam, ref_to_x)
  cbb_list.append(('call', call))
  tensor_type = computation_types.TensorType(np.int32)
  proto = computation_factory.create_lambda_identity(tensor_type)
  function_type = computation_types.FunctionType(tensor_type, tensor_type)
  compiled_comp = building_blocks.CompiledComputation(
      proto, type_signature=function_type
  )
  cbb_list.append(('compiled_comp', compiled_comp))
  placement = building_blocks.Placement(placements.CLIENTS)
  cbb_list.append(('placement', placement))
  return cbb_list


def _get_number_of_nodes_via_transform_postorder(comp, predicate=None):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  count = 0

  def fn(comp):
    nonlocal count
    if predicate is None or predicate(comp):
      count += 1
    return comp, False

  transformation_utils.transform_postorder(comp, fn)
  return count


def _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
    comp, predicate=None
):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
  count = 0

  def fn(comp, ctxt_tree):
    nonlocal count
    del ctxt_tree
    if predicate is None or predicate(comp):
      count += 1
    return comp, False

  transformation_utils.transform_postorder_with_symbol_bindings(
      comp, fn, empty_context_tree
  )

  return count


def _get_number_of_nodes_via_transform_preorder(comp, predicate=None):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  count = 0

  def fn(comp):
    nonlocal count
    if predicate is None or predicate(comp):
      count += 1
    return comp, False

  transformation_utils.transform_preorder(comp, fn)
  return count


class TransformationUtilsTest(parameterized.TestCase):

  def test_transform_postorder_fails_on_none_comp(self):
    def transform(comp):
      return comp, False

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder(None, transform)

  def test_transform_postorder_fails_on_none_transform(self):
    comp = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder(comp, None)

  def test_transform_postorder_with_lambda_call_selection_and_reference(self):
    function_type = computation_types.FunctionType(np.int32, np.int32)
    ref = building_blocks.Reference('FEDERATED_arg', [function_type, np.int32])
    fn = building_blocks.Selection(ref, index=0)
    arg = building_blocks.Selection(ref, index=1)
    call = building_blocks.Call(fn, arg)
    comp = building_blocks.Lambda(ref.name, np.int32, call)
    self.assertEqual(
        str(comp), '(FEDERATED_arg -> FEDERATED_arg[0](FEDERATED_arg[1]))'
    )

    def _transformation_fn_generator():
      n = 0
      while True:
        n = n + 1

        def _fn(x):
          intrinsic_type = computation_types.FunctionType(
              x.type_signature, x.type_signature
          )
          intrinsic = building_blocks.Intrinsic('F{}'.format(n), intrinsic_type)
          call = building_blocks.Call(intrinsic, x)
          return call, True

        yield _fn

    transformation_fn_sequence = _transformation_fn_generator()

    def tx_fn(x):
      return next(transformation_fn_sequence)(x)

    transfomed_comp, modified = transformation_utils.transform_postorder(
        comp, tx_fn
    )
    self.assertEqual(
        transfomed_comp.compact_representation(),
        (
            'F6((FEDERATED_arg ->'
            ' F5(F2(F1(FEDERATED_arg)[0])(F4(F3(FEDERATED_arg)[1])))))'
        ),
    )
    self.assertTrue(modified)

  def test_transform_postorder_with_block_and_data_to_reference(self):
    ref = building_blocks.Reference('x', np.int32)
    data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    blk = building_blocks.Block([('x', data)], ref)

    def _transformation_fn(comp):
      if isinstance(comp, building_blocks.Block):
        return building_blocks.Block(comp.locals, data), True
      return comp, False

    transformed, modified = transformation_utils.transform_preorder(
        blk, _transformation_fn
    )
    self.assertTrue(modified)
    self.assertEqual(transformed.compact_representation(), '(let x=1 in 1)')

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
      + [(
          'complex_tree',
          building_block_test_utils.create_nested_syntax_tree(),
      )]
  )
  def test_transform_postorder_returns_untransformed(self, comp):
    def transform_noop(comp):
      return comp, False

    same_comp, modified = transformation_utils.transform_postorder(
        comp, transform_noop
    )
    self.assertEqual(
        same_comp.compact_representation(), comp.compact_representation()
    )
    self.assertFalse(modified)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
  )
  def test_transform_postorder_does_not_construct_new_internal(self, comp):
    def transform_noop(comp):
      return comp, False

    same_comp, modified = transformation_utils.transform_postorder(
        comp, transform_noop
    )

    self.assertEqual(comp, same_comp)
    self.assertFalse(modified)

  def test_transform_postorder_hits_all_nodes_once(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()
    self.assertEqual(
        _get_number_of_nodes_via_transform_postorder(complex_ast), 22
    )

  def test_transform_postorder_walks_to_leaves_in_postorder(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Literal):
        leaf_name_order.append(comp.value)
      return comp, False

    transformation_utils.transform_postorder(complex_ast, transform)

    self.assertEqual(leaf_name_order, [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 11])

  def test_transform_postorder_walks_block_locals_postorder(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      return comp, False

    transformation_utils.transform_postorder(complex_ast, transform)

    self.assertEqual(leaf_name_order, ['t', 'u', 'v', 'w', 'x', 'y', 'z'])

  def test_transform_postorder_walks_through_all_internal_nodes_postorder(self):
    """Checks `transform_postorder` walks correctly through any internal node.

    This test is split from the one above because it tests extra cases
    in `transform_postorder`; in particular, all instances of
    `building_blocks.ComputationBuildingBlock` which kick off
    recursive calls of `transform_postorder` are exercised in this test,
    while only a subset are exercised in the above. For example, if the
    logic ingesting a `Call` breaks, this test will fail and the one above
    may pass.
    """
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      elif isinstance(comp, building_blocks.Literal):
        leaf_name_order.append(comp.value)
      return comp, False

    transformation_utils.transform_postorder(complex_ast, transform)
    postorder_nodes = [
        1,
        2,
        3,
        4,
        't',
        5,
        6,
        'u',
        7,
        'v',
        10,
        8,
        9,
        'w',
        'x',
        'y',
        'z',
        11,
    ]

    self.assertEqual(leaf_name_order, list(postorder_nodes))

  # TODO: b/113123410 - Add more tests for corner cases of `transform_preorder`.

  def test_transform_postorder_with_symbol_bindings_fails_on_none_comp(self):
    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)

    def transform(comp, ctxt_tree):
      del ctxt_tree
      return comp, False

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          None, transform, empty_context_tree
      )

  def test_transform_postorder_with_symbol_bindings_fails_on_none_transform(
      self,
  ):
    empty_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    whimsy_comp = building_blocks.Reference('x', np.int32)

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          whimsy_comp, None, empty_symbol_tree
      )

  def test_transform_postorder_with_symbol_bindings_fails_on_none_symbol_tree(
      self,
  ):
    whimsy_comp = building_blocks.Reference('x', np.int32)

    def transform(comp, ctxt_tree):
      del ctxt_tree
      return comp, False

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          whimsy_comp, transform, None
      )

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
      + [('complex_ast', building_block_test_utils.create_nested_syntax_tree())]
  )
  def test_transform_postorder_with_symbol_bindings_returns_untransformed(
      self, comp
  ):
    def transform_noop(comp, ctxt_tree):
      del ctxt_tree
      return comp, False

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    same_comp, _ = (
        transformation_utils.transform_postorder_with_symbol_bindings(
            comp, transform_noop, empty_context_tree
        )
    )
    self.assertEqual(
        same_comp.compact_representation(), comp.compact_representation()
    )

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
  )
  def test_transform_postorder_with_symbol_bindings_does_not_constructs_new_internal_nodes(
      self, comp
  ):
    def transform_noop(comp, ctxt_tree):
      del ctxt_tree
      return comp, False

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    same_comp, _ = (
        transformation_utils.transform_postorder_with_symbol_bindings(
            comp, transform_noop, empty_context_tree
        )
    )
    if not isinstance(
        comp,
        (
            building_blocks.CompiledComputation,
            building_blocks.Data,
            building_blocks.Intrinsic,
            building_blocks.Literal,
            building_blocks.Placement,
            building_blocks.Reference,
        ),
    ):
      self.assertEqual(id(comp), id(same_comp))

  def test_transform_postorder_with_symbol_bindings_hits_all_nodes_once(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    simple_count = _get_number_of_nodes_via_transform_postorder(complex_ast)
    with_hooks_count = (
        _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
            complex_ast
        )
    )

    self.assertEqual(with_hooks_count, simple_count)

  @parameterized.named_parameters(
      ('reference', building_blocks.Reference),
      ('lambda', building_blocks.Lambda),
      ('block', building_blocks.Block),
      ('data', building_blocks.Data),
      ('intrinsic', building_blocks.Intrinsic),
      ('struct', building_blocks.Struct),
      ('selection', building_blocks.Selection),
      ('call', building_blocks.Call),
      ('compiled_computation', building_blocks.CompiledComputation),
      ('placement', building_blocks.Placement),
  )
  def test_transform_postorder_with_symbol_bindings_counts_each_type_correctly(
      self, cbb_type
  ):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    simple_count = _get_number_of_nodes_via_transform_postorder(
        complex_ast, predicate=lambda x: isinstance(x, cbb_type)
    )
    with_hooks_count = (
        _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
            complex_ast, predicate=lambda x: isinstance(x, cbb_type)
        )
    )

    self.assertEqual(with_hooks_count, simple_count)

  def test_transform_postorder_hooks_walks_leaves_in_postorder(self):
    leaf_order = []
    outer_comp = building_block_test_utils.create_nested_syntax_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, building_blocks.Literal):
        leaf_order.append(comp.value)
      return comp, False

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree
    )
    self.assertEqual(leaf_order, [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 11])

  def test_transform_postorder_hooks_walks_block_locals_postorder(self):
    block_locals_order = []
    outer_comp = building_block_test_utils.create_nested_syntax_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          block_locals_order.append(name)
      return comp, False

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree
    )
    self.assertEqual(block_locals_order, ['t', 'u', 'v', 'w', 'x', 'y', 'z'])

  def test_transform_postorder_hooks_walks_variable_declarations_in_order(self):
    variable_binding_order = []
    outer_comp = building_block_test_utils.create_nested_syntax_tree()

    class PreorderHookTracker(transformation_utils.BoundVariableTracker):

      def __init__(self, name, value):
        variable_binding_order.append(name)
        super().__init__(name, value)

      def update(self, value):
        pass

      def __str__(self):
        pass

      def __eq__(self, other):
        return NotImplemented

    empty_context_tree = transformation_utils.SymbolTree(PreorderHookTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, lambda x, y: (x, False), empty_context_tree
    )
    self.assertEqual(
        variable_binding_order, ['arg', 'y', 'z', 't', 'u', 'v', 'x', 'w']
    )

  def test_transform_postorder_hooks_walks_postorder_interleaved(self):
    named_node_order = []
    outer_comp = building_block_test_utils.create_nested_syntax_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          named_node_order.append(name)
      elif isinstance(comp, building_blocks.Literal):
        named_node_order.append(comp.value)
      return comp, False

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree
    )
    correct_results = [
        1,
        2,
        3,
        4,
        't',
        5,
        6,
        'u',
        7,
        'v',
        10,
        8,
        9,
        'w',
        'x',
        'y',
        'z',
        11,
    ]
    self.assertEqual(named_node_order, correct_results)

  def test_transform_postorder_with_symbol_bindings_binds_lambda_param(self):
    result = building_blocks.Reference('x', np.int32)
    lam = building_blocks.Lambda('x', np.int32, result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, building_blocks.Reference):
        ctxt_tree.update_payload_with_name(comp.name)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp, False

    transformation_utils.transform_postorder_with_symbol_bindings(
        lam, transform, empty_symbol_tree
    )

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertIsNone(value_holder[0].value)

  def test_transform_postorder_with_symbol_bindings_binds_single_block_local(
      self,
  ):
    result = building_blocks.Reference('x', np.int32)
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block([('x', arg)], result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, building_blocks.Reference):
        ctxt_tree.update_payload_with_name(comp.name)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp, False

    transformation_utils.transform_postorder_with_symbol_bindings(
        block, transform, empty_symbol_tree
    )

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertEqual(value_holder[0].value, arg)

  def test_transform_postorder_with_symbol_bindings_binds_sequential_block_locals(
      self,
  ):
    result = building_blocks.Reference('x', np.int32)
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    arg2 = building_blocks.Reference('x', np.int32)
    block = building_blocks.Block([('x', arg), ('x', arg2)], result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, building_blocks.Reference):
        ctxt_tree.update_payload_with_name(comp.name)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp, False

    transformation_utils.transform_postorder_with_symbol_bindings(
        block, transform, empty_symbol_tree
    )

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertEqual(value_holder[0].value, arg)
    self.assertEqual(value_holder[1].count, 1)
    self.assertEqual(value_holder[1].name, 'x')
    self.assertEqual(value_holder[1].value, arg2)

  def test_symbol_tree_initializes(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    self.assertIsInstance(
        symbol_tree.active_node.payload, transformation_utils._BeginScopePointer
    )

  def test_symbol_tree_node_reuse_fails(self):
    fake_tracker_node_one = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None)
    )
    fake_tracker_node_two = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None)
    )
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_tracker_node_one)
    symbol_tree._move_to_child(0)
    symbol_tree._add_younger_sibling(fake_tracker_node_two)
    symbol_tree.walk_down_one_variable_binding()
    with self.assertRaisesRegex(ValueError, 'can only appear once'):
      symbol_tree._add_child(1, fake_tracker_node_one)
    with self.assertRaisesRegex(ValueError, 'can only appear once'):
      symbol_tree._add_younger_sibling(fake_tracker_node_one)

  def test_symbol_tree_get_payload_resolves_child_parent_name_conflict(self):
    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       x_tracker
                           |
                           V
                      x_tracker2*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(x_tracker)
      )
      symbol_tree._move_to_child(0)
      x_tracker2 = FakeTracker('x', None)
      symbol_tree._add_child(
          1, transformation_utils.SequentialBindingNode(x_tracker2)
      )
      symbol_tree._move_to_child(1)
      return symbol_tree, x_tracker, x_tracker2

    symbol_tree, _, x_tracker2 = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('x')), id(x_tracker2))

  def test_symbol_tree_get_payload_resolves_sibling_name_conflict(self):
    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       x_tracker
                           |
                      x_tracker2*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(x_tracker)
      )
      symbol_tree._move_to_child(0)
      x_tracker2 = FakeTracker('x', None)
      symbol_tree._add_younger_sibling(
          transformation_utils.SequentialBindingNode(x_tracker2)
      )
      symbol_tree.walk_down_one_variable_binding()
      return symbol_tree, x_tracker, x_tracker2

    symbol_tree, _, x_tracker2 = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('x')), id(x_tracker2))

  def test_symbol_tree_get_payload_addresses_parent(self):
    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       z_tracker
                           |
                           V
                       x_tracker*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      z_tracker = FakeTracker('z', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(z_tracker)
      )
      symbol_tree._move_to_child(0)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          1, transformation_utils.SequentialBindingNode(x_tracker)
      )
      symbol_tree._move_to_child(1)
      return symbol_tree, z_tracker, x_tracker

    symbol_tree, z_tracker, _ = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('z')), id(z_tracker))

  def test_symbol_tree_updates_correct_node_across_siblings(self):
    def _construct_symbol_tree():
      r"""Builds symbol tree with the structure below.

                      Outer Context
                           |
                           V
                        x_tracker
                           |
                        elder_y
                           |
                        young_y*

      Returns:
        Returns this tree and the `SequentialBindingNode`s
        used to construct it.
      """
      x_tracker = transformation_utils.SequentialBindingNode(
          UpdatableTracker('x', None)
      )
      elder_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None)
      )
      young_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None)
      )

      complex_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
      complex_symbol_tree._add_child(4, x_tracker)
      complex_symbol_tree._move_to_child(4)
      complex_symbol_tree._add_younger_sibling(elder_y)
      complex_symbol_tree.walk_down_one_variable_binding()
      complex_symbol_tree._add_younger_sibling(young_y)
      complex_symbol_tree.walk_down_one_variable_binding()
      return complex_symbol_tree, x_tracker, elder_y, young_y

    (complex_symbol_tree, x_tracker, elder_y, young_y) = (
        _construct_symbol_tree()
    )
    complex_symbol_tree.update_payload_with_name('x')
    complex_symbol_tree.update_payload_with_name('y')
    self.assertEqual(x_tracker.payload.count, 1)
    self.assertEqual(young_y.payload.count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('x').count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 1)
    self.assertEqual(elder_y.payload.count, 0)
    self.assertIsNone(complex_symbol_tree.get_payload_with_name('z'))

  def test_symbol_tree_updates_correct_node_across_generations(self):
    def _construct_symbol_tree():
      r"""Builds symbol tree with the structure below.

                      Outer Context
                           |
                           V
                        x_tracker
                           |
                        elder_y
                       /      \
                      V        V
                  young_y*   misdirect_z

      Returns:
        Returns this tree and the `SequentialBindingNode`s
        used to construct it.
      """
      x_tracker = transformation_utils.SequentialBindingNode(
          UpdatableTracker('x', None)
      )
      elder_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None)
      )
      young_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None)
      )
      misdirect_z = transformation_utils.SequentialBindingNode(
          UpdatableTracker('z', None)
      )

      complex_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
      complex_symbol_tree.drop_scope_down(4)
      complex_symbol_tree._add_younger_sibling(x_tracker)
      complex_symbol_tree.walk_down_one_variable_binding()
      complex_symbol_tree._add_younger_sibling(elder_y)
      complex_symbol_tree.walk_down_one_variable_binding()
      complex_symbol_tree.drop_scope_down(5)
      complex_symbol_tree._add_younger_sibling(young_y)
      complex_symbol_tree.walk_down_one_variable_binding()
      complex_symbol_tree.drop_scope_down(6)
      complex_symbol_tree._add_younger_sibling(misdirect_z)
      complex_symbol_tree.pop_scope_up()
      return (complex_symbol_tree, x_tracker, elder_y, young_y, misdirect_z)

    (complex_symbol_tree, x_tracker, elder_y, young_y, misdirect_z) = (
        _construct_symbol_tree()
    )
    complex_symbol_tree.update_payload_with_name('x')
    complex_symbol_tree.update_payload_with_name('y')
    self.assertEqual(x_tracker.payload.count, 1)
    self.assertEqual(young_y.payload.count, 1)
    self.assertEqual(elder_y.payload.count, 0)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('x').count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 1)
    self.assertIsNone(complex_symbol_tree.get_payload_with_name('z'))
    complex_symbol_tree.pop_scope_up()
    complex_symbol_tree.update_payload_with_name('y')
    complex_symbol_tree.update_payload_with_name('y')
    self.assertEqual(elder_y.payload.count, 2)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 2)
    self.assertEqual(misdirect_z.payload.count, 0)
    complex_symbol_tree.walk_to_scope_beginning()
    self.assertIsNone(complex_symbol_tree.get_payload_with_name('y'))

  def test_typechecking_in_symbol_tree_resolve_methods(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(TypeError):
      symbol_tree.get_payload_with_name(0)
    with self.assertRaises(TypeError):
      symbol_tree.update_payload_with_name(0)
    with self.assertRaises(ValueError):
      symbol_tree.update_payload_with_name('x')

  def test_symbol_tree_walk_to_scope_beginning_nonempty_scope_moves_to_sentinel(
      self,
  ):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding('x', None)
    fake_tracker_payload = symbol_tree.active_node.payload
    symbol_tree.walk_to_scope_beginning()
    self.assertIsInstance(fake_tracker_payload, FakeTracker)
    self.assertIsInstance(
        symbol_tree.active_node.payload, transformation_utils._BeginScopePointer
    )

  def test_symbol_tree_walk_to_scope_beginning_empty_scope_noops(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    begin_scope_node = symbol_tree.active_node
    symbol_tree.walk_to_scope_beginning()
    self.assertIs(symbol_tree.active_node, begin_scope_node)

  def test_symbol_tree_pop_scope_up_at_top_level_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            FakeTracker('FakeTracker', None)
        ),
    )
    with self.assertRaisesRegex(ValueError, 'highest level'):
      symbol_tree.pop_scope_up()

  def test_symbol_tree_pop_scope_up_one_level_tree_succeeds(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding('x', None)
    symbol_tree.drop_scope_down(0)
    symbol_tree.pop_scope_up()
    self.assertIsInstance(symbol_tree.active_node.payload, FakeTracker)

  def test_symbol_tree_drop_scope_down_fails_bad_type(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(TypeError):
      symbol_tree.drop_scope_down('a')

  def test_symbol_tree_drop_scope_down_moves_to_sentinel(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding('x', None)
    symbol_tree.drop_scope_down(0)
    self.assertIsInstance(
        symbol_tree.active_node.payload, transformation_utils._BeginScopePointer
    )

  def test_symbol_tree_drop_scope_down_equivalent_to_add_child_and_move(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    shadow_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.drop_scope_down(0)
    shadow_symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            transformation_utils._BeginScopePointer()
        ),
    )
    shadow_symbol_tree._move_to_child(0)
    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_symbol_tree_walk_down_bad_variable_binding_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaisesRegex(ValueError, 'nonexistent variable binding'):
      symbol_tree.walk_down_one_variable_binding()

  def test_symbol_tree_walk_down_good_variable_binding_moves_active_node(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree.walk_down_one_variable_binding()
    self.assertIsInstance(symbol_tree.active_node.payload, FakeTracker)

  def test_symbol_tree_walk_down_good_variable_binding_moves_to_bound_variable(
      self,
  ):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding('x', None)
    symbol_tree.walk_to_scope_beginning()
    symbol_tree.walk_down_one_variable_binding()
    self.assertEqual(symbol_tree.get_payload_with_name('x').name, 'x')
    self.assertIsNone(symbol_tree.get_payload_with_name('x').value)

  def test_symbol_tree_ingest_variable_binding_bad_args_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding(
          0, building_blocks.Reference('x', np.int32)
      )
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding('x', 0)

  def test_drop_scope_down_and_ingest_variable_binding_adds_node_to_empty_tree(
      self,
  ):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    shadow_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.drop_scope_down(0)
    symbol_tree.ingest_variable_binding(
        'x', building_blocks.Reference('x', np.int32)
    )
    shadow_symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            transformation_utils._BeginScopePointer()
        ),
    )
    shadow_symbol_tree._move_to_child(0)
    shadow_symbol_tree._add_younger_sibling(
        transformation_utils.SequentialBindingNode(
            FakeTracker('FakeTracker', building_blocks.Reference('x', np.int32))
        )
    )
    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_variable_binding_adds_node_to_empty_tree(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    shadow_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    payload_to_add = FakeTracker(
        'x', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    shadow_symbol_tree._add_younger_sibling(
        transformation_utils.SequentialBindingNode(payload_to_add)
    )

    symbol_tree.ingest_variable_binding(
        payload_to_add.name, payload_to_add.value
    )

    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_variable_binding_adds_node_to_nonempty_tree(self):
    symbol_tree = _construct_complex_symbol_tree()
    shadow_symbol_tree = _construct_complex_symbol_tree()
    payload_to_add = FakeTracker(
        'x', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    shadow_symbol_tree._add_younger_sibling(
        transformation_utils.SequentialBindingNode(payload_to_add)
    )

    symbol_tree.ingest_variable_binding(
        'x', building_blocks.Reference('a', np.int32)
    )

    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_variable_overwrites_existing_node_with_same_name(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.drop_scope_down(1)
    symbol_tree.ingest_variable_binding(
        'y', building_blocks.Literal(2, computation_types.TensorType(np.int32))
    )
    resolved_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(resolved_y.value.value, 2)
    self.assertEqual(str(resolved_y.value.type_signature), 'int32')
    symbol_tree.walk_to_scope_beginning()
    symbol_tree.ingest_variable_binding(
        'y',
        building_blocks.Literal(3.0, computation_types.TensorType(np.float32)),
    )
    changed_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(changed_y.value.value, 3.0)
    self.assertEqual(str(changed_y.value.type_signature), 'float32')

  def test_ingest_variable_overwrite_leaves_unrelated_node_alone(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.drop_scope_down(0)
    symbol_tree.ingest_variable_binding(
        'x',
        building_blocks.Literal(3.0, computation_types.TensorType(np.float32)),
    )
    symbol_tree.drop_scope_down(1)
    symbol_tree.ingest_variable_binding(
        'y', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    resolved_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(resolved_x.value.value, 3.0)
    self.assertEqual(str(resolved_x.value.type_signature), 'float32')
    symbol_tree.pop_scope_up()
    symbol_tree.drop_scope_down(1)
    symbol_tree.ingest_variable_binding(
        'y',
        building_blocks.Literal(4.0, computation_types.TensorType(np.float32)),
    )
    same_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(same_x.value, resolved_x.value)

  def test_ingest_variable_raises_error_on_name_conflict(self):
    symbol_tree = _construct_complex_symbol_tree()
    symbol_tree.drop_scope_down(0)
    symbol_tree.ingest_variable_binding(
        'x',
        building_blocks.Literal(3.0, computation_types.TensorType(np.float32)),
    )
    symbol_tree.drop_scope_down(1)
    symbol_tree.ingest_variable_binding(
        'y', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    symbol_tree.pop_scope_up()
    symbol_tree.drop_scope_down(1)
    with self.assertRaises(ValueError):
      symbol_tree.ingest_variable_binding(
          'z',
          building_blocks.Literal(2, computation_types.TensorType(np.int32)),
      )

  def test_symbol_tree_add_sibling(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None)
    )
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_node)
    symbol_tree.walk_down_one_variable_binding()
    self.assertEqual(id(symbol_tree.active_node), id(fake_node))
    self.assertIsNone(symbol_tree.active_node.children.get(0))
    self.assertIsNone(symbol_tree.active_node.younger_sibling)
    symbol_tree.walk_to_scope_beginning()
    self.assertEqual(
        symbol_tree.active_node.payload,
        transformation_utils._BeginScopePointer(),
    )
    self.assertIsNotNone(symbol_tree.active_node.younger_sibling)
    self.assertIsNone(symbol_tree.active_node.children.get(0))

  def test_symbol_tree_has_younger_sibling(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    self.assertIsNotNone(symbol_tree.active_node.younger_sibling)

  def test_symbol_tree_add_child(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None)
    )
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_node)
    symbol_tree._move_to_child(0)
    self.assertEqual(id(symbol_tree.active_node), id(fake_node))
    symbol_tree.active_node = symbol_tree.active_node.parent
    self.assertEqual(
        symbol_tree.active_node.payload,
        transformation_utils._BeginScopePointer(),
    )

  def test_symbol_tree_move_to_bad_child_fails(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None)
    )
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_node)
    with self.assertRaises(ValueError):
      symbol_tree._move_to_child(1)

  def test_complicated_symbol_tree_equality(self):
    first_tree = _construct_complex_symbol_tree()
    second_tree = _construct_complex_symbol_tree()
    self.assertEqual(first_tree, second_tree)
    second_tree._add_child(
        10,
        transformation_utils.SequentialBindingNode(FakeTracker('alpha', None)),
    )
    self.assertNotEqual(first_tree, second_tree)
    self.assertNotEqual(second_tree, first_tree)

  def test_complicated_symbol_tree_equality_independent_of_active_node(self):
    first_tree = _construct_complex_symbol_tree()
    second_tree = _construct_complex_symbol_tree()
    second_tree.pop_scope_up()
    self.assertEqual(first_tree, second_tree)

  def test_complicated_symbol_tree_resolves_string_correctly(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    for _ in range(2):
      symbol_tree._add_younger_sibling(fake_tracker_node_factory())
      symbol_tree.walk_down_one_variable_binding()
    symbol_tree.drop_scope_down(0)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree.walk_down_one_variable_binding()
    for _ in range(2):
      symbol_tree._add_younger_sibling(fake_tracker_node_factory())
      symbol_tree.walk_down_one_variable_binding()
    symbol_tree.pop_scope_up()
    symbol_tree.drop_scope_down(1)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree.walk_down_one_variable_binding()
    for k in range(2):
      symbol_tree.drop_scope_down(k + 2)
      symbol_tree._add_younger_sibling(fake_tracker_node_factory())
      symbol_tree.walk_down_one_variable_binding()
      symbol_tree.pop_scope_up()

    self.assertEqual(
        str(symbol_tree),
        '[BeginScope]-[FakeTracker]-[FakeTracker]->{([BeginScope]-[FakeTracker]-[FakeTracker]-[FakeTracker]),(([BeginScope]-[FakeTracker*]->{([BeginScope]-[FakeTracker]),(([BeginScope]-[FakeTracker])})}',
    )
    symbol_tree.pop_scope_up()
    self.assertEqual(
        str(symbol_tree),
        '[BeginScope]-[FakeTracker]-[FakeTracker*]->{([BeginScope]-[FakeTracker]-[FakeTracker]-[FakeTracker]),(([BeginScope]-[FakeTracker]->{([BeginScope]-[FakeTracker]),(([BeginScope]-[FakeTracker])})}',
    )

  def test_trivial_subclass_init_fails_bad_args(self):
    with self.assertRaises(TypeError):
      TrivialBoundVariableTracker()
    with self.assertRaises(TypeError):
      TrivialBoundVariableTracker(0, None)
    with self.assertRaises(TypeError):
      TrivialBoundVariableTracker('x', 0)

  def test_trivial_subclass_init(self):
    x = TrivialBoundVariableTracker('x', None)
    self.assertEqual(x.name, 'x')
    self.assertIsNone(x.value)

  def test_sequential_binding_node_fails_bad_args(self):
    with self.assertRaises(TypeError):
      transformation_utils.SequentialBindingNode(None)
    with self.assertRaises(TypeError):
      transformation_utils.SequentialBindingNode(0)

  def test_sequential_binding_node_initialization(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('trivial_name', None)
    )

    self.assertEqual(trivial_instance.payload.name, 'trivial_name')
    self.assertEmpty(trivial_instance.children)
    self.assertIsNone(trivial_instance.payload.value)
    self.assertIsNone(trivial_instance.parent)
    self.assertIsNone(trivial_instance.younger_sibling)
    self.assertIsNone(trivial_instance.older_sibling)

  def test_bound_variable_tracker_trivial_subclass_init_bad_args(self):
    with self.assertRaises(TypeError):
      TrivialBoundVariableTracker(0, None)
    with self.assertRaises(TypeError):
      TrivialBoundVariableTracker('x', 0)

  def test_sequential_binding_node_parent_child_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('trivial_name', None)
    )
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('second_trivial_name', None)
    )

    self.assertNotEqual(trivial_instance, second_trivial_instance)
    second_trivial_instance.set_parent(trivial_instance)
    trivial_instance.add_child(0, second_trivial_instance)
    self.assertEqual(trivial_instance.get_child(0), second_trivial_instance)
    self.assertIsNone(trivial_instance.get_child(1))
    self.assertEqual(second_trivial_instance.parent, trivial_instance)
    with self.assertRaises(TypeError):
      trivial_instance.set_parent(0)
    with self.assertRaises(TypeError):
      second_trivial_instance.add_child(0, 0)

  def test_sequential_binding_node_sibling_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('trivial_name', None)
    )
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('second_trivial_name', None)
    )

    self.assertNotEqual(trivial_instance, second_trivial_instance)
    trivial_instance.set_younger_sibling(second_trivial_instance)
    self.assertEqual(trivial_instance.younger_sibling, second_trivial_instance)
    second_trivial_instance.set_older_sibling(trivial_instance)
    self.assertEqual(second_trivial_instance.older_sibling, trivial_instance)
    with self.assertRaises(TypeError):
      trivial_instance.set_younger_sibling(0)
    with self.assertRaises(TypeError):
      second_trivial_instance.set_older_sibling(0)

  def test_sequential_binding_nodes_cousin_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('trivial_name', None)
    )
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('second_trivial_name', None)
    )
    third_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialBoundVariableTracker('third_trivial_name', None)
    )
    trivial_instance.add_child(0, second_trivial_instance)
    trivial_instance.add_child(1, third_trivial_instance)
    second_trivial_instance.set_parent(trivial_instance)
    third_trivial_instance.set_parent(trivial_instance)
    second_trivial_instance_relations = [
        second_trivial_instance.parent,
        second_trivial_instance.older_sibling,
        second_trivial_instance.younger_sibling,
    ] + list(second_trivial_instance.children.values())

    third_trivial_instance_relations = [
        third_trivial_instance.parent,
        third_trivial_instance.older_sibling,
        third_trivial_instance.younger_sibling,
    ] + list(third_trivial_instance.children.values())
    self.assertNotIn(second_trivial_instance, third_trivial_instance_relations)
    self.assertNotIn(third_trivial_instance, second_trivial_instance_relations)
    self.assertEqual(
        id(second_trivial_instance.parent), id(third_trivial_instance.parent)
    )

  def test_bound_variable_tracker_equality_names(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    whimsy_tracker = TrivialBoundVariableTracker('x', lit)
    second_whimsy_tracker = TrivialBoundVariableTracker('x', lit)
    self.assertEqual(whimsy_tracker, second_whimsy_tracker)
    second_whimsy_tracker.name = 'y'
    self.assertNotEqual(whimsy_tracker, second_whimsy_tracker)
    whimsy_tracker.name = 'y'
    self.assertEqual(whimsy_tracker, second_whimsy_tracker)

  def test_bound_variable_tracker_equality_values(self):
    whimsy_tracker = TrivialBoundVariableTracker(
        'x', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    second_whimsy_tracker = TrivialBoundVariableTracker(
        'x', building_blocks.Literal(2, computation_types.TensorType(np.int32))
    )
    self.assertNotEqual(whimsy_tracker, second_whimsy_tracker)

  def test_outer_context_pointer_equality(self):
    outer_context = transformation_utils._BeginScopePointer()
    other_outer_context = transformation_utils._BeginScopePointer()
    self.assertNotEqual(id(outer_context), id(other_outer_context))
    self.assertEqual(str(outer_context), 'BeginScope')
    self.assertEqual(outer_context, other_outer_context)

  def test_outer_context_pointer_cant_update(self):
    outer_context = transformation_utils._BeginScopePointer()
    with self.assertRaises(RuntimeError):
      outer_context.update()

  def test_reference_tracker_initializes(self):
    whimsy_tracker = transformation_utils.ReferenceCounter(
        'x', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    self.assertEqual(whimsy_tracker.name, 'x')
    self.assertEqual(whimsy_tracker.value.compact_representation(), '1')
    self.assertEqual(whimsy_tracker.count, 0)

  def test_reference_tracker_updates(self):
    whimsy_tracker = transformation_utils.ReferenceCounter(
        'x', building_blocks.Literal(1, computation_types.TensorType(np.int32))
    )
    for k in range(10):
      whimsy_tracker.update()
      self.assertEqual(whimsy_tracker.count, k + 1)

  def test_reference_tracker_equality_instances(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    whimsy_tracker = transformation_utils.ReferenceCounter('x', lit)
    second_whimsy_tracker = transformation_utils.ReferenceCounter('x', lit)
    self.assertEqual(whimsy_tracker, second_whimsy_tracker)
    whimsy_tracker.update()
    self.assertNotEqual(whimsy_tracker, second_whimsy_tracker)
    second_whimsy_tracker.update()
    self.assertEqual(whimsy_tracker, second_whimsy_tracker)

  def test_get_count_of_references_to_variables_simple_block(self):
    simple_block = _construct_simple_block(
        computation_types.TensorType(np.int32)
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        simple_block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      arg = transformation_utils.ReferenceCounter(
          simple_block.locals[0][0], simple_block.locals[0][1]
      )
      arg.update()
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack._add_younger_sibling(
          transformation_utils.SequentialBindingNode(arg)
      )
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_simple_block_two_references(
      self,
  ):
    simple_block = _construct_simple_block(
        computation_types.TensorType(np.int32)
    )
    ref = simple_block.result
    result = building_blocks.Struct([ref, ref])
    simple_block = building_blocks.Block(simple_block.locals, result)
    self.assertEqual(
        simple_block.compact_representation(), '(let x=1 in <x,x>)'
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        simple_block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          simple_block.locals[0][0], simple_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(ref.name)
      constructed_context_stack.update_payload_with_name(ref.name)
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_nested_blocks_conflicting_names(
      self,
  ):
    first_block = _construct_simple_block(
        computation_types.TensorType(np.int32)
    )
    outer_block_output = building_blocks.Reference(
        'x', first_block.type_signature
    )
    second_block = building_blocks.Block(
        [('x', first_block)], outer_block_output
    )

    self.assertEqual(
        second_block.compact_representation(), '(let x=(let x=1 in x) in x)'
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        second_block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.drop_scope_down(2)
      constructed_context_stack.ingest_variable_binding(
          first_block.locals[0][0], first_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(
          first_block.result.name
      )
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.ingest_variable_binding(
          second_block.locals[0][0], second_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(
          second_block.result.name
      )
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(str(context_stack), str(_construct_context_stack()))

  def test_get_count_of_references_to_variables_block_lambda_name_conflict(
      self,
  ):
    innermost_x = building_blocks.Reference('x', np.int32)
    inner_lambda = building_blocks.Lambda('x', np.int32, innermost_x)
    second_x = building_blocks.Reference('x', np.int32)
    called_lambda = building_blocks.Call(inner_lambda, second_x)
    block_input = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    block = building_blocks.Block([('x', block_input)], called_lambda)
    self.assertEqual(block.compact_representation(), '(let x=1 in (x -> x)(x))')
    context_stack = transformation_utils.get_count_of_references_to_variables(
        block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(2)
      constructed_context_stack.ingest_variable_binding(
          block.locals[0][0], block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(second_x.name)
      constructed_context_stack.drop_scope_down(3)
      constructed_context_stack.ingest_variable_binding(
          inner_lambda.parameter_name, None
      )
      constructed_context_stack.update_payload_with_name(innermost_x.name)
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_lambda_name_conflict(self):
    inner_x = building_blocks.Reference('x', np.int32)
    inner_lambda = building_blocks.Lambda('x', np.int32, inner_x)
    outer_x = building_blocks.Reference('x', np.int32)
    call = building_blocks.Call(inner_lambda, outer_x)
    outer_lambda = building_blocks.Lambda('x', np.int32, call)
    self.assertEqual(
        outer_lambda.compact_representation(), '(x -> (x -> x)(x))'
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        outer_lambda
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(0)
      constructed_context_stack.ingest_variable_binding(
          outer_lambda.parameter_name, None
      )
      constructed_context_stack.update_payload_with_name(outer_x.name)
      constructed_context_stack.drop_scope_down(0)
      constructed_context_stack.ingest_variable_binding(
          inner_lambda.parameter_name, None
      )
      constructed_context_stack.update_payload_with_name(inner_x.name)
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_block_local_overwriting_name_in_scope(
      self,
  ):
    arg_comp = building_blocks.Reference('arg', [np.int32, np.int32])
    selected = building_blocks.Selection(arg_comp, index=0)
    internal_arg = building_blocks.Reference('arg', np.int32)
    block = building_blocks.Block([('arg', selected)], internal_arg)
    lam = building_blocks.Lambda('arg', arg_comp.type_signature, block)
    self.assertEqual(
        lam.compact_representation(), '(arg -> (let arg=arg[0] in arg))'
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        lam
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          lam.parameter_name, None
      )
      constructed_context_stack.update_payload_with_name(arg_comp.name)
      constructed_context_stack.drop_scope_down(2)
      constructed_context_stack.ingest_variable_binding(
          block.locals[0][0], block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(internal_arg.name)
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_nested_block_no_name_conflict(
      self,
  ):
    used1 = building_blocks.Reference('used1', np.int32)
    used2 = building_blocks.Literal(2, computation_types.TensorType(np.int32))
    ref = building_blocks.Reference('x', used1.type_signature)
    lower_block = building_blocks.Block([('x', used1)], ref)
    higher_block = building_blocks.Block([('used1', used2)], lower_block)
    self.assertEqual(
        higher_block.compact_representation(),
        '(let used1=2 in (let x=used1 in x))',
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        higher_block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          higher_block.locals[0][0], higher_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(used1.name)
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          lower_block.locals[0][0], lower_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(ref.name)
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_nested_block_no_overwrite(self):
    used1 = building_blocks.Reference('used1', np.int32)
    used2 = building_blocks.Literal(2, computation_types.TensorType(np.int32))
    user_inlined_lower_block = building_blocks.Block([('x', used1)], used1)
    user_inlined_higher_block = building_blocks.Block(
        [('used1', used2)], user_inlined_lower_block
    )
    self.assertEqual(
        user_inlined_higher_block.compact_representation(),
        '(let used1=2 in (let x=used1 in used1))',
    )
    second_context_stack = (
        transformation_utils.get_count_of_references_to_variables(
            user_inlined_higher_block
        )
    )

    def _construct_second_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          user_inlined_higher_block.locals[0][0],
          user_inlined_higher_block.locals[0][1],
      )
      constructed_context_stack.update_payload_with_name(used1.name)
      constructed_context_stack.update_payload_with_name(used1.name)
      constructed_context_stack.drop_scope_down(3)
      constructed_context_stack.ingest_variable_binding(
          user_inlined_lower_block.locals[0][0],
          user_inlined_lower_block.locals[0][1],
      )
      constructed_context_stack.pop_scope_up()
      constructed_context_stack.pop_scope_up()
      return constructed_context_stack

    self.assertEqual(second_context_stack, _construct_second_context_stack())

  def test_get_count_of_references_to_variables_mixed_scope(self):
    innermost = building_blocks.Reference('x', np.int32)
    intermediate_arg = building_blocks.Reference('y', np.int32)
    inner_block = building_blocks.Block([('x', intermediate_arg)], innermost)
    item1 = building_blocks.Reference('x', np.int32)
    mediate_tuple = building_blocks.Struct([item1, inner_block])
    used = building_blocks.Literal(0, computation_types.TensorType(np.int32))
    used1 = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    outer_block = building_blocks.Block(
        [('x', used), ('y', used1)], mediate_tuple
    )
    self.assertEqual(
        outer_block.compact_representation(),
        '(let x=0,y=1 in <x,(let x=y in x)>)',
    )
    context_stack = transformation_utils.get_count_of_references_to_variables(
        outer_block
    )

    def _construct_context_stack():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(1)
      constructed_context_stack.ingest_variable_binding(
          outer_block.locals[0][0], outer_block.locals[0][1]
      )
      constructed_context_stack.ingest_variable_binding(
          outer_block.locals[1][0], outer_block.locals[1][1]
      )
      constructed_context_stack.update_payload_with_name(intermediate_arg.name)
      constructed_context_stack.update_payload_with_name(item1.name)
      constructed_context_stack.drop_scope_down(6)
      constructed_context_stack.ingest_variable_binding(
          inner_block.locals[0][0], inner_block.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(innermost.name)
      return constructed_context_stack

    self.assertEqual(context_stack, _construct_context_stack())

  def test_get_count_of_references_to_variables_sequential_overwrite_in_block_locals(
      self,
  ):
    tensor_type = computation_types.TensorType(np.int32)
    proto = mock.create_autospec(
        computation_pb2.Computation, spec_set=True, instance=True
    )
    function_type = computation_types.FunctionType(None, tensor_type)
    compiled = building_blocks.CompiledComputation(
        proto, name='make_10', type_signature=function_type
    )
    make_10 = building_blocks.Call(compiled, None)

    whimsy_x_reference = building_blocks.Reference('x', np.int32)

    make_13 = building_blocks.Block(
        [
            ('x', make_10),
            ('x', whimsy_x_reference),
            ('x', whimsy_x_reference),
            ('x', whimsy_x_reference),
        ],
        whimsy_x_reference,
    )

    references = transformation_utils.get_count_of_references_to_variables(
        make_13
    )

    child_id = list(references.active_node.children.keys())[0]

    def _make_context_tree():
      constructed_context_stack = transformation_utils.SymbolTree(
          transformation_utils.ReferenceCounter
      )
      constructed_context_stack.drop_scope_down(child_id)
      constructed_context_stack.ingest_variable_binding(
          make_13.locals[0][0], make_13.locals[0][1]
      )
      constructed_context_stack.update_payload_with_name(
          whimsy_x_reference.name
      )
      constructed_context_stack.ingest_variable_binding(
          make_13.locals[1][0], make_13.locals[1][1]
      )
      constructed_context_stack.update_payload_with_name(
          whimsy_x_reference.name
      )
      constructed_context_stack.ingest_variable_binding(
          make_13.locals[2][0], make_13.locals[2][1]
      )
      constructed_context_stack.update_payload_with_name(
          whimsy_x_reference.name
      )
      constructed_context_stack.ingest_variable_binding(
          make_13.locals[3][0], make_13.locals[3][1]
      )
      constructed_context_stack.update_payload_with_name(
          whimsy_x_reference.name
      )
      constructed_context_stack.walk_to_scope_beginning()
      return constructed_context_stack

    constructed_tree = _make_context_tree()
    self.assertEqual(references, constructed_tree)


class TransformPreorderTest(parameterized.TestCase):

  def test_transform_preorder_fails_on_none_comp(self):
    def transform(comp):
      return comp, False

    with self.assertRaises(TypeError):
      transformation_utils.transform_preorder(None, transform)

  def test_transform_preorder_fails_on_none_transform(self):
    comp = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      transformation_utils.transform_preorder(comp, None)

  def test_transform_preorder_with_lambda_call_selection_and_reference(self):
    function_type = computation_types.FunctionType(np.int32, np.int32)
    ref = building_blocks.Reference('FEDERATED_arg', [function_type, np.int32])
    fn = building_blocks.Selection(ref, index=0)
    arg = building_blocks.Selection(ref, index=1)
    call = building_blocks.Call(fn, arg)
    comp = building_blocks.Lambda(ref.name, np.int32, call)
    self.assertEqual(
        comp.compact_representation(),
        '(FEDERATED_arg -> FEDERATED_arg[0](FEDERATED_arg[1]))',
    )

    def _transformation_fn_generator():
      n = 0
      while True:
        n = n + 1

        def _fn(x):
          intrinsic_type = computation_types.FunctionType(
              x.type_signature, x.type_signature
          )
          intrinsic = building_blocks.Intrinsic('F{}'.format(n), intrinsic_type)
          call = building_blocks.Call(intrinsic, x)
          return call, True

        yield _fn

    transformation_fn_sequence = _transformation_fn_generator()

    def tx_fn(x):
      return next(transformation_fn_sequence)(x)

    transfomed_comp, modified = transformation_utils.transform_preorder(
        comp, tx_fn
    )
    self.assertTrue(modified)
    self.assertEqual(
        transfomed_comp.compact_representation(),
        'F1((FEDERATED_arg -> FEDERATED_arg[0](FEDERATED_arg[1])))',
    )
    self.assertTrue(modified)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
      + [(
          'complex_tree',
          building_block_test_utils.create_nested_syntax_tree(),
      )]
  )
  def test_transform_preorder_returns_untransformed(self, comp):
    def transform_noop(comp):
      return comp, False

    same_comp, modified = transformation_utils.transform_preorder(
        comp, transform_noop
    )
    self.assertEqual(
        same_comp.compact_representation(), comp.compact_representation()
    )
    self.assertFalse(modified)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks()
  )
  def test_transform_preorder_does_not_construct_new_internal(self, comp):
    def transform_noop(comp):
      return comp, False

    same_comp, modified = transformation_utils.transform_preorder(
        comp, transform_noop
    )

    self.assertEqual(comp, same_comp)
    self.assertFalse(modified)

  def test_transform_preorder_hits_all_nodes_once(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()
    self.assertEqual(
        _get_number_of_nodes_via_transform_preorder(complex_ast), 22
    )

  def test_transform_preorder_walks_to_leaves_in_preorder(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Literal):
        leaf_name_order.append(comp.value)
      return comp, False

    transformation_utils.transform_preorder(complex_ast, transform)

    self.assertEqual(leaf_name_order, [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 11])

  def test_transform_preorder_walks_block_locals_preorder(self):
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      return comp, False

    transformation_utils.transform_preorder(complex_ast, transform)

    self.assertEqual(leaf_name_order, ['y', 'z', 'v', 't', 'u', 'x', 'w'])

  def test_transform_preorder_walks_through_all_internal_nodes_preorder(self):
    """Checks `transform_preorder` walks correctly through any internal node.

    This test is split from the one above because it tests extra cases
    in `transform_preorder`; in particular, all instances of
    `building_blocks.ComputationBuildingBlock` which kick off
    recursive calls of `transform_preorder` are exercised in this test,
    while only a subset are exercised in the above. For example, if the
    logic ingesting a `Call` breaks, this test will fail and the one above
    may pass.
    """
    complex_ast = building_block_test_utils.create_nested_syntax_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      elif isinstance(comp, building_blocks.Literal):
        leaf_name_order.append(comp.value)
      return comp, False

    transformation_utils.transform_preorder(complex_ast, transform)
    preorder_nodes = [
        'y',
        'z',
        1,
        2,
        'v',
        't',
        3,
        4,
        'u',
        5,
        6,
        7,
        'x',
        10,
        'w',
        8,
        9,
        11,
    ]

    self.assertEqual(leaf_name_order, list(preorder_nodes))

  def test_transform_preorder_passes_transform_through_tuple_correctly(self):

    def transform_intrinsic_to_reference(comp):
      if isinstance(comp, building_blocks.Literal):
        return (
            building_blocks.Reference(str(comp.value), comp.type_signature),
            True,
        )
      return comp, False

    tuple_holding_data = building_blocks.Struct(
        [building_blocks.Literal(1, computation_types.TensorType(np.int32))]
    )
    literal_replaced, modified = transformation_utils.transform_preorder(
        tuple_holding_data, transform_intrinsic_to_reference
    )
    self.assertTrue(modified)
    self.assertEqual(
        literal_replaced.compact_representation(),
        tuple_holding_data.compact_representation(),
    )
    self.assertLen(literal_replaced, 1)
    self.assertIsInstance(literal_replaced[0], building_blocks.Reference)


class GetUniqueNamesTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformation_utils.get_unique_names(None)

  def test_returns_names_single_lambda(self):
    ref = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, ref)
    names = transformation_utils.get_unique_names(lambda_1)
    self.assertCountEqual(names, ('x',))

  def test_returns_names_nested_lambdas_with_different_variable_name(self):
    ref = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, ref)
    lambda_2 = building_blocks.Lambda('y', np.int32, lambda_1)
    names = transformation_utils.get_unique_names(lambda_2)
    self.assertCountEqual(names, ('x', 'y'))

  def test_returns_names_single_block(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block([('x', lit)], lit)
    names = transformation_utils.get_unique_names(block)
    self.assertCountEqual(names, ('x',))

  def test_returns_names_nested_blocks_with_different_variable_name(self):
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block_1 = building_blocks.Block([('x', lit)], lit)
    block_2 = building_blocks.Block([('y', lit)], block_1)
    names = transformation_utils.get_unique_names(block_2)
    self.assertCountEqual(names, ('x', 'y'))

  def test_captures_reference_name(self):
    ref_to_x = building_blocks.Reference('x', np.int32)
    names = transformation_utils.get_unique_names(ref_to_x)
    self.assertCountEqual(names, 'x')

  def test_captures_unbound_reference_name(self):
    ref_to_z = building_blocks.Reference('z', np.int32)
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block_1 = building_blocks.Block([('x', lit)], ref_to_z)
    block_2 = building_blocks.Block([('y', lit)], block_1)
    names = transformation_utils.get_unique_names(block_2)
    self.assertCountEqual(names, ('x', 'y', 'z'))


class GetMapOfUnboundReferencesTest(absltest.TestCase):

  def test_lambda_under_call_to_ref_gets_nothing_unbound(self):
    y_ref = building_blocks.Reference('y', np.int32)
    lambda_1 = building_blocks.Lambda('y', y_ref.type_signature, y_ref)
    x_ref = building_blocks.Reference('x', np.int32)
    call_on_x_ref = building_blocks.Call(lambda_1, x_ref)
    unbound_refs = transformation_utils.get_map_of_unbound_references(
        call_on_x_ref
    )[lambda_1]
    self.assertEmpty(unbound_refs)


if __name__ == '__main__':
  absltest.main()
