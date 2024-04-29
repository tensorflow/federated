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

import collections
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_test_utils


class UniqueNameGeneratorTest(absltest.TestCase):

  def test_does_not_raise_type_error_with_none_comp(self):
    try:
      building_block_factory.unique_name_generator(None)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_returns_unique_names_with_none_comp_and_none_prefix(self):
    name_generator = building_block_factory.unique_name_generator(
        None, prefix=None
    )
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_none_comp_and_unset_prefix(self):
    name_generator = building_block_factory.unique_name_generator(None)
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_none_comp_and_prefix(self):
    name_generator = building_block_factory.unique_name_generator(
        None, prefix='_test'
    )
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_comp_and_none_prefix(self):
    ref = building_blocks.Reference('a', np.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix=None
    )
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_comp_and_unset_prefix(self):
    ref = building_blocks.Reference('a', np.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(comp)
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_comp_and_prefix(self):
    ref = building_blocks.Reference('a', np.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix='_test'
    )
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_conflicting_prefix(self):
    ref = building_blocks.Reference('_test', np.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix='_test'
    )
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertNotEqual(prefix, '_test')
    self.assertTrue(all(n.startswith(prefix) for n in names))


class CreateFederatedGetitemCompTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getitem_comp(None, 0)

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS), ('server', placements.SERVER)
  )
  def test_returns_comp(self, placement):
    federated_value = building_blocks.Reference(
        'test',
        computation_types.FederatedType(
            [('a', np.int32), ('b', np.bool_)], placement
        ),
    )
    get_0_comp = building_block_factory.create_federated_getitem_comp(
        federated_value, 0
    )
    self.assertEqual(str(get_0_comp), '(x -> x[0])')
    get_slice_comp = building_block_factory.create_federated_getitem_comp(
        federated_value, slice(None, None, -1)
    )
    self.assertEqual(str(get_slice_comp), '(x -> <b=x[1],a=x[0]>)')


class CreateFederatedGetattrCompTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getattr_comp(None, 'x')

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS), ('server', placements.SERVER)
  )
  def test_returns_comp(self, placement):
    federated_value = building_blocks.Reference(
        'test',
        computation_types.FederatedType(
            [('a', np.int32), ('b', np.bool_)], placement
        ),
    )
    get_a_comp = building_block_factory.create_federated_getattr_comp(
        federated_value, 'a'
    )
    self.assertEqual(str(get_a_comp), '(x -> x.a)')
    get_b_comp = building_block_factory.create_federated_getattr_comp(
        federated_value, 'b'
    )
    self.assertEqual(str(get_b_comp), '(x -> x.b)')
    non_federated_arg = building_blocks.Reference(
        'test', computation_types.StructType([('a', np.int32), ('b', np.bool_)])
    )
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_getattr_comp(
          non_federated_arg, 'a'
      )
    with self.assertRaisesRegex(ValueError, 'has no element of name `c`'):
      _ = building_block_factory.create_federated_getattr_comp(
          federated_value, 'c'
      )


class CreateFederatedGetattrCallTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getattr_call(None, 'x')

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
  )
  def test_returns_named(self, placement):
    federated_comp_named = building_blocks.Reference(
        'test',
        computation_types.FederatedType(
            [('a', np.int32), ('b', np.bool_), np.int32], placement
        ),
    )
    self.assertEqual(
        str(federated_comp_named.type_signature.member),
        '<a=int32,b=bool,int32>',
    )
    name_a = building_block_factory.create_federated_getattr_call(
        federated_comp_named, 'a'
    )
    name_b = building_block_factory.create_federated_getattr_call(
        federated_comp_named, 'b'
    )
    self.assertIsInstance(
        name_a.type_signature, computation_types.FederatedType
    )
    self.assertIsInstance(
        name_b.type_signature, computation_types.FederatedType
    )
    self.assertEqual(str(name_a.type_signature.member), 'int32')
    self.assertEqual(str(name_b.type_signature.member), 'bool')
    try:
      type_analysis.check_federated_type(
          name_a.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    try:
      type_analysis.check_federated_type(
          name_b.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    with self.assertRaisesRegex(ValueError, 'has no element of name `c`'):
      _ = building_block_factory.create_federated_getattr_call(
          federated_comp_named, 'c'
      )


class CreateFederatedGetitemCallTest(parameterized.TestCase):

  def test_fails_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getitem_call(None, 0)

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
  )
  def test_returns_named(self, placement):
    federated_comp_named = building_blocks.Reference(
        'test',
        computation_types.FederatedType(
            [('a', np.int32), ('b', np.bool_)], placement
        ),
    )
    self.assertEqual(
        str(federated_comp_named.type_signature.member), '<a=int32,b=bool>'
    )
    idx_0 = building_block_factory.create_federated_getitem_call(
        federated_comp_named, 0
    )
    idx_1 = building_block_factory.create_federated_getitem_call(
        federated_comp_named, 1
    )
    self.assertIsInstance(idx_0.type_signature, computation_types.FederatedType)
    self.assertIsInstance(idx_1.type_signature, computation_types.FederatedType)
    self.assertEqual(str(idx_0.type_signature.member), 'int32')
    self.assertEqual(str(idx_1.type_signature.member), 'bool')
    try:
      type_analysis.check_federated_type(
          idx_0.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    try:
      type_analysis.check_federated_type(
          idx_1.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    flipped = building_block_factory.create_federated_getitem_call(
        federated_comp_named, slice(None, None, -1)
    )
    self.assertIsInstance(
        flipped.type_signature, computation_types.FederatedType
    )
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    try:
      type_analysis.check_federated_type(
          flipped.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
  )
  def test_returns_unnamed(self, placement):
    federated_comp_unnamed = building_blocks.Reference(
        'test', computation_types.FederatedType([np.int32, np.bool_], placement)
    )
    self.assertEqual(
        str(federated_comp_unnamed.type_signature.member), '<int32,bool>'
    )
    unnamed_idx_0 = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, 0
    )
    unnamed_idx_1 = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, 1
    )
    self.assertIsInstance(
        unnamed_idx_0.type_signature, computation_types.FederatedType
    )
    self.assertIsInstance(
        unnamed_idx_1.type_signature, computation_types.FederatedType
    )
    self.assertEqual(str(unnamed_idx_0.type_signature.member), 'int32')
    self.assertEqual(str(unnamed_idx_1.type_signature.member), 'bool')
    try:
      type_analysis.check_federated_type(
          unnamed_idx_0.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    try:
      type_analysis.check_federated_type(
          unnamed_idx_1.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )
    unnamed_flipped = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, slice(None, None, -1)
    )
    self.assertIsInstance(
        unnamed_flipped.type_signature, computation_types.FederatedType
    )
    self.assertEqual(str(unnamed_flipped.type_signature.member), '<bool,int32>')
    try:
      type_analysis.check_federated_type(
          unnamed_flipped.type_signature, placement=placement
      )
    except TypeError:
      self.fail(
          "Function 'check_federated_type' raised TypeError unexpectedly."
      )


class CreateFederatedAggregateTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = building_blocks.Reference('z', np.int32)
    accumulate_type = computation_types.StructType((np.int32, np.int32))
    accumulate_result = building_blocks.Reference('a', np.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((np.int32, np.int32))
    merge_result = building_blocks.Reference('m', np.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', np.int32)
    report = building_blocks.Lambda(
        report_ref.name, report_ref.type_signature, report_ref
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(
          None, zero, accumulate, merge, report
      )

  def test_raises_type_error_with_none_zero(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(0, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    accumulate_type = computation_types.StructType((np.int32, np.int32))
    accumulate_result = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((np.int32, np.int32))
    merge_result = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', np.int32)
    report = building_blocks.Lambda(
        report_ref.name, report_ref.type_signature, report_ref
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(
          value, None, accumulate, merge, report
      )

  def test_raises_type_error_with_none_accumulate(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(0, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    zero = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    merge_type = computation_types.StructType((np.int32, np.int32))
    merge_result = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', np.int32)
    report = building_blocks.Lambda(
        report_ref.name, report_ref.type_signature, report_ref
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(
          value, zero, None, merge, report
      )

  def test_raises_type_error_with_none_merge(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(0, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    zero = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    accumulate_type = computation_types.StructType((np.int32, np.int32))
    accumulate_result = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    report_ref = building_blocks.Reference('r', np.int32)
    report = building_blocks.Lambda(
        report_ref.name, report_ref.type_signature, report_ref
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(
          value, zero, accumulate, None, report
      )

  def test_raises_type_error_with_none_report(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(0, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    zero = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    accumulate_type = computation_types.StructType((np.int32, np.int32))
    accumulate_result = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((np.int32, np.int32))
    merge_result = building_blocks.Literal(
        3, computation_types.TensorType(np.int32)
    )
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(
          value, zero, accumulate, merge, None
      )

  def test_returns_federated_aggregate(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(0, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    zero = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    accumulate_type = computation_types.StructType((np.int32, np.int32))
    accumulate_result = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((np.int32, np.int32))
    merge_result = building_blocks.Literal(
        3, computation_types.TensorType(np.int32)
    )
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', np.int32)
    report = building_blocks.Lambda(
        report_ref.name, report_ref.type_signature, report_ref
    )
    comp = building_block_factory.create_federated_aggregate(
        value, zero, accumulate, merge, report
    )
    self.assertEqual(
        comp.compact_representation(),
        'federated_aggregate(<federated_value_at_clients(0),1,(x -> 2),(x ->'
        ' 3),(r -> r)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.SERVER,
    )
    comp = building_block_factory.create_federated_apply(fn, arg)
    self.assertEqual(
        comp.compact_representation(),
        'federated_apply(<(x -> x),federated_value_at_server(1)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedBroadcastTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_broadcast(None)

  def test_returns_federated_broadcast(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.SERVER,
    )
    comp = building_block_factory.create_federated_broadcast(value)
    self.assertEqual(
        comp.compact_representation(),
        'federated_broadcast(federated_value_at_server(1))',
    )
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedEvalTest(absltest.TestCase):

  def assert_type_error(self, fn, placement):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_eval(fn, placement)

  def test_raises_type_error_with_none_fn(self):
    self.assert_type_error(None, placements.CLIENTS)

  def test_raises_type_error_with_nonfunctional_fn(self):
    fn = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    self.assert_type_error(fn, placements.CLIENTS)

  def test_returns_federated_eval(self):
    fn = building_blocks.Reference(
        'y', computation_types.FunctionType(None, np.int32)
    )
    comp = building_block_factory.create_federated_eval(fn, placements.CLIENTS)
    self.assertEqual(
        comp.compact_representation(), 'federated_eval_at_clients(y)'
    )
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map(fn, None)

  def test_returns_federated_map(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_map(fn, arg)
    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(x -> x),federated_value_at_clients(1)>)',
    )
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMapAllEqualTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_all_equal(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_all_equal(fn, None)

  def test_returns_federated_map_all_equal(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_map_all_equal(fn, arg)
    self.assertEqual(
        comp.compact_representation(),
        'federated_map_all_equal(<(x -> x),federated_value_at_clients(1)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedMapOrApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_or_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_or_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.SERVER,
    )
    comp = building_block_factory.create_federated_map_or_apply(fn, arg)
    self.assertEqual(
        comp.compact_representation(),
        'federated_apply(<(x -> x),federated_value_at_server(1)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_map(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_map_or_apply(fn, arg)
    self.assertEqual(
        comp.compact_representation(),
        'federated_map_all_equal(<(x -> x),federated_value_at_clients(1)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedMeanTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_mean(None, None)

  def test_returns_federated_mean(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_mean(value, None)
    self.assertEqual(
        comp.compact_representation(),
        'federated_mean(federated_value_at_clients(1))',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_weighted_mean(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    weight = building_block_factory.create_federated_value(
        building_blocks.Literal(2, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_mean(value, weight)
    self.assertEqual(
        comp.compact_representation(),
        'federated_weighted_mean(<federated_value_at_clients(1),federated_value_at_clients(2)>)',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedMinTest(absltest.TestCase):

  def test_returns_federated_min(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_min(value)
    self.assertEqual(
        comp.compact_representation(),
        'federated_min(federated_value_at_clients(1))',
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class CreateFederatedMaxTest(absltest.TestCase):

  def test_returns_federated_max(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_max(value)
    self.assertEqual(
        comp.compact_representation(),
        'federated_max(federated_value_at_clients(1))',
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class CreateFederatedSecureModularSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    modulus = mock.create_autospec(
        building_blocks.CompiledComputation, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_modular_sum(None, modulus)

  def test_raises_type_error_with_none_modulus(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_modular_sum(value, None)

  def test_returns_federated_sum(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    modulus_type = computation_types.TensorType(np.int32)
    modulus = building_blocks.Literal(2, modulus_type)
    comp = building_block_factory.create_federated_secure_modular_sum(
        value, modulus
    )
    # Regex replaces compiled computations such as `comp#b03f` to ensure a
    # consistent output.
    golden.check_string(
        'federated_secure_modular_sum.expected',
        re.sub(
            r'comp\#\w*', 'some_compiled_comp', comp.formatted_representation()
        ),
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class CreateFederatedSecureSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    max_input = mock.create_autospec(
        building_blocks.CompiledComputation, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum(None, max_input)

  def test_raises_type_error_with_none_max_input(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum(value, None)

  def test_returns_federated_sum(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    max_value_type = computation_types.TensorType(np.int32)
    max_value = building_blocks.Literal(2, max_value_type)
    comp = building_block_factory.create_federated_secure_sum(value, max_value)
    self.assertEqual(
        comp.compact_representation(),
        'federated_secure_sum(<federated_value_at_clients(1),2>)',
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class CreateFederatedSecureSumBitwidthTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    bitwidth = mock.create_autospec(
        building_blocks.CompiledComputation, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum_bitwidth(
          None, bitwidth
      )

  def test_raises_type_error_with_none_bitwidth(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum_bitwidth(value, None)

  def test_returns_federated_sum(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    bitwidth_type = computation_types.TensorType(np.int32)
    bitwidth = building_blocks.Literal(2, bitwidth_type)
    comp = building_block_factory.create_federated_secure_sum_bitwidth(
        value, bitwidth
    )
    self.assertEqual(
        comp.compact_representation(),
        'federated_secure_sum_bitwidth(<federated_value_at_clients(1),2>)',
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class CreateFederatedSelectTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_secure', False, 'federated_select'),
      ('secure', True, 'federated_secure_select'),
  )
  def test_returns_federated_select(self, secure, name):
    client_keys = building_block_factory.create_federated_value(
        building_blocks.Literal(
            np.array([5, 4, 3, 2, 1]),
            computation_types.TensorType(np.int32, [5]),
        ),
        placement=placements.CLIENTS,
    )
    max_key = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.SERVER,
    )
    server_val_type = computation_types.SequenceType(np.str_)
    server_val = building_blocks.Reference(
        'server_val',
        computation_types.FederatedType(server_val_type, placements.SERVER),
    )
    select_fn = building_blocks.Reference(
        'select_fn',
        computation_types.FunctionType(
            computation_types.StructType([
                ('some_name_for_server_val', server_val_type),
                ('some_name_for_key', np.int32),
            ]),
            np.str_,
        ),
    )
    comp = building_block_factory.create_federated_select(
        client_keys, max_key, server_val, select_fn, secure
    )
    self.assertEqual(
        comp.compact_representation(),
        f'{name}(<federated_value_at_clients([5 4 3 2'
        ' 1]),federated_value_at_server(1),server_val,(a -> select_fn(a))>)',
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), '{str*}@CLIENTS'
    )


class CreateFederatedSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_sum(None)

  def test_returns_federated_sum(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    comp = building_block_factory.create_federated_sum(value)
    self.assertEqual(
        comp.compact_representation(),
        'federated_sum(federated_value_at_clients(1))',
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedUnzipTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_unzip(None)

  def test_returns_tuple_federated_map_with_empty_value(self):
    value_type = computation_types.FederatedType([], placements.CLIENTS)
    value = building_blocks.Reference('v', value_type)
    with self.assertRaises(ValueError):
      building_block_factory.create_federated_unzip(value)

  def test_returns_tuple_federated_map_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType(
        (np.int32,), placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_map(<(arg -> arg[0]),value>)>)',
    )
    self.assertEqual(str(comp.type_signature), '<{int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_one_value_named(self):
    type_signature = computation_types.StructType((('a', np.int32),))
    value_type = computation_types.FederatedType(
        type_signature, placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_map(<(arg -> arg[0]),value>)>)',
    )
    self.assertEqual(str(comp.type_signature), '<a={int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType(
        (np.int32, np.int32), placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_map_with_two_values_unnamed.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(
        str(comp.type_signature), '<{int32}@CLIENTS,{int32}@CLIENTS>'
    )

  def test_returns_tuple_federated_map_with_two_values_named(self):
    type_signature = computation_types.StructType(
        (('a', np.int32), ('b', np.int32))
    )
    value_type = computation_types.FederatedType(
        type_signature, placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_map_with_two_values_named.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(
        str(comp.type_signature), '<a={int32}@CLIENTS,b={int32}@CLIENTS>'
    )

  def test_returns_tuple_federated_map_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType(
        (np.int32, np.bool_), placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_map_with_two_values_different_typed.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(
        str(comp.type_signature), '<{int32}@CLIENTS,{bool}@CLIENTS>'
    )

  def test_returns_tuple_federated_apply_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType((np.int32,), placements.SERVER)
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_apply(<(arg -> arg[0]),value>)>)',
    )
    self.assertEqual(str(comp.type_signature), '<int32@SERVER>')

  def test_returns_tuple_federated_apply_with_one_value_named(self):
    type_signature = computation_types.StructType((('a', np.int32),))
    value_type = computation_types.FederatedType(
        type_signature, placements.SERVER
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_apply(<(arg -> arg[0]),value>)>)',
    )
    self.assertEqual(str(comp.type_signature), '<a=int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType(
        (np.int32, np.int32), placements.SERVER
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_apply_with_two_values_unnamed.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_named(self):
    type_signature = computation_types.StructType(
        (('a', np.int32), ('b', np.int32))
    )
    value_type = computation_types.FederatedType(
        type_signature, placements.SERVER
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_apply_with_two_values_named.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(
        str(comp.type_signature), '<a=int32@SERVER,b=int32@SERVER>'
    )

  def test_returns_tuple_federated_apply_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType(
        (np.int32, np.bool_), placements.SERVER
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_apply_with_two_values_different_typed.expected',
        comp.formatted_representation(),
    )
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,bool@SERVER>')


class CreateFederatedValueTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(None, placements.CLIENTS)

  def test_raises_type_error_with_none_placement(self):
    value = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(value, None)

  def test_raises_type_error_with_unknown_placement(self):
    value = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(value, 'unknown')

  def test_returns_federated_value_at_clients(self):
    value = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    comp = building_block_factory.create_federated_value(
        value, placements.CLIENTS
    )
    self.assertEqual(
        comp.compact_representation(), 'federated_value_at_clients(1)'
    )
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')

  def test_returns_federated_value_at_server(self):
    value = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    comp = building_block_factory.create_federated_value(
        value, placements.SERVER
    )
    self.assertEqual(
        comp.compact_representation(), 'federated_value_at_server(1)'
    )
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


INT_AT_CLIENTS = computation_types.FederatedType(np.int32, placements.CLIENTS)
BOOL_AT_CLIENTS = computation_types.FederatedType(np.bool_, placements.CLIENTS)
INT_AT_SERVER = computation_types.FederatedType(np.int32, placements.SERVER)
BOOL_AT_SERVER = computation_types.FederatedType(np.bool_, placements.SERVER)


class CreateFederatedZipTest(parameterized.TestCase, absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaisesRegex(TypeError, 'found NoneType'):
      building_block_factory.create_federated_zip(None)

  def test_raises_type_error_with_empty_value(self):
    value_type = computation_types.StructType([])
    value = building_blocks.Reference('v', value_type)
    with self.assertRaisesRegex(TypeError, 'at least one FederatedType'):
      building_block_factory.create_federated_zip(value)

  @parameterized.named_parameters([
      (
          'one_unnamed',
          computation_types.StructType((INT_AT_CLIENTS,)),
          computation_types.StructType((np.int32,)),
      ),
      (
          'one_named',
          computation_types.StructType((('a', INT_AT_CLIENTS),)),
          computation_types.StructType((('a', np.int32),)),
      ),
      (
          'two_unnamed',
          computation_types.StructType((INT_AT_CLIENTS,) * 2),
          computation_types.StructType((np.int32,) * 2),
      ),
      (
          'two_named',
          computation_types.StructType(
              (('a', INT_AT_CLIENTS), ('b', INT_AT_CLIENTS))
          ),
          computation_types.StructType((('a', np.int32), ('b', np.int32))),
      ),
      (
          'different_typed',
          computation_types.StructType((BOOL_AT_CLIENTS, INT_AT_CLIENTS)),
          computation_types.StructType((np.bool_, np.int32)),
      ),
      ('three_tuple', (INT_AT_CLIENTS,) * 3, (np.int32, np.int32, np.int32)),
      (
          'three_dict',
          collections.OrderedDict(
              a=INT_AT_CLIENTS, b=INT_AT_CLIENTS, c=BOOL_AT_CLIENTS
          ),
          collections.OrderedDict(a=np.int32, b=np.int32, c=np.bool_),
      ),
  ])
  def test_returns_zip_at_clients(self, value_type, expected_zipped_type):
    value_type = computation_types.to_type(value_type)
    expected_zipped_type = computation_types.FederatedType(
        expected_zipped_type, placements.CLIENTS
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(
        comp.formatted_representation(), 'federated_zip_at_clients(v)'
    )
    type_test_utils.assert_types_equivalent(
        expected_zipped_type, comp.type_signature
    )

  @parameterized.named_parameters([
      (
          'one_unnamed',
          computation_types.StructType((INT_AT_SERVER,)),
          computation_types.StructType((np.int32,)),
      ),
      (
          'one_named',
          computation_types.StructType((('a', INT_AT_SERVER),)),
          computation_types.StructType((('a', np.int32),)),
      ),
      (
          'two_unnamed',
          computation_types.StructType((INT_AT_SERVER,) * 2),
          computation_types.StructType((np.int32,) * 2),
      ),
      (
          'two_named',
          computation_types.StructType(
              (('a', INT_AT_SERVER), ('b', INT_AT_SERVER))
          ),
          computation_types.StructType((('a', np.int32), ('b', np.int32))),
      ),
      (
          'different_typed',
          computation_types.StructType((BOOL_AT_SERVER, INT_AT_SERVER)),
          computation_types.StructType((np.bool_, np.int32)),
      ),
      ('three_tuple', (INT_AT_SERVER,) * 3, (np.int32, np.int32, np.int32)),
      (
          'three_dict',
          collections.OrderedDict(
              a=INT_AT_SERVER, b=INT_AT_SERVER, c=BOOL_AT_SERVER
          ),
          collections.OrderedDict(a=np.int32, b=np.int32, c=np.bool_),
      ),
  ])
  def test_returns_zip_at_server(self, value_type, expected_zipped_type):
    value_type = computation_types.to_type(value_type)
    expected_zipped_type = computation_types.FederatedType(
        expected_zipped_type, placements.SERVER
    )
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(
        comp.formatted_representation(), 'federated_zip_at_server(v)'
    )
    type_test_utils.assert_types_equivalent(
        expected_zipped_type, comp.type_signature
    )

  def test_flat_raises_type_error_with_inconsistent_placement(self):
    client_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    server_type = computation_types.FederatedType(
        np.int32, placements.SERVER, all_equal=True
    )
    value_type = computation_types.StructType(
        [('a', client_type), ('b', server_type)]
    )
    value = building_blocks.Reference('v', value_type)
    self.assertEqual(
        value.type_signature.compact_representation(),
        '<a=int32@CLIENTS,b=int32@SERVER>',
    )
    with self.assertRaisesRegex(TypeError, 'same placement'):
      building_block_factory.create_federated_zip(value)

  def test_nested_raises_type_error_with_inconsistent_placement(self):
    client_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    server_type = computation_types.FederatedType(
        np.int32, placements.SERVER, all_equal=True
    )
    tuple_type = computation_types.StructType(
        [('c', server_type), ('d', server_type)]
    )
    value_type = computation_types.StructType(
        [('a', client_type), ('b', tuple_type)]
    )
    value = building_blocks.Reference('v', value_type)
    self.assertEqual(
        value.type_signature.compact_representation(),
        '<a=int32@CLIENTS,b=<c=int32@SERVER,d=int32@SERVER>>',
    )
    with self.assertRaisesRegex(TypeError, 'same placement'):
      building_block_factory.create_federated_zip(value)

  def test_flat_raises_type_error_with_unplaced(self):
    client_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    value_type = computation_types.StructType(
        [('a', client_type), ('b', np.int32)]
    )
    value = building_blocks.Reference('v', value_type)
    self.assertEqual(
        value.type_signature.compact_representation(),
        '<a=int32@CLIENTS,b=int32>',
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_zip(value)

  def test_nested_raises_type_error_with_unplaced(self):
    client_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    tuple_type = computation_types.StructType(
        [('c', np.int32), ('d', np.int32)]
    )
    value_type = computation_types.StructType(
        [('a', client_type), ('b', tuple_type)]
    )
    value = building_blocks.Reference('v', value_type)
    self.assertEqual(
        value.type_signature.compact_representation(),
        '<a=int32@CLIENTS,b=<c=int32,d=int32>>',
    )
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_zip(value)


class CreateSequenceMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg_type = computation_types.SequenceType(np.int32)
    arg = building_blocks.Reference('y', arg_type)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_map(fn, None)

  def test_returns_sequence_map(self):
    ref = building_blocks.Reference('x', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.SequenceType(np.int32)
    arg = building_blocks.Reference('y', arg_type)
    comp = building_block_factory.create_sequence_map(fn, arg)
    self.assertEqual(
        comp.compact_representation(), 'sequence_map(<(x -> x),y>)'
    )
    self.assertEqual(str(comp.type_signature), 'int32*')


class CreateSequenceReduceTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = building_blocks.Reference('z', np.int32)
    op_type = computation_types.StructType((np.int32, np.int32))
    op_result = building_blocks.Reference('o', np.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.SequenceType(np.int32)
    value = building_blocks.Reference('v', value_type)
    op_type = computation_types.StructType((np.int32, np.int32))
    op_result = building_blocks.Reference('o', np.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(value, None, op)

  def test_raises_type_error_with_none_op(self):
    value_type = computation_types.SequenceType(np.int32)
    value = building_blocks.Reference('v', value_type)
    zero = building_blocks.Reference('z', np.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(value, zero, None)

  def test_returns_sequence_reduce(self):
    value_type = computation_types.SequenceType(np.int32)
    value = building_blocks.Reference('v', value_type)
    zero = building_blocks.Reference('z', np.int32)
    op_type = computation_types.StructType((np.int32, np.int32))
    op_result = building_blocks.Reference('o', np.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    comp = building_block_factory.create_sequence_reduce(value, zero, op)
    self.assertEqual(
        comp.compact_representation(), 'sequence_reduce(<v,z,(x -> o)>)'
    )
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateSequenceSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_sum(None)

  def test_returns_federated_sum(self):
    value_type = computation_types.SequenceType(np.int32)
    value = building_blocks.Reference('v', value_type)
    comp = building_block_factory.create_sequence_sum(value)
    self.assertEqual(comp.compact_representation(), 'sequence_sum(v)')
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateNamedTupleTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(None, ('a',))

  def test_raises_type_error_with_wrong_comp_type(self):
    comp = building_blocks.Reference('data', np.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, ('a',))

  def test_raises_type_error_with_wrong_names_type_string(self):
    type_signature = computation_types.StructType((np.int32, np.int32))
    comp = building_blocks.Reference('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_type_error_with_wrong_names_type_ints(self):
    type_signature = computation_types.StructType((np.int32, np.int32))
    comp = building_blocks.Reference('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_value_error_with_wrong_lengths(self):
    type_signature = computation_types.StructType((np.int32, np.int32))
    comp = building_blocks.Reference('data', type_signature)
    with self.assertRaises(ValueError):
      building_block_factory.create_named_tuple(comp, ('a',))

  def test_creates_named_tuple_from_unamed_tuple(self):
    type_signature = computation_types.StructType((np.int32, np.int32))
    comp = building_blocks.Reference('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('a', 'b'))
    expected_type_signature = computation_types.StructType(
        (('a', np.int32), ('b', np.int32))
    )
    self.assertEqual(named_comp.type_signature, expected_type_signature)

  def test_creates_named_tuple_from_named_tuple(self):
    type_signature = computation_types.StructType(
        (('a', np.int32), ('b', np.int32))
    )
    comp = building_blocks.Reference('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('c', 'd'))
    expected_type_signature = computation_types.StructType(
        (('c', np.int32), ('d', np.int32))
    )
    self.assertEqual(named_comp.type_signature, expected_type_signature)


def identity_for_type(
    input_type: computation_types.Type,
) -> building_blocks.Lambda:
  """Returns an identity computation for the provided `input_type`."""
  return building_blocks.Lambda(
      'x', input_type, building_blocks.Reference('x', input_type)
  )


class SelectOutputFromLambdaTest(absltest.TestCase):

  def test_raises_on_non_str_int_index(self):
    lam = identity_for_type(computation_types.StructType([np.int32]))
    with self.assertRaisesRegex(TypeError, 'Invalid selection type'):
      building_block_factory.select_output_from_lambda(lam, [dict()])

  def test_selects_single_output(self):
    input_type = computation_types.StructType([np.int32, np.float32])
    lam = identity_for_type(input_type)
    zero_selected = building_block_factory.select_output_from_lambda(lam, 0)
    type_test_utils.assert_types_equivalent(
        zero_selected.type_signature.parameter, lam.type_signature.parameter
    )
    type_test_utils.assert_types_equivalent(
        zero_selected.type_signature.result, lam.type_signature.result[0]
    )
    self.assertEqual(str(zero_selected), '(x -> x[0])')

  def test_selects_single_output_by_str(self):
    input_type = computation_types.StructType([('a', np.int32)])
    lam = identity_for_type(input_type)
    selected = building_block_factory.select_output_from_lambda(lam, 'a')
    type_test_utils.assert_types_equivalent(
        selected.type_signature,
        computation_types.FunctionType(
            lam.parameter_type, lam.type_signature.result['a']
        ),
    )

  def test_selects_from_struct_by_removing_struct_wrapper(self):
    lam = building_blocks.Lambda(
        'x',
        np.int32,
        building_blocks.Struct([building_blocks.Reference('x', np.int32)]),
    )
    selected = building_block_factory.select_output_from_lambda(lam, 0)
    type_test_utils.assert_types_equivalent(
        selected.type_signature.result, computation_types.TensorType(np.int32)
    )
    self.assertEqual(str(selected), '(x -> x)')

  def test_selects_struct_of_outputs(self):
    input_type = computation_types.StructType([np.int32, np.int64, np.float32])
    lam = identity_for_type(input_type)
    tuple_selected = building_block_factory.select_output_from_lambda(
        lam, [0, 1]
    )
    type_test_utils.assert_types_equivalent(
        tuple_selected.type_signature.parameter, lam.type_signature.parameter
    )
    type_test_utils.assert_types_equivalent(
        tuple_selected.type_signature.result,
        computation_types.StructType(
            [lam.type_signature.result[0], lam.type_signature.result[1]]
        ),
    )
    self.assertEqual(
        str(tuple_selected), '(x -> (let _var1=x in <_var1[0],_var1[1]>))'
    )

  def test_selects_struct_of_outputs_by_str_name(self):
    input_type = computation_types.StructType(
        [('a', np.int32), ('b', np.int64), ('c', np.float32)]
    )
    lam = identity_for_type(input_type)
    selected = building_block_factory.select_output_from_lambda(lam, ['a', 'b'])
    type_test_utils.assert_types_equivalent(
        selected.type_signature,
        computation_types.FunctionType(
            lam.parameter_type,
            computation_types.StructType(
                [lam.type_signature.result.a, lam.type_signature.result.b]
            ),
        ),
    )

  def test_selects_nested_federated_outputs(self):
    input_type = computation_types.StructType([
        ('a', computation_types.StructType([('inner', np.int32)])),
        ('b', np.int32),
    ])
    lam = identity_for_type(input_type)
    tuple_selected = building_block_factory.select_output_from_lambda(
        lam, [('a', 'inner'), 'b']
    )
    type_test_utils.assert_types_equivalent(
        tuple_selected.type_signature.parameter, lam.type_signature.parameter
    )
    type_test_utils.assert_types_equivalent(
        tuple_selected.type_signature.result,
        computation_types.StructType(
            [lam.type_signature.result.a.inner, lam.type_signature.result.b]
        ),
    )
    self.assertEqual(
        str(tuple_selected), '(x -> (let _var1=x in <_var1.a.inner,_var1.b>))'
    )


class ZipUpToTest(absltest.TestCase):

  def test_zips_struct_of_federated_values(self):
    comp = building_blocks.Struct([
        building_blocks.Reference(
            'x',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
        building_blocks.Reference(
            'y',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
    ])
    zippable_type = computation_types.FederatedType(
        computation_types.StructType([(None, np.int32), (None, np.int32)]),
        placements.CLIENTS,
    )
    zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=comp, target_type=zippable_type
    )
    type_test_utils.assert_types_equivalent(
        zipped.type_signature, zippable_type
    )

  def test_does_not_zip_different_placement_target(self):
    comp = building_blocks.Struct([
        building_blocks.Reference(
            'x',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
        building_blocks.Reference(
            'y',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
    ])
    non_zippable_type = computation_types.FederatedType(
        computation_types.StructType([(None, np.int32), (None, np.int32)]),
        placements.SERVER,
    )
    zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=comp, target_type=non_zippable_type
    )
    self.assertIsNone(zipped)

  def test_zips_struct_of_federated_values_under_struct(self):
    comp = building_blocks.Struct(
        [
            building_blocks.Struct([
                building_blocks.Reference(
                    'x',
                    computation_types.FederatedType(
                        np.int32, placements.CLIENTS
                    ),
                ),
                building_blocks.Reference(
                    'y',
                    computation_types.FederatedType(
                        np.int32, placements.CLIENTS
                    ),
                ),
            ])
        ]
    )
    zippable_type = computation_types.StructType([(
        None,
        computation_types.FederatedType(
            computation_types.StructType([(None, np.int32), (None, np.int32)]),
            placements.CLIENTS,
        ),
    )])
    zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=comp, target_type=zippable_type
    )
    type_test_utils.assert_types_equivalent(
        zipped.type_signature, zippable_type
    )

  def test_assignability_with_names(self):
    # This would correspond to an implicit downcast in TFF's typesystem; the
    # result would not be assignable to the requested type.
    comp = building_blocks.Struct(
        [
            building_blocks.Struct([
                (
                    'a',
                    building_blocks.Reference(
                        'x',
                        computation_types.FederatedType(
                            np.int32, placements.CLIENTS
                        ),
                    ),
                ),
                (
                    'b',
                    building_blocks.Reference(
                        'y',
                        computation_types.FederatedType(
                            np.int32, placements.CLIENTS
                        ),
                    ),
                ),
            ])
        ]
    )
    unnamed_zippable_type = computation_types.StructType([(
        None,
        computation_types.FederatedType(
            computation_types.StructType([(None, np.int32), (None, np.int32)]),
            placements.CLIENTS,
        ),
    )])
    named_zippable_type = computation_types.StructType([(
        None,
        computation_types.FederatedType(
            computation_types.StructType([('a', np.int32), ('b', np.int32)]),
            placements.CLIENTS,
        ),
    )])

    not_zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=comp, target_type=unnamed_zippable_type
    )
    zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=comp, target_type=named_zippable_type
    )

    self.assertFalse(
        unnamed_zippable_type.is_assignable_from(named_zippable_type)
    )

    self.assertIsNone(not_zipped)

    type_test_utils.assert_types_equivalent(
        zipped.type_signature, named_zippable_type
    )

  def test_does_not_zip_under_function(self):
    result_comp = building_blocks.Struct([
        building_blocks.Reference(
            'x',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
        building_blocks.Reference(
            'y',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
    ])
    lam = building_blocks.Lambda(None, None, result_comp)
    zippable_function_type = computation_types.FunctionType(
        None,
        computation_types.FederatedType(
            computation_types.StructType([(None, np.int32), (None, np.int32)]),
            placements.CLIENTS,
        ),
    )

    zipped = building_block_factory.zip_to_match_type(
        comp_to_zip=lam, target_type=zippable_function_type
    )

    self.assertIsNone(zipped)


if __name__ == '__main__':
  absltest.main()
