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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tree

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import data_type_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import array
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization


_TEST_LITERAL = building_blocks.Literal(
    1, computation_types.TensorType(np.int32)
)


def _to_python(obj):
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    return obj


class ComputationBuildingBlocksTest(absltest.TestCase):

  def test_basic_functionality_of_reference_class(self):
    x = building_blocks.Reference('foo', np.int32)
    self.assertEqual(x.name, 'foo')
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), "Reference('foo', TensorType(np.int32))")
    self.assertEqual(x.compact_representation(), 'foo')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'reference')
    self.assertEqual(x_proto.reference.name, x.name)
    self._serialize_deserialize_roundtrip_test(x)

  def test_reference_children_is_empty(self):
    ref = building_blocks.Reference('foo', np.int32)
    self.assertEqual([], list(ref.children()))

  def test_basic_functionality_of_selection_class(self):
    x = building_blocks.Reference('foo', [('bar', np.int32), ('baz', np.bool_)])
    y = building_blocks.Selection(x, name='bar')
    self.assertEqual(y.name, 'bar')
    self.assertIsNone(y.index)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(
        repr(y),
        (
            "Selection(Reference('foo', StructType(["
            "('bar', TensorType(np.int32)), ('baz', TensorType(np.bool_))]))"
            ", name='bar')"
        ),
    )
    self.assertEqual(y.compact_representation(), 'foo.bar')
    z = building_blocks.Selection(x, name='baz')
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(z.compact_representation(), 'foo.baz')
    with self.assertRaises(ValueError):
      _ = building_blocks.Selection(x, name='bak')
    x0 = building_blocks.Selection(x, index=0)
    self.assertIsNone(x0.name)
    self.assertEqual(x0.index, 0)
    self.assertEqual(str(x0.type_signature), 'int32')
    self.assertEqual(
        repr(x0),
        (
            "Selection(Reference('foo', StructType(["
            "('bar', TensorType(np.int32)), ('baz', TensorType(np.bool_))]))"
            ', index=0)'
        ),
    )
    self.assertEqual(x0.compact_representation(), 'foo[0]')
    x1 = building_blocks.Selection(x, index=1)
    self.assertEqual(str(x1.type_signature), 'bool')
    self.assertEqual(x1.compact_representation(), 'foo[1]')
    with self.assertRaises(ValueError):
      _ = building_blocks.Selection(x, index=2)
    with self.assertRaises(ValueError):
      _ = building_blocks.Selection(x, index=-1)
    y_proto = y.proto
    self.assertEqual(
        type_serialization.deserialize_type(y_proto.type), y.type_signature
    )
    self.assertEqual(y_proto.WhichOneof('computation'), 'selection')
    self.assertEqual(str(y_proto.selection.source), str(x.proto))
    # Our serialized representation only uses indices.
    self.assertEqual(y_proto.selection.index, 0)
    self._serialize_deserialize_roundtrip_test(y)
    self._serialize_deserialize_roundtrip_test(z)
    self._serialize_deserialize_roundtrip_test(x0)
    self._serialize_deserialize_roundtrip_test(x1)

  def test_reference_children_yields_source(self):
    source = building_blocks.Reference('foo', (np.int32, np.int32))
    selection = building_blocks.Selection(source, index=1)
    self.assertEqual([source], list(selection.children()))

  def test_basic_functionality_of_struct_class(self):
    x = building_blocks.Reference('foo', np.int32)
    y = building_blocks.Reference('bar', np.bool_)
    z = building_blocks.Struct([x, ('y', y)])
    with self.assertRaises(ValueError):
      _ = building_blocks.Struct([('', y)])
    self.assertIsInstance(z, structure.Struct)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(
        repr(z),
        (
            "Struct([(None, Reference('foo', TensorType(np.int32))), ('y', "
            "Reference('bar', TensorType(np.bool_)))])"
        ),
    )
    self.assertEqual(z.compact_representation(), '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertIs(z.y, y)
    self.assertLen(z, 2)
    self.assertIs(z[0], x)
    self.assertIs(z[1], y)
    self.assertEqual(
        ','.join(e.compact_representation() for e in iter(z)), 'foo,bar'
    )
    z_proto = z.proto
    self.assertEqual(
        type_serialization.deserialize_type(z_proto.type), z.type_signature
    )
    self.assertEqual(z_proto.WhichOneof('computation'), 'struct')
    self.assertEqual([e.name for e in z_proto.struct.element], ['', 'y'])
    self._serialize_deserialize_roundtrip_test(z)

  def test_struct_children_yields_elements(self):
    e1 = building_blocks.Reference('a', np.int32)
    e2 = building_blocks.Reference('b', np.int32)
    struct_ = building_blocks.Struct([(None, e1), (None, e2)])
    self.assertEqual([e1, e2], list(struct_.children()))

  def test_struct_with_container_type(self):
    x = building_blocks.Reference('foo', np.int32)
    y = building_blocks.Reference('bar', np.bool_)
    z = building_blocks.Struct([x, ('y', y)], tuple)
    self.assertEqual(
        z.type_signature,
        computation_types.StructWithPythonType(
            [np.int32, ('y', np.bool_)], tuple
        ),
    )

  def test_basic_functionality_of_call_class(self):
    x = building_blocks.Reference(
        'foo', computation_types.FunctionType(np.int32, np.bool_)
    )
    y = building_blocks.Reference('bar', np.int32)
    z = building_blocks.Call(x, y)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertIs(z.function, x)
    self.assertIs(z.argument, y)
    self.assertEqual(
        repr(z),
        (
            "Call(Reference('foo', "
            'FunctionType(TensorType(np.int32), TensorType(np.bool_))), '
            "Reference('bar', TensorType(np.int32)))"
        ),
    )
    self.assertEqual(z.compact_representation(), 'foo(bar)')
    with self.assertRaises(TypeError):
      building_blocks.Call(x)
    w = building_blocks.Reference('bak', np.float32)
    with self.assertRaises(TypeError):
      building_blocks.Call(x, w)
    z_proto = z.proto
    self.assertEqual(
        type_serialization.deserialize_type(z_proto.type), z.type_signature
    )
    self.assertEqual(z_proto.WhichOneof('computation'), 'call')
    self.assertEqual(str(z_proto.call.function), str(x.proto))
    self.assertEqual(str(z_proto.call.argument), str(y.proto))
    self._serialize_deserialize_roundtrip_test(z)

  def test_call_children_with_no_arg_yields_function(self):
    fn = building_blocks.Reference(
        'a', computation_types.FunctionType(None, np.int32)
    )
    call = building_blocks.Call(fn)
    self.assertEqual([fn], list(call.children()))

  def test_call_children_with_arg_yields_function_and_arg(self):
    fn = building_blocks.Reference(
        'a', computation_types.FunctionType(np.int32, np.int32)
    )
    arg = building_blocks.Reference('b', np.int32)
    call = building_blocks.Call(fn, arg)
    self.assertEqual([fn, arg], list(call.children()))

  def test_basic_functionality_of_lambda_class(self):
    arg_name = 'arg'
    arg_type = [
        ('f', computation_types.FunctionType(np.int32, np.int32)),
        ('x', np.int32),
    ]
    arg = building_blocks.Reference(arg_name, arg_type)
    arg_f = building_blocks.Selection(arg, name='f')
    arg_x = building_blocks.Selection(arg, name='x')
    x = building_blocks.Lambda(
        arg_name,
        arg_type,
        building_blocks.Call(arg_f, building_blocks.Call(arg_f, arg_x)),
    )
    self.assertEqual(
        str(x.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)'
    )
    self.assertEqual(x.parameter_name, arg_name)
    self.assertEqual(str(x.parameter_type), '<f=(int32 -> int32),x=int32>')
    self.assertEqual(x.result.compact_representation(), 'arg.f(arg.f(arg.x))')
    arg_type_repr = (
        'StructType(['
        "('f', FunctionType(TensorType(np.int32), TensorType(np.int32))), "
        "('x', TensorType(np.int32))])"
    )
    self.assertEqual(
        repr(x),
        "Lambda('arg', {0}, "
        "Call(Selection(Reference('arg', {0}), name='f'), "
        "Call(Selection(Reference('arg', {0}), name='f'), "
        "Selection(Reference('arg', {0}), name='x'))))".format(arg_type_repr),
    )
    self.assertEqual(x.compact_representation(), '(arg -> arg.f(arg.f(arg.x)))')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'lambda')
    self.assertEqual(getattr(x_proto, 'lambda').parameter_name, arg_name)
    self.assertEqual(
        str(getattr(x_proto, 'lambda').result), str(x.result.proto)
    )
    self._serialize_deserialize_roundtrip_test(x)

  def test_lambda_children_returns_result(self):
    result = building_blocks.Reference('a', np.int32)
    lambda_ = building_blocks.Lambda('a', np.int32, result)
    self.assertEqual([result], list(lambda_.children()))

  def test_basic_functionality_of_block_class(self):
    x = building_blocks.Block(
        [
            ('x', building_blocks.Reference('arg', (np.int32, np.int32))),
            (
                'y',
                building_blocks.Selection(
                    building_blocks.Reference('x', (np.int32, np.int32)),
                    index=0,
                ),
            ),
        ],
        building_blocks.Reference('y', np.int32),
    )
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(
        [(k, v.compact_representation()) for k, v in x.locals],
        [('x', 'arg'), ('y', 'x[0]')],
    )
    self.assertEqual(x.result.compact_representation(), 'y')
    self.assertEqual(
        repr(x),
        (
            "Block([('x', Reference('arg', StructType([TensorType(np.int32),"
            " TensorType(np.int32)]) as tuple)), ('y', Selection(Reference('x',"
            ' StructType([TensorType(np.int32), TensorType(np.int32)]) as'
            " tuple), index=0))], Reference('y', TensorType(np.int32)))"
        ),
    )
    self.assertEqual(x.compact_representation(), '(let x=arg,y=x[0] in y)')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'block')
    self.assertEqual(str(x_proto.block.result), str(x.result.proto))
    for idx, loc_proto in enumerate(x_proto.block.local):
      loc_name, loc_value = x.locals[idx]
      self.assertEqual(loc_proto.name, loc_name)
      self.assertEqual(str(loc_proto.value), str(loc_value.proto))
      self._serialize_deserialize_roundtrip_test(x)

  def test_block_children_returns_locals_then_result(self):
    l1 = building_blocks.Reference('a', np.int32)
    l2 = building_blocks.Reference('b', np.int32)
    result = building_blocks.Reference('c', np.int32)
    block = building_blocks.Block([('1', l1), ('2', l2)], result)
    self.assertEqual([l1, l2, result], list(block.children()))

  def test_basic_functionality_of_intrinsic_class(self):
    x = building_blocks.Intrinsic(
        'add_one', computation_types.FunctionType(np.int32, np.int32)
    )
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(x.uri, 'add_one')
    self.assertEqual(
        repr(x),
        (
            "Intrinsic('add_one', "
            'FunctionType(TensorType(np.int32), TensorType(np.int32)))'
        ),
    )
    self.assertEqual(x.compact_representation(), 'add_one')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(x_proto.intrinsic.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_intrinsic_children_is_empty(self):
    intrinsic = building_blocks.Intrinsic(
        'a', computation_types.FunctionType(np.int32, np.int32)
    )
    self.assertEqual([], list(intrinsic.children()))

  def test_basic_intrinsic_functionality_plus_canonical_typecheck(self):
    x = building_blocks.Intrinsic(
        'generic_plus',
        computation_types.FunctionType([np.int32, np.int32], np.int32),
    )
    self.assertEqual(str(x.type_signature), '(<int32,int32> -> int32)')
    self.assertEqual(x.uri, 'generic_plus')
    self.assertEqual(x.compact_representation(), 'generic_plus')
    x_proto = x.proto
    deserialized_type = type_serialization.deserialize_type(x_proto.type)
    x.type_signature.check_assignable_from(deserialized_type)
    self.assertEqual(x_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(x_proto.intrinsic.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_intrinsic_class_fails_bad_type(self):
    with self.assertRaises(TypeError):
      _ = building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          computation_types.FunctionType([np.int32, np.int32], np.float32),
      )

  def test_intrinsic_class_fails_struct_type_with_names(self):
    with self.assertRaises(TypeError):
      _ = building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          computation_types.FunctionType(
              [('a', np.int32), ('b', np.int32)], np.int32
          ),
      )

  def test_intrinsic_class_succeeds_simple_federated_map(self):
    simple_function = computation_types.FunctionType(np.int32, np.float32)
    federated_arg = computation_types.FederatedType(
        simple_function.parameter, placements.CLIENTS
    )
    federated_result = computation_types.FederatedType(
        simple_function.result, placements.CLIENTS
    )
    federated_map_concrete_type = computation_types.FunctionType(
        computation_types.StructType((simple_function, federated_arg)),
        federated_result,
    )
    concrete_federated_map = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri, federated_map_concrete_type
    )
    self.assertIsInstance(concrete_federated_map, building_blocks.Intrinsic)
    self.assertEqual(
        str(concrete_federated_map.type_signature),
        '(<(int32 -> float32),{int32}@CLIENTS> -> {float32}@CLIENTS)',
    )
    self.assertEqual(concrete_federated_map.uri, 'federated_map')
    self.assertEqual(
        concrete_federated_map.compact_representation(), 'federated_map'
    )
    concrete_federated_map_proto = concrete_federated_map.proto
    self.assertEqual(
        type_serialization.deserialize_type(concrete_federated_map_proto.type),
        concrete_federated_map.type_signature,
    )
    self.assertEqual(
        concrete_federated_map_proto.WhichOneof('computation'), 'intrinsic'
    )
    self.assertEqual(
        concrete_federated_map_proto.intrinsic.uri, concrete_federated_map.uri
    )
    self._serialize_deserialize_roundtrip_test(concrete_federated_map)

  def test_basic_functionality_of_data_class(self):
    test_proto = array.to_proto(np.array([1, 2, 3], np.int32))
    any_proto = any_pb2.Any()
    any_proto.Pack(test_proto)
    x = building_blocks.Data(
        any_proto, computation_types.SequenceType(np.int32)
    )
    self.assertEqual(str(x.type_signature), 'int32*')
    self.assertEqual(x.content, any_proto)
    arr = array_pb2.Array()
    x.content.Unpack(arr)
    self.assertEqual(arr, test_proto)
    as_string = str(id(any_proto))
    self.assertEqual(
        repr(x), f'Data({as_string}, SequenceType(TensorType(np.int32)))'
    )
    self.assertEqual(
        x.compact_representation(),
        as_string,
    )
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'data')
    self.assertEqual(x_proto.data.content, x.content)
    self._serialize_deserialize_roundtrip_test(x)

  def test_data_children_is_empty(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array(1, np.int32)
    )
    data = building_blocks.Data(any_proto, np.int32)
    self.assertEqual([], list(data.children()))

  def test_basic_functionality_of_compiled_computation_class(self):
    type_spec = computation_types.TensorType(np.int32)
    x_proto = computation_factory.create_lambda_identity(type_spec)
    x_type = computation_types.FunctionType(type_spec, type_spec)
    x = building_blocks.CompiledComputation(
        x_proto, name='a', type_signature=x_type
    )
    self.assertEqual(
        x.type_signature.compact_representation(), '(int32 -> int32)'
    )
    self.assertIsInstance(x.proto, computation_pb2.Computation)
    self.assertEqual(x.name, 'a')
    self.assertTrue(
        repr(x),
        (
            "CompiledComputation('a', FunctionType(TensorType(np.int32),"
            ' TensorType(np.int32)))'
        ),
    )
    self.assertTrue(x.compact_representation(), 'comp#a')
    y_proto = computation_factory.create_lambda_identity(type_spec)
    y_type = computation_types.FunctionType(type_spec, type_spec)
    y = building_blocks.CompiledComputation(
        y_proto, name='a', type_signature=y_type
    )
    self._serialize_deserialize_roundtrip_test(y)

  def test_compiled_computation_children_is_empty(self):
    comp_type = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(comp_type)
    comp = building_blocks.CompiledComputation(
        proto, name='a', type_signature=comp_type
    )
    self.assertEqual([], list(comp.children()))

  def test_basic_functionality_of_placement_class(self):
    x = building_blocks.Placement(placements.CLIENTS)
    self.assertEqual(str(x.type_signature), 'placement')
    self.assertEqual(x.uri, 'clients')
    self.assertEqual(repr(x), "Placement('clients')")
    self.assertEqual(x.compact_representation(), 'CLIENTS')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature
    )
    self.assertEqual(x_proto.WhichOneof('computation'), 'placement')
    self.assertEqual(x_proto.placement.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_placement_children_is_empty(self):
    placement = building_blocks.Placement(placements.CLIENTS)
    self.assertEqual([], list(placement.children()))

  def _serialize_deserialize_roundtrip_test(self, target):
    """Performs roundtrip serialization/deserialization of the given target.

    Args:
      target: An instane of ComputationBuildingBlock to serialize-deserialize.
    """
    self.assertIsInstance(target, building_blocks.ComputationBuildingBlock)
    serialized = target.proto
    deserialized = building_blocks.ComputationBuildingBlock.from_proto(
        serialized
    )
    reserialized = deserialized.proto
    self.assertEqual(str(serialized), str(reserialized))
    # Note: This is not an equality comparison because ser/de is not an identity
    # transform: it will drop the container from `StructWithPythonType`.
    target.type_signature.check_assignable_from(deserialized.type_signature)


class ReferenceTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_signature = computation_types.TensorType(np.int32)
    reference = building_blocks.Reference('reference', type_signature)

    self.assertIs(reference, reference)
    self.assertEqual(reference, reference)

  @parameterized.named_parameters(
      (
          'same_name_and_type_signature',
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
      ),
      (
          'different_name',
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
          building_blocks.Reference(
              'different', computation_types.TensorType(np.int32)
          ),
      ),
      (
          'different_type_signature',
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.float32)
          ),
      ),
  )
  def test_eq_returns_false(self, reference, other):
    self.assertIsNot(reference, other)
    self.assertNotEqual(reference, other)

  def test_hash_returns_same_value(self):
    type_signature = computation_types.TensorType(np.int32)
    reference = building_blocks.Reference('reference', type_signature)
    other = building_blocks.Reference('reference', type_signature)

    self.assertEqual(hash(reference), hash(other))

  @parameterized.named_parameters(
      (
          'different_name',
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
          building_blocks.Reference(
              'different', computation_types.TensorType(np.int32)
          ),
      ),
      (
          'different_type_signature',
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.int32)
          ),
          building_blocks.Reference(
              'reference', computation_types.TensorType(np.float32)
          ),
      ),
  )
  def test_hash_returns_different_value(self, reference, other):
    self.assertNotEqual(reference, other)
    self.assertNotEqual(hash(reference), hash(other))


class SelectionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'name',
          building_blocks.Selection(
              building_blocks.Struct([('x', _TEST_LITERAL)]),
              name='x',
          ),
          building_blocks.Selection(
              building_blocks.Struct([('x', _TEST_LITERAL)]),
              name='x',
          ),
      ),
      (
          'index',
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL]),
              index=0,
          ),
      ),
  )
  def test_eq_returns_true(self, selection, other):
    self.assertIsNot(selection, other)
    self.assertEqual(selection, other)

  @parameterized.named_parameters(
      (
          'different_source',
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL, _TEST_LITERAL]),
              index=0,
          ),
      ),
      (
          'different_name',
          building_blocks.Selection(
              building_blocks.Struct(
                  [('x', _TEST_LITERAL), ('y', _TEST_LITERAL)]
              ),
              name='x',
          ),
          building_blocks.Selection(
              building_blocks.Struct(
                  [('x', _TEST_LITERAL), ('y', _TEST_LITERAL)]
              ),
              name='y',
          ),
      ),
      (
          'different_index',
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL, _TEST_LITERAL]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([_TEST_LITERAL, _TEST_LITERAL]),
              index=1,
          ),
      ),
  )
  def test_eq_returns_false(self, selection, other):
    self.assertIsNot(selection, other)
    self.assertNotEqual(selection, other)

  @parameterized.named_parameters(
      (
          'name',
          building_blocks.Selection(
              building_blocks.Struct([
                  (
                      'x',
                      building_blocks.Literal(
                          1, computation_types.TensorType(np.int32)
                      ),
                  ),
              ]),
              name='x',
          ),
          building_blocks.Selection(
              building_blocks.Struct([
                  (
                      'x',
                      building_blocks.Literal(
                          1, computation_types.TensorType(np.int32)
                      ),
                  ),
              ]),
              name='x',
          ),
      ),
      (
          'index',
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=0,
          ),
      ),
  )
  def test_hash_returns_same_value(self, selection, other):
    self.assertEqual(hash(selection), hash(other))

  @parameterized.named_parameters(
      (
          'different_source',
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=0,
          ),
      ),
      (
          'different_name',
          building_blocks.Selection(
              building_blocks.Struct([
                  (
                      'x',
                      building_blocks.Literal(
                          2, computation_types.TensorType(np.int32)
                      ),
                  ),
                  (
                      'y',
                      building_blocks.Literal(
                          1, computation_types.TensorType(np.int32)
                      ),
                  ),
              ]),
              name='x',
          ),
          building_blocks.Selection(
              building_blocks.Struct([
                  (
                      'x',
                      building_blocks.Literal(
                          1, computation_types.TensorType(np.int32)
                      ),
                  ),
                  (
                      'y',
                      building_blocks.Literal(
                          1, computation_types.TensorType(np.int32)
                      ),
                  ),
              ]),
              name='y',
          ),
      ),
      (
          'different_index',
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=0,
          ),
          building_blocks.Selection(
              building_blocks.Struct([
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ]),
              index=1,
          ),
      ),
  )
  def test_hash_returns_different_value(self, selection, other):
    self.assertNotEqual(selection, other)
    self.assertNotEqual(hash(selection), hash(other))


class StructTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'container_type_none',
          building_blocks.Struct([
              building_blocks.Literal(
                  1, computation_types.TensorType(np.int32)
              ),
          ]),
          building_blocks.Struct([
              building_blocks.Literal(
                  1, computation_types.TensorType(np.int32)
              ),
          ]),
      ),
      (
          'container_type_list',
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
      ),
  )
  def test_eq_returns_true(self, struct, other):
    self.assertIsNot(struct, other)
    self.assertEqual(struct, other)

  @parameterized.named_parameters(
      (
          'different_elements',
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
      ),
      (
          'different_container_type',
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=tuple,
          ),
      ),
  )
  def test_eq_returns_false(self, struct, other):
    self.assertIsNot(struct, other)
    self.assertNotEqual(struct, other)

  @parameterized.named_parameters(
      (
          'container_type_none',
          building_blocks.Struct([
              building_blocks.Literal(
                  1, computation_types.TensorType(np.int32)
              ),
          ]),
          building_blocks.Struct([
              building_blocks.Literal(
                  1, computation_types.TensorType(np.int32)
              ),
          ]),
      ),
      (
          'container_type_list',
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
      ),
  )
  def test_hash_returns_same_value(self, struct, other):
    self.assertEqual(hash(struct), hash(other))

  def test_hash_returns_same_value_with_different_container_type(self):
    type_signature = computation_types.TensorType(np.int32)
    element = building_blocks.Literal(1, type_signature)
    struct = building_blocks.Struct([element], container_type=list)
    other = building_blocks.Struct([element], container_type=tuple)

    self.assertEqual(hash(struct), hash(other))

  @parameterized.named_parameters(
      (
          'different_elements',
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
          building_blocks.Struct(
              [
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              ],
              container_type=list,
          ),
      ),
  )
  def test_hash_returns_different_value(self, struct, other):
    self.assertNotEqual(struct, other)
    self.assertNotEqual(hash(struct), hash(other))


class CallTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_signature = computation_types.TensorType(np.int32)
    result = building_blocks.Reference('x', type_signature)
    fn = building_blocks.Lambda('x', type_signature, result)
    arg = building_blocks.Literal(1, type_signature)
    call = building_blocks.Call(fn, arg)
    other = building_blocks.Call(fn, arg)

    self.assertIsNot(call, other)
    self.assertEqual(call, other)

  @parameterized.named_parameters(
      (
          'different_fn',
          building_blocks.Call(
              building_blocks.Lambda(
                  'x',
                  computation_types.TensorType(np.int32),
                  building_blocks.Reference(
                      'x', computation_types.TensorType(np.int32)
                  ),
              ),
              building_blocks.Literal(
                  1, computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Call(
              building_blocks.Reference(
                  'different',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'arg', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_arg',
          building_blocks.Call(
              building_blocks.Reference(
                  'fn',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'arg', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Call(
              building_blocks.Reference(
                  'fn',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_eq_returns_false(self, call, other):
    self.assertIsNot(call, other)
    self.assertNotEqual(call, other)

  def test_hash_returns_same_value(self):
    type_signature = computation_types.TensorType(np.int32)
    fn = building_blocks.Reference(
        'fn', computation_types.FunctionType(type_signature, type_signature)
    )
    arg = building_blocks.Reference('arg', type_signature)
    call = building_blocks.Call(fn, arg)
    other = building_blocks.Call(fn, arg)

    self.assertEqual(hash(call), hash(other))

  @parameterized.named_parameters(
      (
          'different_fn',
          building_blocks.Call(
              building_blocks.Reference(
                  'fn',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'arg', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Call(
              building_blocks.Reference(
                  'different',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'arg', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_arg',
          building_blocks.Call(
              building_blocks.Reference(
                  'fn',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'arg', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Call(
              building_blocks.Reference(
                  'fn',
                  computation_types.FunctionType(
                      computation_types.TensorType(np.int32),
                      computation_types.TensorType(np.int32),
                  ),
              ),
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_hash_returns_different_value(self, call, other):
    self.assertNotEqual(call, other)
    self.assertNotEqual(hash(call), hash(other))


class LambdaTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_signature = computation_types.TensorType(np.int32)
    result = building_blocks.Reference('result', type_signature)
    fn = building_blocks.Lambda('parameter', type_signature, result)
    other = building_blocks.Lambda('parameter', type_signature, result)

    self.assertIsNot(fn, other)
    self.assertEqual(fn, other)

  @parameterized.named_parameters(
      (
          'different_parameter_name',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'different',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_parameter_type',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.float32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_result',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_eq_returns_false(self, fn, other):
    self.assertIsNot(fn, other)
    self.assertNotEqual(fn, other)

  def test_hash_returns_same_value(self):
    type_signature = computation_types.TensorType(np.int32)
    result = building_blocks.Reference('result', type_signature)
    fn = building_blocks.Lambda('parameter', type_signature, result)
    other = building_blocks.Lambda('parameter', type_signature, result)

    self.assertEqual(hash(fn), hash(other))

  @parameterized.named_parameters(
      (
          'different_parameter_name',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'different',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_parameter_type',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.float32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_result',
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Lambda(
              'parameter',
              computation_types.TensorType(np.int32),
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_hash_returns_different_value(self, fn, other):
    self.assertNotEqual(fn, other)
    self.assertNotEqual(hash(fn), hash(other))


class BlockTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_signature = computation_types.TensorType(np.int32)
    local = building_blocks.Reference('local', type_signature)
    result = building_blocks.Reference('result', type_signature)
    block = building_blocks.Block([('x', local)], result)
    other = building_blocks.Block([('x', local)], result)

    self.assertIsNot(block, other)
    self.assertEqual(block, other)

  @parameterized.named_parameters(
      (
          'different_locals',
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Block(
              [(
                  'different',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_result',
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      2, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      2, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_eq_returns_false(self, block, other):
    self.assertIsNot(block, other)
    self.assertNotEqual(block, other)

  def test_hash_returns_same_value(self):
    type_signature = computation_types.TensorType(np.int32)
    local = building_blocks.Reference('local', type_signature)
    result = building_blocks.Reference('result', type_signature)
    block = building_blocks.Block([('x', local)], result)
    other = building_blocks.Block([('x', local)], result)

    self.assertEqual(hash(block), hash(other))

  @parameterized.named_parameters(
      (
          'different_locals',
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Block(
              [(
                  'different',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
      ),
      (
          'different_result',
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'result', computation_types.TensorType(np.int32)
              ),
          ),
          building_blocks.Block(
              [(
                  'x',
                  building_blocks.Literal(
                      1, computation_types.TensorType(np.int32)
                  ),
              )],
              building_blocks.Reference(
                  'different', computation_types.TensorType(np.int32)
              ),
          ),
      ),
  )
  def test_hash_returns_different_value(self, block, other):
    self.assertNotEqual(block, other)
    self.assertNotEqual(hash(block), hash(other))


class IntrinsicTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_signature = computation_types.TensorType(np.int32)
    intrinsic = building_blocks.Intrinsic('intrinsic', type_signature)
    other = building_blocks.Intrinsic('intrinsic', type_signature)

    self.assertIsNot(intrinsic, other)
    self.assertEqual(intrinsic, other)

  @parameterized.named_parameters(
      (
          'different_uri',
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.int32)
          ),
          building_blocks.Intrinsic(
              'different', computation_types.TensorType(np.int32)
          ),
      ),
      (
          'different_type_signature',
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.int32)
          ),
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.float32)
          ),
      ),
  )
  def test_eq_returns_false(self, intrinsic, other):
    self.assertIsNot(intrinsic, other)
    self.assertNotEqual(intrinsic, other)

  def test_hash_returns_same_value(self):
    type_signature = computation_types.TensorType(np.int32)
    intrinsic = building_blocks.Intrinsic('intrinsic', type_signature)
    other = building_blocks.Intrinsic('intrinsic', type_signature)

    self.assertEqual(hash(intrinsic), hash(other))

  @parameterized.named_parameters(
      (
          'different_uri',
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.int32)
          ),
          building_blocks.Intrinsic(
              'different', computation_types.TensorType(np.int32)
          ),
      ),
      (
          'different_type_signature',
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.int32)
          ),
          building_blocks.Intrinsic(
              'intrinsic', computation_types.TensorType(np.float32)
          ),
      ),
  )
  def test_hash_returns_different_value(self, intrinsic, other):
    self.assertNotEqual(intrinsic, other)
    self.assertNotEqual(hash(intrinsic), hash(other))


class DataTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3], np.int32)
    )
    type_signature = computation_types.TensorType(np.int32)
    data = building_blocks.Data(any_proto, type_signature)
    other = building_blocks.Data(any_proto, type_signature)

    self.assertIsNot(data, other)
    self.assertEqual(data, other)

  def test_eq_returns_false_different_content(self):
    any_proto1 = (
        building_block_test_utils.create_test_any_proto_for_array_value(
            np.array([1, 2, 3], np.int32)
        )
    )
    type_signature = computation_types.TensorType(np.int32)
    data = building_blocks.Data(any_proto1, type_signature)

    any_proto2 = (
        building_block_test_utils.create_test_any_proto_for_array_value(
            np.array([4], np.int32)
        )
    )
    other = building_blocks.Data(any_proto2, type_signature)
    self.assertIsNot(data, other)
    self.assertNotEqual(data, other)

  def test_eq_returns_false_different_type_signatures(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 1, 1], np.int32)
    )
    type_signature1 = computation_types.TensorType(np.int32)
    type_signature2 = computation_types.TensorType(np.float32)
    data = building_blocks.Data(any_proto, type_signature1)
    other = building_blocks.Data(any_proto, type_signature2)

    self.assertIsNot(data, other)
    self.assertNotEqual(data, other)

  def test_hash_returns_same_value(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3], np.int32)
    )
    type_signature = computation_types.TensorType(np.int32)
    data = building_blocks.Data(any_proto, type_signature)
    other = building_blocks.Data(any_proto, type_signature)

    self.assertEqual(hash(data), hash(other))

  def test_hash_returns_different_value_for_different_content(self):
    any_proto1 = (
        building_block_test_utils.create_test_any_proto_for_array_value(
            np.array([1, 2, 3], np.int32)
        )
    )
    type_signature = computation_types.TensorType(np.int32)
    data = building_blocks.Data(any_proto1, type_signature)

    any_proto2 = (
        building_block_test_utils.create_test_any_proto_for_array_value(
            np.array([4], np.int32)
        )
    )
    other = building_blocks.Data(any_proto2, type_signature)
    self.assertNotEqual(data, other)
    self.assertNotEqual(hash(data), hash(other))

  def test_hash_returns_different_value_for_different_type_signatures(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 1, 1], np.int32)
    )
    type_signature1 = computation_types.TensorType(np.int32)
    type_signature2 = computation_types.TensorType(np.float32)
    data = building_blocks.Data(any_proto, type_signature1)
    other = building_blocks.Data(any_proto, type_signature2)

    self.assertNotEqual(data, other)
    self.assertNotEqual(hash(data), hash(other))


class CompiledComputationTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    type_spec = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(type_spec)
    type_signature = computation_types.FunctionType(type_spec, type_spec)
    compiled = building_blocks.CompiledComputation(
        proto, name='compiled', type_signature=type_signature
    )
    other = building_blocks.CompiledComputation(
        proto, name='compiled', type_signature=type_signature
    )

    self.assertIsNot(compiled, other)
    self.assertEqual(compiled, other)

  @parameterized.named_parameters(
      (
          'different_proto',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.float32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
      ),
      (
          'different_name',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='different',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
      ),
      (
          'different_type_signature',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(
                  np.float32, np.float32
              ),
          ),
      ),
  )
  def test_eq_returns_false(self, compiled, other):
    self.assertIsNot(compiled, other)
    self.assertNotEqual(compiled, other)

  def test_hash_returns_same_value(self):
    type_spec = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(type_spec)
    type_signature = computation_types.FunctionType(type_spec, type_spec)
    compiled = building_blocks.CompiledComputation(
        proto, name='compiled', type_signature=type_signature
    )
    other = building_blocks.CompiledComputation(
        proto, name='compiled', type_signature=type_signature
    )

    self.assertEqual(hash(compiled), hash(other))

  @parameterized.named_parameters(
      (
          'different_proto',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.float32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
      ),
      (
          'different_name',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='different',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
      ),
      (
          'different_type_signature',
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(np.int32, np.int32),
          ),
          building_blocks.CompiledComputation(
              computation_factory.create_lambda_identity(
                  computation_types.TensorType(np.int32)
              ),
              name='compiled',
              type_signature=computation_types.FunctionType(
                  np.float32, np.float32
              ),
          ),
      ),
  )
  def test_hash_returns_different_value(self, compiled, other):
    self.assertNotEqual(compiled, other)
    self.assertNotEqual(hash(compiled), hash(other))


class PlacementTest(parameterized.TestCase):

  def test_eq_returns_true(self):
    placement = building_blocks.Placement(placements.CLIENTS)
    other = building_blocks.Placement(placements.CLIENTS)

    self.assertIsNot(placement, other)
    self.assertEqual(placement, other)

  @parameterized.named_parameters(
      (
          'different_literal',
          building_blocks.Placement(placements.CLIENTS),
          building_blocks.Placement(placements.SERVER),
      ),
  )
  def test_eq_returns_false(self, placement, other):
    self.assertIsNot(placement, other)
    self.assertNotEqual(placement, other)

  def test_hash_returns_same_value(self):
    placement = building_blocks.Placement(placements.CLIENTS)
    other = building_blocks.Placement(placements.CLIENTS)

    self.assertEqual(hash(placement), hash(other))

  @parameterized.named_parameters(
      (
          'different_literal',
          building_blocks.Placement(placements.CLIENTS),
          building_blocks.Placement(placements.SERVER),
      ),
  )
  def test_hash_returns_different_value(self, placement, other):
    self.assertNotEqual(placement, other)
    self.assertNotEqual(hash(placement), hash(other))


class LiteralTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bool', True, computation_types.TensorType(np.bool_)),
      ('int8', 1, computation_types.TensorType(np.int8)),
      ('int16', 1, computation_types.TensorType(np.int16)),
      ('int32', 1, computation_types.TensorType(np.int32)),
      ('int64', 1, computation_types.TensorType(np.int64)),
      ('uint8', 1, computation_types.TensorType(np.uint8)),
      ('uint16', 1, computation_types.TensorType(np.uint16)),
      ('uint32', 1, computation_types.TensorType(np.uint32)),
      ('uint64', 1, computation_types.TensorType(np.uint64)),
      ('float16', 1.0, computation_types.TensorType(np.float16)),
      ('float32', 1.0, computation_types.TensorType(np.float32)),
      ('float64', 1.0, computation_types.TensorType(np.float64)),
      ('complex64', (1.0 + 1.0j), computation_types.TensorType(np.complex64)),
      ('complex128', (1.0 + 1.0j), computation_types.TensorType(np.complex128)),
      ('str', 'a', computation_types.TensorType(np.str_)),
      ('bytes', b'a', computation_types.TensorType(np.str_)),
      ('generic_int', np.int32(1), computation_types.TensorType(np.int32)),
      (
          'generic_float',
          np.float32(1.0),
          computation_types.TensorType(np.float32),
      ),
      (
          'generic_complex',
          np.complex64(1.0 + 1.0j),
          computation_types.TensorType(np.complex64),
      ),
      ('generic_bool', np.bool_(True), computation_types.TensorType(np.bool_)),
      ('generic_str', np.str_('a'), computation_types.TensorType(np.str_)),
      ('generic_bytes', np.bytes_(b'a'), computation_types.TensorType(np.str_)),
      (
          'array_bool',
          np.array([True, False], np.bool_),
          computation_types.TensorType(np.bool_, shape=[2]),
      ),
      (
          'array_int',
          np.array([1, 2, 3], np.int32),
          computation_types.TensorType(np.int32, shape=[3]),
      ),
      (
          'array_float',
          np.array([1.0, 2.0, 3.0], np.float32),
          computation_types.TensorType(np.float32, shape=[3]),
      ),
      (
          'array_complex',
          np.array([(1.0 + 1.0j), (2.0 + 1.0j), (3.0 + 1.0j)], np.complex64),
          computation_types.TensorType(np.complex64, shape=[3]),
      ),
      (
          'array_str',
          np.array(['a', 'b', 'c'], np.str_),
          computation_types.TensorType(np.str_, shape=[3]),
      ),
      (
          'array_bytes',
          np.array([b'a', b'b', b'c'], np.bytes_),
          computation_types.TensorType(np.bytes_, shape=[3]),
      ),
      (
          'array_no_dimensions',
          np.array([], np.int32),
          computation_types.TensorType(np.int32, shape=[0]),
      ),
      (
          'array_multiple_dimensions',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
  )
  def test_init_does_not_raise_value_error(self, value, type_signature):
    try:
      building_blocks.Literal(value, type_signature)
    except ValueError:
      self.fail('Raised `ValueError` unexpectedly.')

  @parameterized.named_parameters(
      ('str', 'a', computation_types.TensorType(np.str_), b'a'),
      (
          'generic_str',
          np.str_('a'),
          computation_types.TensorType(np.str_),
          b'a',
      ),
      (
          'array_str',
          np.array(['a', 'b', 'c'], np.str_),
          computation_types.TensorType(np.str_, shape=[3]),
          np.array([b'a', b'b', b'c'], np.bytes_),
      ),
  )
  def test_init_normalizes_value_str(
      self, value, type_signature, expected_value
  ):
    literal = building_blocks.Literal(value, type_signature)

    tree.assert_same_structure(literal.value, expected_value)
    actual_value = _to_python(literal.value)
    expected_value = _to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'scalar_and_incompatible_dtype_kind',
          1,
          computation_types.TensorType(np.float32),
      ),
      (
          'scalar_and_incompatible_dtype_size',
          np.iinfo(np.int64).max,
          computation_types.TensorType(np.int32),
      ),
      (
          'scalar_and_incompatible_shape',
          1,
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'generic_and_incompatible_dtype_kind',
          np.int32(1),
          computation_types.TensorType(np.float32),
      ),
      (
          'generic_and_incompatible_dtype_size',
          np.int64(np.iinfo(np.int64).max),
          computation_types.TensorType(np.int32),
      ),
      (
          'generic_and_incompatible_shape',
          np.int32(1),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'array_and_incompatible_dtype_kind',
          np.array([1, 2, 3], np.int32),
          computation_types.TensorType(np.float32, shape=[3]),
      ),
      (
          'array_and_incompatible_dtype_size',
          np.array([np.iinfo(np.int64).max] * 3, np.int64),
          computation_types.TensorType(np.int32, shape=[3]),
      ),
      (
          'array_and_incompatible_shape',
          np.array([1, 2, 3], np.int32),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
  )
  def test_init_raises_value_error(self, value, type_signature):
    with self.assertRaises(ValueError):
      building_blocks.Literal(value, type_signature)

  def test_from_proto_returns_value(self):
    proto = computation_pb2.Computation(
        type=computation_pb2.Type(
            tensor=computation_pb2.TensorType(
                dtype=data_type_pb2.DataType.DT_INT32, dims=[]
            )
        ),
        literal=computation_pb2.Literal(
            value=array_pb2.Array(
                dtype=data_type_pb2.DataType.DT_INT32,
                shape=array_pb2.ArrayShape(dim=[]),
                int32_list=array_pb2.Array.IntList(value=[1]),
            )
        ),
    )

    actual_value = building_blocks.Literal.from_proto(proto)

    expected_value = building_blocks.Literal(
        1, computation_types.TensorType(np.int32)
    )
    self.assertEqual(actual_value, expected_value)

  def test_from_proto_raises_value_error_with_wrong_type(self):
    proto = computation_pb2.Computation(
        type=computation_pb2.Type(
            federated=computation_pb2.FederatedType(
                placement=computation_pb2.PlacementSpec(
                    value=computation_pb2.Placement(uri=placements.CLIENTS.uri)
                ),
                member=computation_pb2.Type(
                    tensor=computation_pb2.TensorType(
                        dtype=data_type_pb2.DataType.DT_INT32, dims=[]
                    ),
                ),
            ),
        ),
        literal=computation_pb2.Literal(
            value=array_pb2.Array(
                dtype=data_type_pb2.DataType.DT_INT32,
                shape=array_pb2.ArrayShape(dim=[]),
                int32_list=array_pb2.Array.IntList(value=[1]),
            )
        ),
    )

    with self.assertRaises(ValueError):
      building_blocks.Literal.from_proto(proto)

  def test_from_proto_raises_value_error_with_wrong_computation(self):
    proto = computation_pb2.Computation(
        type=computation_pb2.Type(
            tensor=computation_pb2.TensorType(
                dtype=data_type_pb2.DataType.DT_INT32, dims=[]
            )
        ),
        data=computation_pb2.Data(),
    )

    with self.assertRaises(ValueError):
      building_blocks.Literal.from_proto(proto)

  def test_to_proto_returns_value(self):
    literal = building_blocks.Literal(1, computation_types.TensorType(np.int32))

    actual_value = literal._proto()

    expected_value = computation_pb2.Computation(
        type=computation_pb2.Type(
            tensor=computation_pb2.TensorType(
                dtype=data_type_pb2.DataType.DT_INT32, dims=[]
            )
        ),
        literal=computation_pb2.Literal(
            value=array_pb2.Array(
                dtype=data_type_pb2.DataType.DT_INT32,
                shape=array_pb2.ArrayShape(dim=[]),
                int32_list=array_pb2.Array.IntList(value=[1]),
            )
        ),
    )
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'bool',
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
      ),
      (
          'int',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
      ),
      (
          'float',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
      ),
      (
          'complex',
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
      ),
      (
          'str',
          building_blocks.Literal('a', computation_types.TensorType(np.str_)),
      ),
      (
          'bytes',
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
      ),
      (
          'generic',
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'array',
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
      ),
  )
  def test_children_empty(self, literal):
    actual_children = list(literal.children())
    self.assertEmpty(actual_children)

  @parameterized.named_parameters(
      (
          'bool',
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
      ),
      (
          'int8',
          building_blocks.Literal(1, computation_types.TensorType(np.int8)),
          building_blocks.Literal(1, computation_types.TensorType(np.int8)),
      ),
      (
          'int16',
          building_blocks.Literal(1, computation_types.TensorType(np.int16)),
          building_blocks.Literal(1, computation_types.TensorType(np.int16)),
      ),
      (
          'int32',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
      ),
      (
          'int64',
          building_blocks.Literal(1, computation_types.TensorType(np.int64)),
          building_blocks.Literal(1, computation_types.TensorType(np.int64)),
      ),
      (
          'uint8',
          building_blocks.Literal(1, computation_types.TensorType(np.uint8)),
          building_blocks.Literal(1, computation_types.TensorType(np.uint8)),
      ),
      (
          'uint16',
          building_blocks.Literal(1, computation_types.TensorType(np.uint16)),
          building_blocks.Literal(1, computation_types.TensorType(np.uint16)),
      ),
      (
          'uint32',
          building_blocks.Literal(1, computation_types.TensorType(np.uint32)),
          building_blocks.Literal(1, computation_types.TensorType(np.uint32)),
      ),
      (
          'uint64',
          building_blocks.Literal(1, computation_types.TensorType(np.uint64)),
          building_blocks.Literal(1, computation_types.TensorType(np.uint64)),
      ),
      (
          'float16',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float16)
          ),
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float16)
          ),
      ),
      (
          'float32',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
      ),
      (
          'float64',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float64)
          ),
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float64)
          ),
      ),
      (
          'complex64',
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
      ),
      (
          'complex128',
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex128)
          ),
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex128)
          ),
      ),
      (
          'str',
          building_blocks.Literal('a', computation_types.TensorType(np.str_)),
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
      ),
      (
          'bytes',
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
      ),
      (
          'generic',
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'array',
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
      ),
  )
  def test_eq_returns_true(self, literal, other):
    self.assertIsNot(literal, other)
    self.assertEqual(literal, other)

  @parameterized.named_parameters(
      (
          'different_value',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(2, computation_types.TensorType(np.int32)),
      ),
      (
          'different_type_signature',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(1, computation_types.TensorType(np.int64)),
      ),
  )
  def test_eq_returns_false(self, literal, other):
    self.assertIsNot(literal, other)
    self.assertNotEqual(literal, other)

  @parameterized.named_parameters(
      (
          'bool',
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
      ),
      (
          'int',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
      ),
      (
          'uint',
          building_blocks.Literal(1, computation_types.TensorType(np.uint32)),
          building_blocks.Literal(1, computation_types.TensorType(np.uint32)),
      ),
      (
          'float',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
      ),
      (
          'complex',
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
      ),
      (
          'str',
          building_blocks.Literal('a', computation_types.TensorType(np.str_)),
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
      ),
      (
          'bytes',
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
      ),
      (
          'generic',
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'array',
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
      ),
  )
  def test_hash_returns_same_value(self, literal, other):
    self.assertEqual(hash(literal), hash(other))

  @parameterized.named_parameters(
      (
          'different_value',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(2, computation_types.TensorType(np.int32)),
      ),
      (
          'different_type_signature',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          building_blocks.Literal(1, computation_types.TensorType(np.int64)),
      ),
  )
  def test_hash_returns_different_value(self, literal, other):
    self.assertNotEqual(literal, other)
    self.assertNotEqual(hash(literal), hash(other))

  @parameterized.named_parameters(
      (
          'bool',
          building_blocks.Literal(True, computation_types.TensorType(np.bool_)),
          'Literal(True, TensorType(np.bool_))',
      ),
      (
          'int',
          building_blocks.Literal(1, computation_types.TensorType(np.int32)),
          'Literal(1, TensorType(np.int32))',
      ),
      (
          'float',
          building_blocks.Literal(
              1.0, computation_types.TensorType(np.float32)
          ),
          'Literal(1.0, TensorType(np.float32))',
      ),
      (
          'complex',
          building_blocks.Literal(
              (1.0 + 1.0j), computation_types.TensorType(np.complex64)
          ),
          'Literal((1+1j), TensorType(np.complex64))',
      ),
      (
          'str',
          building_blocks.Literal('a', computation_types.TensorType(np.str_)),
          "Literal(b'a', TensorType(np.str_))",
      ),
      (
          'bytes',
          building_blocks.Literal(b'a', computation_types.TensorType(np.str_)),
          "Literal(b'a', TensorType(np.str_))",
      ),
      (
          'generic',
          building_blocks.Literal(
              np.int32(1), computation_types.TensorType(np.int32)
          ),
          'Literal(1, TensorType(np.int32))',
      ),
      (
          'array',
          building_blocks.Literal(
              np.array([1, 2, 3], np.int32),
              computation_types.TensorType(np.int32, shape=[3]),
          ),
          (
              'Literal(np.array([1, 2, 3], dtype=np.int32),'
              ' TensorType(np.int32, (3,)))'
          ),
      ),
  )
  def test_repr(self, literal, expected_repr):
    self.assertEqual(repr(literal), expected_repr)


class RepresentationTest(absltest.TestCase):

  def test_returns_string_for_block(self):
    data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    ref = building_blocks.Reference('c', np.int32)
    comp = building_blocks.Block((('a', data), ('b', data)), ref)

    self.assertEqual(comp.compact_representation(), '(let a=1,b=1 in c)')
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  a=1,\n'
        '  b=1\n'
        ' in c)'
    )
    self.assertEqual(
        comp.structural_representation(),
        '                     Block\n'
        '                    /     \\\n'
        '[a=Lit(1), b=Lit(1)]       Ref(c)'
    )
    # pyformat: enable

  def test_returns_string_for_call_with_arg(self):
    fn_type = computation_types.FunctionType(np.int32, np.int32)
    fn = building_blocks.Reference('a', fn_type)
    arg = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    comp = building_blocks.Call(fn, arg)

    self.assertEqual(comp.compact_representation(), 'a(1)')
    self.assertEqual(comp.formatted_representation(), 'a(1)')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        '       Call\n'
        '      /    \\\n'
        'Ref(a)      Lit(1)'
    )
    # pyformat: enable

  def test_returns_string_for_call_with_no_arg(self):
    fn_type = computation_types.FunctionType(None, np.int32)
    fn = building_blocks.Reference('a', fn_type)
    comp = building_blocks.Call(fn)

    self.assertEqual(comp.compact_representation(), 'a()')
    self.assertEqual(comp.formatted_representation(), 'a()')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        '       Call\n'
        '      /\n'
        'Ref(a)'
    )
    # pyformat: enable

  def test_returns_string_for_compiled_computation(self):
    tensor_type = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(tensor_type)
    comp = building_blocks.CompiledComputation(
        proto, name='a', type_signature=tensor_type
    )

    self.assertEqual(comp.compact_representation(), 'comp#a')
    self.assertEqual(comp.formatted_representation(), 'comp#a')
    self.assertEqual(comp.structural_representation(), 'Compiled(a)')

  def test_returns_string_for_data(self):
    any_proto = building_block_test_utils.create_test_any_proto_for_array_value(
        np.array([1, 2, 3], np.int32)
    )
    comp = building_blocks.Data(any_proto, np.int32)

    s = str(id(any_proto))
    self.assertEqual(comp.compact_representation(), s)
    self.assertEqual(comp.formatted_representation(), s)
    self.assertEqual(comp.structural_representation(), f'Data({s})')

  def test_returns_string_for_intrinsic(self):
    comp_type = computation_types.TensorType(np.int32)
    comp = building_blocks.Intrinsic('intrinsic', comp_type)

    self.assertEqual(comp.compact_representation(), 'intrinsic')
    self.assertEqual(comp.formatted_representation(), 'intrinsic')
    self.assertEqual(comp.structural_representation(), 'intrinsic')

  def test_returns_string_for_lambda(self):
    ref = building_blocks.Reference('a', np.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)

    self.assertEqual(comp.compact_representation(), '(a -> a)')
    self.assertEqual(comp.formatted_representation(), '(a -> a)')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        'Lambda(a)\n'
        '|\n'
        'Ref(a)'
    )
    # pyformat: enable

  def test_returns_string_for_placement(self):
    comp = building_blocks.Placement(placements.CLIENTS)

    self.assertEqual(comp.compact_representation(), 'CLIENTS')
    self.assertEqual(comp.formatted_representation(), 'CLIENTS')
    self.assertEqual(comp.structural_representation(), 'Placement')

  def test_returns_string_for_reference(self):
    comp = building_blocks.Reference('a', np.int32)

    self.assertEqual(comp.compact_representation(), 'a')
    self.assertEqual(comp.formatted_representation(), 'a')
    self.assertEqual(comp.structural_representation(), 'Ref(a)')

  def test_returns_string_for_selection_with_name(self):
    ref = building_blocks.Reference('a', (('b', np.int32), ('c', np.bool_)))
    comp = building_blocks.Selection(ref, name='b')

    self.assertEqual(comp.compact_representation(), 'a.b')
    self.assertEqual(comp.formatted_representation(), 'a.b')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        'Sel(b)\n'
        '|\n'
        'Ref(a)'
    )
    # pyformat: enable

  def test_returns_string_for_selection_with_index(self):
    ref = building_blocks.Reference('a', (('b', np.int32), ('c', np.bool_)))
    comp = building_blocks.Selection(ref, index=0)

    self.assertEqual(comp.compact_representation(), 'a[0]')
    self.assertEqual(comp.formatted_representation(), 'a[0]')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        'Sel(0)\n'
        '|\n'
        'Ref(a)'
    )
    # pyformat: enable

  def test_returns_string_for_struct_with_names(self):
    literally = building_blocks.Literal(
        2, computation_types.TensorType(np.int32)
    )
    comp = building_blocks.Struct([('a', literally), ('b', literally)])

    self.assertEqual(comp.compact_representation(), '<a=2,b=2>')
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  a=2,\n'
        '  b=2\n'
        '>'
    )
    self.assertEqual(
        comp.structural_representation(),
        'Struct\n'
        '|\n'
        '[a=Lit(2), b=Lit(2)]'
    )
    # pyformat: enable

  def test_returns_string_for_struct_with_no_names(self):
    data = building_blocks.Literal(3, computation_types.TensorType(np.int32))
    comp = building_blocks.Struct([data, data])

    self.assertEqual(comp.compact_representation(), '<3,3>')
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  3,\n'
        '  3\n'
        '>'
    )
    self.assertEqual(
        comp.structural_representation(),
        'Struct\n'
        '|\n'
        '[Lit(3), Lit(3)]'
    )
    # pyformat: enable

  def test_returns_string_for_struct_with_no_elements(self):
    comp = building_blocks.Struct([])

    self.assertEqual(comp.compact_representation(), '<>')
    self.assertEqual(comp.formatted_representation(), '<>')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        'Struct\n'
        '|\n'
        '[]'
    )
    # pyformat: enable

  def test_returns_string_for_federated_aggregate(self):
    comp = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
    )

    self.assertEqual(
        comp.compact_representation(),
        'federated_aggregate(<federated_value_at_clients(1),1,(a -> 1),(b -> 1)'
        ',(c -> 1)>)',
    )
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_aggregate(<\n'
        '  federated_value_at_clients(1),\n'
        '  1,\n'
        '  (a -> 1),\n'
        '  (b -> 1),\n'
        '  (c -> 1)\n'
        '>)'
    )
    self.assertEqual(
        comp.structural_representation(),
        '                     Call\n'
        '                    /    \\\n'
        ' federated_aggregate      Struct\n'
        '                          |\n'
        '                          [Call, Lit(1),  Lambda(a), Lambda(b), '
        'Lambda(c)]\n'
        '                          /    \\          |          |          |\n'
        'federated_value_at_clients      Lit(1)    Lit(1)     Lit(1)     Lit(1)'
    )
    # pyformat: enable

  def test_returns_string_for_federated_map(self):
    comp = building_block_test_utils.create_whimsy_called_federated_map(
        parameter_name='a'
    )

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_value_at_clients(1)>)',
    )
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (a -> a),\n'
        '  federated_value_at_clients(1)\n'
        '>)'
    )
    self.assertEqual(
        comp.structural_representation(),
        '              Call\n'
        '             /    \\\n'
        'federated_map      Struct\n'
        '                   |\n'
        '                   [Lambda(a),                           Call]\n'
        '                    |                                   /    \\\n'
        '                    Ref(a)    federated_value_at_clients      Lit(1)'
    )
    # pyformat: enable

  def test_returns_string_for_comp_with_left_overhang(self):
    fn_1_type = computation_types.FunctionType(np.int32, np.int32)
    fn_1 = building_blocks.Reference('a', fn_1_type)
    fn_2_type = computation_types.FunctionType(None, np.int32)
    fn_2 = building_blocks.Reference('bbbbbbbbbb', fn_2_type)
    arg = building_blocks.Call(fn_2)
    comp = building_blocks.Call(fn_1, arg)

    self.assertEqual(comp.compact_representation(), 'a(bbbbbbbbbb())')
    self.assertEqual(comp.formatted_representation(), 'a(bbbbbbbbbb())')
    # pyformat: disable
    self.assertEqual(
        comp.structural_representation(),
        '           Call\n'
        '          /    \\\n'
        '    Ref(a)      Call\n'
        '               /\n'
        'Ref(bbbbbbbbbb)'
    )
    # pyformat: enable

  def test_returns_string_for_comp_with_right_overhang(self):
    ref = building_blocks.Reference('a', np.int32)
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    tup = building_blocks.Struct([ref, lit, lit, lit, lit])
    sel = building_blocks.Selection(tup, index=0)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, sel)
    comp = building_blocks.Call(fn, lit)

    self.assertEqual(comp.compact_representation(), '(a -> <a,1,1,1,1>[0])(1)')
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(a -> <\n'
        '  a,\n'
        '  1,\n'
        '  1,\n'
        '  1,\n'
        '  1\n'
        '>[0])(1)'
    )
    self.assertEqual(
        comp.structural_representation(),
        '          Call\n'
        '         /    \\\n'
        'Lambda(a)      Lit(1)\n'
        '|\n'
        'Sel(0)\n'
        '|\n'
        'Struct\n'
        '|\n'
        '[Ref(a), Lit(1), Lit(1), Lit(1), Lit(1)]'
    )
    # pyformat: enable


if __name__ == '__main__':
  absltest.main()
