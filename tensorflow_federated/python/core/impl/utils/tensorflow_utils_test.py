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

import collections

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class GraphUtilsTest(test_case.TestCase):

  def _assert_binding_matches_type_and_value(self, binding, type_spec, val,
                                             graph, is_output):
    """Asserts that 'bindings' matches the given type, value, and graph."""
    self.assertIsInstance(binding, pb.TensorFlow.Binding)
    self.assertIsInstance(type_spec, computation_types.Type)
    binding_oneof = binding.WhichOneof('binding')
    if binding_oneof == 'tensor':
      self.assertTrue(tf.is_tensor(val))
      if is_output:
        # Output tensor names must not match, because `val` might also be in the
        # input binding, causing the same tensor to appear in the `feeds` and
        # `fetches` of the `Session.run()` wich is disallowed by TensorFlow.
        self.assertNotEqual(binding.tensor.tensor_name, val.name)
      else:
        # Input binding names are expected to match
        self.assertEqual(binding.tensor.tensor_name, val.name)
      self.assertIsInstance(type_spec, computation_types.TensorType)
      self.assertEqual(type_spec.dtype, val.dtype.base_dtype)
      self.assertEqual(repr(type_spec.shape), repr(val.shape))
    elif binding_oneof == 'sequence':
      self.assertIsInstance(val,
                            type_conversions.TF_DATASET_REPRESENTATION_TYPES)
      sequence_oneof = binding.sequence.WhichOneof('binding')
      self.assertEqual(sequence_oneof, 'variant_tensor_name')
      variant_tensor = graph.get_tensor_by_name(
          binding.sequence.variant_tensor_name)
      op = str(variant_tensor.op.type)
      self.assertTrue((op == 'Placeholder') or ('Dataset' in op))
      self.assertEqual(variant_tensor.dtype, tf.variant)
      self.assertIsInstance(type_spec, computation_types.SequenceType)
      self.assertEqual(
          computation_types.to_type(val.element_spec), type_spec.element)
    elif binding_oneof == 'struct':
      self.assertIsInstance(type_spec, computation_types.StructType)
      if not isinstance(val, (list, tuple, structure.Struct)):
        self.assertIsInstance(val, dict)
        if isinstance(val, collections.OrderedDict):
          val = list(val.values())
        else:
          val = [v for _, v in sorted(val.items())]
      for idx, e in enumerate(structure.to_elements(type_spec)):
        self._assert_binding_matches_type_and_value(binding.struct.element[idx],
                                                    e[1], val[idx], graph,
                                                    is_output)
    else:
      self.fail('Unknown binding.')

  def _assert_input_binding_matches_type_and_value(self, binding, type_spec,
                                                   val, graph):
    self._assert_binding_matches_type_and_value(
        binding, type_spec, val, graph, is_output=False)

  def _assert_output_binding_matches_type_and_value(self, binding, type_spec,
                                                    val, graph):
    self._assert_binding_matches_type_and_value(
        binding, type_spec, val, graph, is_output=True)

  def _assert_captured_result_eq_dtype(self, type_spec, binding, dtype):
    self.assertIsInstance(type_spec, computation_types.TensorType)
    self.assertEqual(str(type_spec), dtype)
    self.assertEqual(binding.WhichOneof('binding'), 'tensor')

  def _assert_is_placeholder(self, x, name, dtype, shape, graph):
    """Verifies that 'x' is a placeholder with the given attributes."""
    self.assertEqual(x.name, name)
    self.assertEqual(x.dtype, dtype)
    self.assertEqual(x.shape.ndims, len(shape))
    for i, s in enumerate(shape):
      self.assertEqual(x.shape.dims[i].value, s)
    self.assertEqual(x.op.type, 'Placeholder')
    self.assertIs(x.graph, graph)

  def _checked_capture_result(self, result):
    """Returns the captured result type after first verifying the binding."""
    graph = tf.compat.v1.get_default_graph()
    type_spec, binding = tensorflow_utils.capture_result_from_graph(
        result, graph)
    # If the input is a tensor (but not a tf.Variable), ensure that an identity
    # operation was added.
    if tf.is_tensor(result) and not hasattr(result, 'read_value'):
      self.assertNotEqual(result.name, binding.tensor.tensor_name)
    self._assert_output_binding_matches_type_and_value(binding, type_spec,
                                                       result, graph)
    return type_spec

  def _checked_stamp_parameter(self, name, spec, graph=None):
    """Returns object stamped in the graph after verifying its bindings."""
    if graph is None:
      graph = tf.compat.v1.get_default_graph()
    val, binding = tensorflow_utils.stamp_parameter_in_graph(name, spec, graph)
    self._assert_input_binding_matches_type_and_value(
        binding, computation_types.to_type(spec), val, graph)
    return val

  def test_stamp_parameter_in_graph_with_scalar_int_explicit_graph(self):
    my_graph = tf.Graph()
    x = self._checked_stamp_parameter('foo', tf.int32, my_graph)
    self._assert_is_placeholder(x, 'foo:0', tf.int32, [], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_implicit_graph(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [5]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [5], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_undefined_size(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [None]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [None], my_graph)

  def test_stamp_parameter_in_graph_with_struct(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter(
          'foo', computation_types.StructType([('a', tf.int32),
                                               ('b', tf.bool)]))
    self.assertIsInstance(x, structure.Struct)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_struct_with_python_type(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.StructWithPythonType([('a', tf.int32),
                                                  ('b', tf.bool)],
                                                 collections.OrderedDict))
    self.assertIsInstance(x, structure.Struct)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_bool_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter('foo',
                                        computation_types.SequenceType(tf.bool))
      self.assertIsInstance(x, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
      self.assertEqual(x.element_spec, tf.TensorSpec(shape=(), dtype=tf.bool))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', computation_types.SequenceType((tf.int32, [50])))
      self.assertIsInstance(x, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
      self.assertEqual(x.element_spec,
                       tf.TensorSpec(shape=(50,), dtype=tf.int32))

  def test_stamp_parameter_in_graph_with_tensor_ordered_dict_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.SequenceType(
              collections.OrderedDict([('A', (tf.float32, [3, 4, 5])),
                                       ('B', (tf.int32, [1]))])))
      self.assertIsInstance(x, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
      self.assertEqual(
          x.element_spec, {
              'A': tf.TensorSpec(shape=(3, 4, 5), dtype=tf.float32),
              'B': tf.TensorSpec(shape=(1,), dtype=tf.int32),
          })

  def test_capture_result_with_string(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          'a', graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'string')

  def test_capture_result_with_int(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(1, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_float(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          1.0, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          True, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_int32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.int32(1), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_np_int64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.int64(1), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int64')

  def test_capture_result_with_np_float32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.float32(1.0), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_np_float64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.float64(1.0), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float64')

  def test_capture_result_with_np_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.bool(True), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_ndarray(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.ndarray(shape=(2, 0), dtype=np.int32), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32[2,0]')

  def test_capture_result_with_ragged_tensor(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          tf.RaggedTensor.from_row_splits([0, 0, 0, 0], [0, 1, 4]), graph)
      del binding
      self.assert_types_identical(
          type_spec,
          computation_types.StructWithPythonType([
              ('flat_values', computation_types.TensorType(tf.int32, [4])),
              ('nested_row_splits',
               computation_types.StructWithPythonType([
                   (None, computation_types.TensorType(tf.int64, [3]))
               ], tuple)),
          ], tf.RaggedTensor))

  def test_capture_result_with_sparse_tensor(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          tf.SparseTensor(indices=[[1]], values=[2], dense_shape=[5]), graph)
      del binding
      self.assert_types_identical(
          type_spec,
          computation_types.StructWithPythonType([
              ('indices', computation_types.TensorType(tf.int64, [1, 1])),
              ('values', computation_types.TensorType(tf.int32, [1])),
              ('dense_shape', computation_types.TensorType(tf.int64, [1])),
              ('static_dense_shape_marker',
               computation_types.marker_type_from_static_shape([5], 'N/A')),
          ], tf.SparseTensor))

  @test_utils.graph_mode_test
  def test_capture_result_with_int_placeholder(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.compat.v1.placeholder(tf.int32, shape=[]))), 'int32')

  @test_utils.graph_mode_test
  def test_capture_result_with_int_variable(self):
    # Verifies that the variable dtype is not being captured as `int32_ref`,
    # since TFF has no concept of passing arguments by reference.
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.Variable(
                    initial_value=0, name='foo', dtype=tf.int32, shape=[]))),
        'int32')

  @test_utils.graph_mode_test
  def test_capture_result_with_list_of_constants(self):
    t = self._checked_capture_result([tf.constant(1), tf.constant(True)])
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIs(t.python_container, list)

  @test_utils.graph_mode_test
  def test_capture_result_with_tuple_of_constants(self):
    t = self._checked_capture_result((tf.constant(1), tf.constant(True)))
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIs(t.python_container, tuple)

  @test_utils.graph_mode_test
  def test_capture_result_with_dict_of_constants(self):
    t1 = self._checked_capture_result({
        'a': tf.constant(1),
        'b': tf.constant(True),
    })
    self.assertEqual(str(t1), '<a=int32,b=bool>')
    self.assertIs(t1.python_container, dict)

    t2 = self._checked_capture_result({
        'b': tf.constant(True),
        'a': tf.constant(1),
    })
    self.assertEqual(str(t2), '<a=int32,b=bool>')
    self.assertIs(t2.python_container, dict)

  @test_utils.graph_mode_test
  def test_capture_result_with_ordered_dict_of_constants(self):
    t = self._checked_capture_result(
        collections.OrderedDict([
            ('b', tf.constant(True)),
            ('a', tf.constant(1)),
        ]))
    self.assertEqual(str(t), '<b=bool,a=int32>')
    self.assertIs(t.python_container, collections.OrderedDict)

  @test_utils.graph_mode_test
  def test_capture_result_with_namedtuple_of_constants(self):
    test_named_tuple = collections.namedtuple('_', 'x y')
    t = self._checked_capture_result(
        test_named_tuple(tf.constant(1), tf.constant(True)))
    self.assertEqual(str(t), '<x=int32,y=bool>')
    self.assertIs(t.python_container, test_named_tuple)

  @test_utils.graph_mode_test
  def test_capture_result_with_attrs_of_constants(self):

    @attr.s
    class TestFoo(object):
      x = attr.ib()
      y = attr.ib()

    graph = tf.compat.v1.get_default_graph()
    type_spec, _ = tensorflow_utils.capture_result_from_graph(
        TestFoo(tf.constant(1), tf.constant(True)), graph)
    self.assertEqual(str(type_spec), '<x=int32,y=bool>')
    self.assertIs(type_spec.python_container, TestFoo)

  @test_utils.graph_mode_test
  def test_capture_result_with_struct_of_constants(self):
    t = self._checked_capture_result(
        structure.Struct([
            ('x', tf.constant(10)),
            (None, tf.constant(True)),
            ('y', tf.constant(0.66)),
        ]))
    self.assertEqual(str(t), '<x=int32,bool,y=float32>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)

  @test_utils.graph_mode_test
  def test_capture_result_with_nested_lists_and_tuples(self):
    named_tuple_type = collections.namedtuple('_', 'a b')
    t = self._checked_capture_result(
        structure.Struct([
            ('x',
             named_tuple_type({'p': {
                 'q': tf.constant(True)
             }}, [tf.constant(False)])),
            (None, [[tf.constant(10)]]),
        ]))
    self.assertEqual(str(t), '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertIsInstance(t.x, computation_types.StructWithPythonType)
    self.assertIs(t.x.python_container, named_tuple_type)
    self.assertIsInstance(t[1], computation_types.StructWithPythonType)
    self.assertIs(t[1].python_container, list)

  @test_utils.graph_mode_test
  def test_capture_result_with_sequence_of_ints_using_from_tensors(self):
    ds = tf.data.Dataset.from_tensors(tf.constant(10))
    self.assertEqual(str(self._checked_capture_result(ds)), 'int32*')

  @test_utils.graph_mode_test
  def test_capture_result_with_sequence_of_ints_using_range(self):
    ds = tf.data.Dataset.range(10)
    self.assertEqual(str(self._checked_capture_result(ds)), 'int64*')

  def test_capture_result_with_sequence_of_dicts_fails(self):
    ds = tf.data.Dataset.from_tensor_slices({'A': [1, 2, 3], 'B': [4, 5, 6]})
    with golden.check_raises_traceback(
        'capture_result_with_sequence_of_dicts.expected', TypeError):
      self._checked_capture_result(ds)

  def test_compute_map_from_bindings_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, source = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                   ('bar', tf.constant(20, name='B'))]), graph)
      _, target = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(30, name='C')),
                                   ('bar', tf.constant(40, name='D'))]), graph)
    result = tensorflow_utils.compute_map_from_bindings(source, target)
    self.assertAllEqual(
        result,
        collections.OrderedDict([('Identity:0', 'Identity_2:0'),
                                 ('Identity_1:0', 'Identity_3:0')]))

  def test_compute_map_from_bindings_with_sequence(self):
    source = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo'))
    target = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='bar'))
    result = tensorflow_utils.compute_map_from_bindings(source, target)
    self.assertEqual(str(result), 'OrderedDict([(\'foo\', \'bar\')])')

  def test_extract_tensor_names_from_binding_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, binding = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                   ('bar', tf.constant(20, name='B'))]), graph)
    result = tensorflow_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(result, ['Identity:0', 'Identity_1:0'])

  def test_extract_tensor_names_from_binding_with_sequence(self):
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo'))
    result = tensorflow_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(str(sorted(result)), '[\'foo\']')

  @test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_named_tuple(self):
    test_named_tuple = collections.namedtuple('_', 'X Y')
    type_spec = test_named_tuple(tf.int32, tf.int32)
    binding = pb.TensorFlow.Binding(
        struct=pb.TensorFlow.StructBinding(element=[
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='P')),
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='Q'))
        ]))
    tensor_a = tf.constant(1, name='A')
    tensor_b = tf.constant(2, name='B')
    output_map = {'P': tensor_a, 'Q': tensor_b}
    result = tensorflow_utils.assemble_result_from_graph(
        type_spec, binding, output_map)
    self.assertIsInstance(result, test_named_tuple)
    self.assertEqual(result.X, tensor_a)
    self.assertEqual(result.Y, tensor_b)

  @test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_odicts(self):
    type_spec = computation_types.SequenceType(
        collections.OrderedDict([('X', tf.int32), ('Y', tf.int32)]))
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo'))
    data_set = tf.data.Dataset.from_tensors({
        'X': tf.constant(1),
        'Y': tf.constant(2)
    })
    output_map = {'foo': tf.data.experimental.to_variant(data_set)}
    result = tensorflow_utils.assemble_result_from_graph(
        type_spec, binding, output_map)
    self.assertIsInstance(result,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        result.element_spec,
        collections.OrderedDict([
            ('X', tf.TensorSpec(shape=(), dtype=tf.int32)),
            ('Y', tf.TensorSpec(shape=(), dtype=tf.int32)),
        ]),
    )

  @test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_namedtuples(self):
    named_tuple_type = collections.namedtuple('TestNamedTuple', 'X Y')
    type_spec = computation_types.SequenceType(
        named_tuple_type(tf.int32, tf.int32))
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo'))
    data_set = tf.data.Dataset.from_tensors({
        'X': tf.constant(1),
        'Y': tf.constant(2)
    })
    output_map = {'foo': tf.data.experimental.to_variant(data_set)}
    result = tensorflow_utils.assemble_result_from_graph(
        type_spec, binding, output_map)
    self.assertIsInstance(result,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        result.element_spec,
        named_tuple_type(
            X=tf.TensorSpec(shape=(), dtype=tf.int32),
            Y=tf.TensorSpec(shape=(), dtype=tf.int32),
        ))

  def test_make_whimsy_element_for_type_spec_raises_SequenceType(self):
    type_spec = computation_types.SequenceType(tf.float32)
    with self.assertRaisesRegex(ValueError,
                                'Cannot construct array for TFF type'):
      tensorflow_utils.make_whimsy_element_for_type_spec(type_spec)

  def test_make_whimsy_element_for_type_spec_raises_negative_none_dim_replacement(
      self):
    with self.assertRaisesRegex(ValueError, 'nonnegative'):
      tensorflow_utils.make_whimsy_element_for_type_spec(tf.float32, -1)

  def test_make_whimsy_element_TensorType(self):
    type_spec = computation_types.TensorType(tf.float32,
                                             [None, 10, None, 10, 10])
    elem = tensorflow_utils.make_whimsy_element_for_type_spec(type_spec)
    correct_elem = np.zeros([0, 10, 0, 10, 10], np.float32)
    self.assertAllClose(elem, correct_elem)

  def test_make_whimsy_element_tensor_type_backed_by_tf_dimension(self):
    type_spec = computation_types.TensorType(tf.float32, [
        tf.compat.v1.Dimension(None),
        tf.compat.v1.Dimension(10),
        tf.compat.v1.Dimension(None),
        tf.compat.v1.Dimension(10),
        tf.compat.v1.Dimension(10)
    ])
    elem = tensorflow_utils.make_whimsy_element_for_type_spec(type_spec)
    correct_elem = np.zeros([0, 10, 0, 10, 10], np.float32)
    self.assertAllClose(elem, correct_elem)

  def test_make_whimsy_element_string_tensor(self):
    type_spec = computation_types.TensorType(tf.string, [None])
    elem = tensorflow_utils.make_whimsy_element_for_type_spec(
        type_spec, none_dim_replacement=1)
    self.assertIsInstance(elem, np.ndarray)
    self.assertAllEqual(elem.shape, [1])
    self.assertEqual(elem[0], '')

  def test_make_whimsy_element_tensor_type_none_replaced_by_1(self):
    type_spec = computation_types.TensorType(tf.float32,
                                             [None, 10, None, 10, 10])
    elem = tensorflow_utils.make_whimsy_element_for_type_spec(
        type_spec, none_dim_replacement=1)
    correct_elem = np.zeros([1, 10, 1, 10, 10], np.float32)
    self.assertAllClose(elem, correct_elem)

  def test_make_whimsy_element_StructType(self):
    tensor1 = computation_types.TensorType(tf.float32, [None, 10, None, 10, 10])
    tensor2 = computation_types.TensorType(tf.int32, [10, None, 10])
    namedtuple = computation_types.StructType([('x', tensor1), ('y', tensor2)])
    unnamedtuple = computation_types.StructType([('x', tensor1),
                                                 ('y', tensor2)])
    elem = tensorflow_utils.make_whimsy_element_for_type_spec(namedtuple)
    correct_list = [
        np.zeros([0, 10, 0, 10, 10], np.float32),
        np.zeros([10, 0, 10], np.int32)
    ]
    self.assertEqual(len(elem), len(correct_list))
    for k in range(len(elem)):
      self.assertAllClose(elem[k], correct_list[k])
    unnamed_elem = tensorflow_utils.make_whimsy_element_for_type_spec(
        unnamedtuple)
    self.assertEqual(len(unnamed_elem), len(correct_list))
    for k in range(len(unnamed_elem)):
      self.assertAllClose(unnamed_elem[k], correct_list[k])

  def test_nested_structures_equal(self):
    self.assertTrue(
        tensorflow_utils.nested_structures_equal([10, 20], [10, 20]))
    self.assertFalse(tensorflow_utils.nested_structures_equal([10, 20], ['x']))

  def test_make_data_set_from_elements_in_eager_context(self):
    ds = tensorflow_utils.make_data_set_from_elements(None, [10, 20], tf.int32)
    self.assertCountEqual([x.numpy() for x in iter(ds)], [10, 20])

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [], tf.float32)
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tensor(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [],
        computation_types.TensorType(tf.float32, [None, 10]))
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(ds.element_spec,
                     tf.TensorSpec(shape=(0, 10), dtype=tf.float32))
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tuple(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [], [
            computation_types.TensorType(tf.float32, [None, 10]),
            computation_types.TensorType(tf.float32, [None, 5])
        ])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(ds.element_spec, (
        tf.TensorSpec(shape=(0, 10), dtype=tf.float32),
        tf.TensorSpec(shape=(0, 5), dtype=tf.float32),
    ))

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_ints(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [1, 2, 3, 4], tf.int32)
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(0, lambda x, y: x + y)), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'a': 1,
            'b': 2,
        }, {
            'a': 3,
            'b': 4,
        }], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_ordered_dicts(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [
            collections.OrderedDict([
                ('a', 1),
                ('b', 2),
            ]),
            collections.OrderedDict([
                ('a', 3),
                ('b', 4),
            ]),
        ], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_lists(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [
            [[1], [2]],
            [[3], [4]],
        ], [[tf.int32], [tf.int32]])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + tf.reduce_sum(y))), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_structs(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [
            structure.Struct([
                ('a', 1),
                ('b', 2),
            ]),
            structure.Struct([
                ('a', 3),
                ('b', 4),
            ]),
        ], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_lists(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'a': [1],
            'b': [2],
        }, {
            'a': [3],
            'b': [4],
        }], [('a', [tf.int32]), ('b', [tf.int32])])

    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_tensors(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'a': 1,
            'b': 2,
        }, {
            'a': 3,
            'b': 4,
        }], [('a', tf.int32), ('b', tf.int32)])

    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_np_array(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'a': np.array([1], dtype=np.int32),
            'b': np.array([2], dtype=np.int32),
        }, {
            'a': np.array([3], dtype=np.int32),
            'b': np.array([4], dtype=np.int32),
        }], [('a', (tf.int32, [1])), ('b', (tf.int32, [1]))])
    self.assertIsInstance(ds, type_conversions.TF_DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test_utils.graph_mode_test
  def test_fetch_value_in_session_with_string(self):
    x = tf.constant('abc')
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), 'abc')

  @test_utils.graph_mode_test
  def test_fetch_value_in_session_without_data_sets(self):
    x = structure.Struct([
        ('a', structure.Struct([
            ('b', tf.constant(10)),
        ])),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=10>>')

  @test_utils.graph_mode_test
  def test_fetch_value_in_session_with_empty_structure(self):
    x = structure.Struct([
        ('a', structure.Struct([
            ('b', structure.Struct([])),
        ])),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=<>>>')

  @test_utils.graph_mode_test
  def test_fetch_value_in_session_with_partially_empty_structure(self):
    x = structure.Struct([
        ('a',
         structure.Struct([
             ('b', structure.Struct([])),
             ('c', tf.constant(10)),
         ])),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=<>,c=10>>')

  def test_make_empty_list_structure_for_element_type_spec_w_tuple_dict(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    result = tensorflow_utils.make_empty_list_structure_for_element_type_spec(
        type_spec)
    self.assertEqual(
        str(result), '([], OrderedDict([(\'a\', []), (\'b\', [])]))')

  def test_append_to_list_structure_for_element_type_spec_w_tuple_dict(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    result = tuple([[], collections.OrderedDict([('a', []), ('b', [])])])
    for value in [[10, {'a': True, 'b': 30}], (40, [False, 60])]:
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, value, type_spec)
    self.assertEqual(
        str(result), '([<tf.Tensor: shape=(), dtype=int32, numpy=10>, '
        '<tf.Tensor: shape=(), dtype=int32, numpy=40>], OrderedDict([(\'a\', ['
        '<tf.Tensor: shape=(), dtype=bool, numpy=True>, '
        '<tf.Tensor: shape=(), dtype=bool, numpy=False>]), (\'b\', ['
        '<tf.Tensor: shape=(), dtype=float32, numpy=30.0>, '
        '<tf.Tensor: shape=(), dtype=float32, numpy=60.0>])]))')

  def test_append_to_list_structure_with_too_few_element_keys(self):
    type_spec = computation_types.to_type([('a', tf.int32), ('b', tf.int32)])
    result = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10}
    with self.assertRaises(TypeError):
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, value, type_spec)

  def test_append_to_list_structure_with_too_many_element_keys(self):
    type_spec = computation_types.to_type([('a', tf.int32), ('b', tf.int32)])
    result = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10, 'b': 20, 'c': 30}
    with self.assertRaises(TypeError):
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, value, type_spec)

  def test_append_to_list_structure_with_too_few_unnamed_elements(self):
    type_spec = computation_types.to_type([tf.int32, tf.int32])
    result = tuple([[], []])
    value = [10]
    with self.assertRaises(TypeError):
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, value, type_spec)

  def test_append_to_list_structure_with_too_many_unnamed_elements(self):
    type_spec = computation_types.to_type([tf.int32, tf.int32])
    result = tuple([[], []])
    value = [10, 20, 30]
    with self.assertRaises(TypeError):
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, value, type_spec)

  def test_replace_empty_leaf_lists_with_numpy_arrays(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    result = tuple([[], collections.OrderedDict([('a', []), ('b', [])])])
    result = (
        tensorflow_utils.replace_empty_leaf_lists_with_numpy_arrays(
            result, type_spec))

    expected_structure = tuple([
        np.array([], dtype=np.int32),
        collections.OrderedDict([('a', np.array([], dtype=np.bool)),
                                 ('b', np.array([], dtype=np.float32))])
    ])

    self.assertEqual(
        str(result).replace(' ', ''),
        str(expected_structure).replace(' ', ''))

  def _test_list_structure(self, type_spec, elements, expected_output_str):
    result = tensorflow_utils.make_empty_list_structure_for_element_type_spec(
        type_spec)
    for element_value in elements:
      tensorflow_utils.append_to_list_structure_for_element_type_spec(
          result, element_value, type_spec)
    result = (
        tensorflow_utils.replace_empty_leaf_lists_with_numpy_arrays(
            result, type_spec))
    self.assertEqual(
        str(result).replace(' ', ''), expected_output_str.replace(' ', ''))

  def test_list_structures_from_element_type_spec_with_none_value(self):
    self._test_list_structure([tf.int32, [('a', tf.bool), ('b', tf.float32)]],
                              [None],
                              str(
                                  tuple([
                                      np.array([], dtype=np.int32),
                                      collections.OrderedDict([
                                          ('a', np.array([], dtype=np.bool)),
                                          ('b', np.array([], dtype=np.float32)),
                                      ])
                                  ])))

  def test_list_structures_from_element_type_spec_with_int_value(self):
    self._test_list_structure(tf.int32, [1],
                              '[<tf.Tensor:shape=(),dtype=int32,numpy=1>]')

  def test_list_structures_from_element_type_spec_with_empty_dict_value(self):
    self._test_list_structure(
        computation_types.StructType([]), [{}], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_with_dict_value(self):
    self._test_list_structure([('a', tf.int32), ('b', tf.int32)], [{
        'a': 1,
        'b': 2
    }, {
        'a': 1,
        'b': 2
    }], 'OrderedDict([(\'a\',['
                              '<tf.Tensor:shape=(),dtype=int32,numpy=1>,'
                              '<tf.Tensor:shape=(),dtype=int32,numpy=1>'
                              ']),(\'b\',['
                              '<tf.Tensor:shape=(),dtype=int32,numpy=2>,'
                              '<tf.Tensor:shape=(),dtype=int32,numpy=2>'
                              '])])')

  def test_list_structures_from_element_type_spec_with_no_values(self):
    self._test_list_structure(tf.int32, [], '[]')

  def test_list_structures_from_element_type_spec_with_int_values(self):
    self._test_list_structure(
        tf.int32, [1, 2, 3], '[<tf.Tensor:shape=(),dtype=int32,numpy=1>,'
        '<tf.Tensor:shape=(),dtype=int32,numpy=2>,'
        '<tf.Tensor:shape=(),dtype=int32,numpy=3>]')

  def test_list_structures_from_element_type_spec_with_empty_dict_values(self):
    self._test_list_structure(
        computation_types.StructType([]), [{}, {}, {}], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_with_structures(self):
    self._test_list_structure(
        computation_types.StructType([('a', tf.int32)]),
        [structure.Struct([('a', 1)]),
         structure.Struct([('a', 2)])], 'OrderedDict([(\'a\', ['
        '<tf.Tensor:shape=(),dtype=int32,numpy=1>,'
        '<tf.Tensor:shape=(),dtype=int32,numpy=2>])])')

  def test_list_structures_from_element_type_spec_with_empty_anon_tuples(self):
    self._test_list_structure(
        computation_types.StructType([]),
        [structure.Struct([]), structure.Struct([])], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_w_list_of_anon_tuples(self):
    self._test_list_structure(
        computation_types.StructType([
            computation_types.StructType([('a', tf.int32)])
        ]), [[structure.Struct([('a', 1)])], [structure.Struct([('a', 2)])]],
        '(OrderedDict([(\'a\', ['
        '<tf.Tensor:shape=(),dtype=int32,numpy=1>,'
        '<tf.Tensor:shape=(),dtype=int32,numpy=2>])]),)')

  def test_make_data_set_from_elements_with_wrong_elements(self):
    with self.assertRaises(TypeError):
      tensorflow_utils.make_data_set_from_elements(
          tf.compat.v1.get_default_graph(), [{
              'a': 1
          }, {
              'a': 2
          }], tf.int32)

  def test_make_data_set_from_elements_with_odd_last_batch(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [np.array([1, 2]), np.array([3])],
        computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'x': np.array([1, 2])
        }, {
            'x': np.array([3])
        }],
        [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_make_data_set_from_elements_with_odd_all_batches(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [
            np.array([1, 2]),
            np.array([3]),
            np.array([4, 5, 6]),
            np.array([7, 8])
        ], computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'x': np.array([1, 2])
        }, {
            'x': np.array([3])
        }, {
            'x': np.array([4, 5, 6])
        }, {
            'x': np.array([7, 8])
        }],
        [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_make_data_set_from_elements_with_just_one_batch(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [np.array([1])],
        computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [{
            'x': np.array([1])
        }],
        [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_make_dataset_from_variant_tensor_constructs_dataset(self):
    with tf.Graph().as_default():
      ds = tensorflow_utils.make_dataset_from_variant_tensor(
          tf.data.experimental.to_variant(tf.data.Dataset.range(5)), tf.int64)
      self.assertIsInstance(ds, tf.data.Dataset)
      result = ds.reduce(np.int64(0), lambda x, y: x + y)
      with tf.compat.v1.Session() as sess:
        self.assertEqual(sess.run(result), 10)

  def test_make_dataset_from_variant_tensor_fails_with_bad_tensor(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        tensorflow_utils.make_dataset_from_variant_tensor(
            tf.constant(10), tf.int32)

  def test_make_dataset_from_variant_tensor_fails_with_bad_type(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        tensorflow_utils.make_dataset_from_variant_tensor(
            tf.data.experimental.to_variant(tf.data.Dataset.range(5)), 'a')

  def test_to_node_name(self):
    self.assertEqual(tensorflow_utils.to_node_name('foo'), 'foo')
    self.assertEqual(tensorflow_utils.to_node_name('^foo'), 'foo')
    self.assertEqual(tensorflow_utils.to_node_name('foo:0'), 'foo')
    self.assertEqual(tensorflow_utils.to_node_name('^foo:0'), 'foo')

  def test_get_deps_for_graph_node(self):
    # Creates a graph (double edges are regular dependencies, single edges are
    # control dependencies) like this:
    #                      foo
    #                   //      \\
    #               foo:0        foo:1
    #                  ||       //
    #       abc       bar      //
    #     //    \   //   \\   //
    #  abc:0     bak       baz
    #    ||
    #   def
    #    |
    #   ghi
    #
    graph_def = tf.compat.v1.GraphDef(node=[
        tf.compat.v1.NodeDef(name='foo', input=[]),
        tf.compat.v1.NodeDef(name='bar', input=['foo:0']),
        tf.compat.v1.NodeDef(name='baz', input=['foo:1', 'bar']),
        tf.compat.v1.NodeDef(name='bak', input=['bar', '^abc']),
        tf.compat.v1.NodeDef(name='abc', input=[]),
        tf.compat.v1.NodeDef(name='def', input=['abc:0']),
        tf.compat.v1.NodeDef(name='ghi', input=['^def']),
    ])

    def _get_deps(x):
      return ','.join(
          sorted(list(tensorflow_utils.get_deps_for_graph_node(graph_def, x))))

    self.assertEqual(_get_deps('foo'), '')
    self.assertEqual(_get_deps('bar'), 'foo')
    self.assertEqual(_get_deps('baz'), 'bar,foo')
    self.assertEqual(_get_deps('bak'), 'abc,bar,foo')
    self.assertEqual(_get_deps('abc'), '')
    self.assertEqual(_get_deps('def'), 'abc')
    self.assertEqual(_get_deps('ghi'), 'abc,def')

  def test_add_control_deps_for_init_op(self):
    # Creates a graph (double edges are regular dependencies, single edges are
    # control dependencies) like this:
    #
    #  ghi
    #   |
    #  def
    #   ||
    #  def:0         foo
    #   ||        //     ||
    #  abc      bar      ||
    #     \   //   \\    ||
    #      bak        baz
    #
    graph_def = tf.compat.v1.GraphDef(node=[
        tf.compat.v1.NodeDef(name='foo', input=[]),
        tf.compat.v1.NodeDef(name='bar', input=['foo']),
        tf.compat.v1.NodeDef(name='baz', input=['foo', 'bar']),
        tf.compat.v1.NodeDef(name='bak', input=['bar', '^abc']),
        tf.compat.v1.NodeDef(name='abc', input=['def:0']),
        tf.compat.v1.NodeDef(name='def', input=['^ghi']),
        tf.compat.v1.NodeDef(name='ghi', input=[]),
    ])
    new_graph_def = tensorflow_utils.add_control_deps_for_init_op(
        graph_def, 'abc')
    self.assertEqual(
        ','.join('{}({})'.format(node.name, ','.join(node.input))
                 for node in new_graph_def.node),
        'foo(^abc),bar(foo,^abc),baz(foo,bar,^abc),'
        'bak(bar,^abc),abc(def:0),def(^ghi),ghi()')

  def test_coerce_dataset_elements_noop(self):
    x = tf.data.Dataset.range(5)
    y = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        x, computation_types.TensorType(tf.int64))
    self.assertEqual(x.element_spec, y.element_spec)

  def test_coerce_dataset_elements_nested_structure(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return {
          'b':
              tf.cast(x, tf.int32),
          'a':
              tuple([
                  x,
                  test_tuple_type(x * 2, x * 3),
                  collections.OrderedDict([('x', x**2), ('y', x**3)])
              ]),
          'c':
              tf.cast(x, tf.float32),
      }

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)

    element_type = computation_types.StructType([
        ('a',
         computation_types.StructType([
             (None, tf.int64),
             (None, test_tuple_type(tf.int64, tf.int64)),
             (None,
              computation_types.StructType([('x', tf.int64), ('y', tf.int64)])),
         ])),
        ('b', tf.int32),
        ('c', tf.float32),
    ])

    y = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        x, element_type)

    computation_types.to_type(y.element_spec).check_equivalent_to(element_type)


class TensorFlowDeserializationTest(test_case.TestCase):

  @test_utils.graph_mode_test
  def test_deserialize_and_call_tf_computation_with_add_one(self):

    with tf.Graph().as_default() as graph:
      parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
          'x', tf.int32, graph)
      result = tf.identity(parameter_value)
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          result, graph)
    parameter_type = computation_types.TensorType(tf.int32)
    type_signature = computation_types.FunctionType(parameter_type, result_type)
    tensorflow_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
        parameter=parameter_binding,
        result=result_binding)
    serialized_type = type_serialization.serialize_type(type_signature)
    computation_proto = pb.Computation(
        type=serialized_type, tensorflow=tensorflow_proto)
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto, tf.constant(10), tf.compat.v1.get_default_graph())
    self.assertTrue(tf.is_tensor(result))
    with tf.compat.v1.Session() as sess:
      if init_op:
        sess.run(init_op)
      result_val = sess.run(result)
    self.assertEqual(result_val, 10)


if __name__ == '__main__':
  test_case.main()
