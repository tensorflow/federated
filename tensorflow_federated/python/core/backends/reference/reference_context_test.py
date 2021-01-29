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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.reference import reference_context
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import intrinsic_factory
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks as bb
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


def zero_for(type_spec, context_stack):
  type_spec = computation_types.to_type(type_spec)
  return value_impl.ValueImpl(
      building_block_factory.create_generic_constant(type_spec, 0),
      context_stack)


class ReferenceContextTest(test_case.TestCase, parameterized.TestCase):

  def test_computed_value(self):
    v = reference_context.ComputedValue(10, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertEqual(v.value, 10)

  def test_to_representation_for_type_with_tensor_type(self):
    self.assertEqual(
        reference_context.to_representation_for_type(10, tf.int32), 10)
    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type(0.1, tf.int32)
    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type([], tf.int32)

  def test_to_representation_for_type_with_named_tuple_type(self):
    foo = structure.Struct([('x', 10), ('y', 20)])
    self.assertEqual(
        reference_context.to_representation_for_type(foo, [('x', tf.int32),
                                                           ('y', tf.int32)]),
        foo)
    foo = structure.Struct([
        ('x', structure.Struct([(None, 10), (None, 20)])),
        ('y', 30),
    ])
    self.assertEqual(
        reference_context.to_representation_for_type(
            foo, [('x', [tf.int32, tf.int32]), ('y', tf.int32)]), foo)
    self.assertEqual(
        reference_context.to_representation_for_type(
            structure.Struct([('x', [10, 20]), ('y', 30)]),
            [('x', [tf.int32, tf.int32]), ('y', tf.int32)]),
        structure.Struct([('x', structure.Struct([(None, 10), (None, 20)])),
                          ('y', 30)]))
    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type(10, [tf.int32, tf.int32])

    unordered_dict = {'a': 10, 'b': 20}
    self.assertEqual(
        str(
            reference_context.to_representation_for_type(
                unordered_dict, [('a', tf.int32), ('b', tf.int32)])),
        '<a=10,b=20>')
    self.assertEqual(
        str(
            reference_context.to_representation_for_type(
                unordered_dict, [('b', tf.int32), ('a', tf.int32)])),
        '<b=20,a=10>')

  def test_to_representation_for_type_with_sequence_type(self):
    foo = [1, 2, 3]
    self.assertEqual(
        reference_context.to_representation_for_type(
            foo, computation_types.SequenceType(tf.int32)), foo)

  def test_to_representation_for_type_with_sequence_type_empty_tensor_slices(
      self):
    ds = tf.data.Dataset.from_tensor_slices([])
    self.assertEqual(
        reference_context.to_representation_for_type(
            ds, computation_types.SequenceType(tf.float32)), [])
    with self.assertRaisesRegex(TypeError, 'not assignable to expected type'):
      reference_context.to_representation_for_type(
          ds, computation_types.SequenceType(tf.string))

  def test_to_representation_for_type_with_sequence_type_empty_generator(self):

    # A dummy generator to be able to force the types and shapes of the empty
    # dataset.
    def _empty_generator():
      return iter(())

    ds = tf.data.Dataset.from_generator(
        _empty_generator,
        output_types=(tf.int32, tf.float32),
        output_shapes=((2,), (2, 4)))
    self.assertEqual(
        reference_context.to_representation_for_type(
            ds,
            computation_types.SequenceType(
                computation_types.StructType([
                    (None, computation_types.TensorType(tf.int32, (2,))),
                    (None, computation_types.TensorType(tf.float32, (2, 4))),
                ]))), [])

  def test_to_representation_for_type_with_function_type(self):

    def foo(x):
      self.assertIsInstance(x, reference_context.ComputedValue)
      return reference_context.ComputedValue(str(x.value), tf.string)

    self.assertIs(
        reference_context.to_representation_for_type(
            foo, computation_types.FunctionType(tf.int32, tf.string),
            lambda x, t: x), foo)

    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type(
          foo, computation_types.FunctionType(tf.int32, tf.string))

    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type(
          10, computation_types.FunctionType(tf.int32, tf.string))

  def test_to_representation_for_type_with_abstract_type(self):
    with self.assertRaises(TypeError):
      reference_context.to_representation_for_type(
          10, computation_types.AbstractType('T'))

  def test_to_representation_for_type_with_placement_type(self):
    self.assertIs(
        reference_context.to_representation_for_type(
            placement_literals.CLIENTS, computation_types.PlacementType()),
        placement_literals.CLIENTS)

  def test_to_representation_for_type_with_federated_type(self):
    self.assertEqual(
        reference_context.to_representation_for_type(
            10,
            computation_types.FederatedType(
                tf.int32, placement_literals.SERVER, all_equal=True)), 10)
    x = [1, 2, 3]
    self.assertEqual(
        reference_context.to_representation_for_type(
            x,
            computation_types.FederatedType(
                tf.int32, placement_literals.CLIENTS, all_equal=False)), x)

  def test_stamp_computed_value_into_graph_with_undefined_tensor_dims(self):
    v_type = computation_types.TensorType(tf.int32, [None])
    v_value = np.array([1, 2, 3], dtype=np.int32)
    v = reference_context.ComputedValue(v_value, v_type)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_context.stamp_computed_value_into_graph(v, graph)
      with tf.compat.v1.Session(graph=graph) as sess:
        v_result = tensorflow_utils.fetch_value_in_session(sess, stamped_v)
    self.assertTrue(np.array_equal(v_result, np.array([1, 2, 3])))

  def test_stamp_computed_value_into_graph_with_tuples_of_tensors(self):
    v_val = structure.Struct([('x', 10), ('y', structure.Struct([('z', 0.6)]))])
    v_type = [('x', tf.int32), ('y', [('z', tf.float32)])]
    v = reference_context.ComputedValue(
        reference_context.to_representation_for_type(v_val, v_type), v_type)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_context.stamp_computed_value_into_graph(v, graph)
      with tf.compat.v1.Session(graph=graph) as sess:
        stampped_v_val = tensorflow_utils.fetch_value_in_session(
            sess, stamped_v)
    elements = structure.to_elements(stampped_v_val)
    self.assertEqual(elements[0], ('x', 10))
    self.assertEqual(elements[1][0], 'y')
    nested_elements = structure.to_elements(elements[1][1])
    self.assertEqual(nested_elements[0][0], 'z')
    self.assertAlmostEqual(nested_elements[0][1], 0.6)

  def test_computation_context_resolve_reference(self):
    c1 = reference_context.ComputationContext()
    c2 = reference_context.ComputationContext(
        c1, {'foo': reference_context.ComputedValue(10, tf.int32)})
    c3 = reference_context.ComputationContext(
        c2, {'bar': reference_context.ComputedValue(11, tf.int32)})
    c4 = reference_context.ComputationContext(c3)
    c5 = reference_context.ComputationContext(
        c4, {'foo': reference_context.ComputedValue(12, tf.int32)})
    self.assertRaises(ValueError, c1.resolve_reference, 'foo')
    self.assertEqual(c2.resolve_reference('foo').value, 10)
    self.assertEqual(c3.resolve_reference('foo').value, 10)
    self.assertEqual(c4.resolve_reference('foo').value, 10)
    self.assertEqual(c5.resolve_reference('foo').value, 12)
    self.assertRaises(ValueError, c1.resolve_reference, 'bar')
    self.assertRaises(ValueError, c2.resolve_reference, 'bar')
    self.assertEqual(c3.resolve_reference('bar').value, 11)
    self.assertEqual(c4.resolve_reference('bar').value, 11)
    self.assertEqual(c5.resolve_reference('bar').value, 11)

  def test_computation_context_get_cardinality(self):
    c1 = reference_context.ComputationContext(None, None,
                                              {placement_literals.CLIENTS: 10})
    self.assertEqual(c1.get_cardinality(placement_literals.CLIENTS), 10)
    with self.assertRaises(ValueError):
      c1.get_cardinality(placement_literals.SERVER)
    c2 = reference_context.ComputationContext(c1)
    self.assertEqual(c2.get_cardinality(placement_literals.CLIENTS), 10)

  def test_tensorflow_computation_with_no_argument(self):

    @computations.tf_computation()
    def foo():
      return 1

    self.assertEqual(foo(), 1)

  def test_tensorflow_computation_with_string(self):

    @computations.tf_computation()
    def foo():
      return 'abc'

    self.assertEqual(foo(), 'abc')

  def test_tensorflow_computation_with_constant(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    self.assertEqual(foo(10), 11)

  def test_tensorflow_computation_with_constants(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    self.assertEqual(foo(10, 20), 30)
    self.assertEqual(foo(20, 10), 30)

  def test_tensorflow_computation_with_empty_tuple(self):
    tuple_type = computation_types.StructType([])

    @computations.tf_computation(tuple_type)
    def foo(z):
      self.assertEmpty(z)
      return tf.constant(1)

    self.assertEqual(foo(()), 1)

  def test_tensorflow_computation_with_tuple_of_one_constant(self):
    tuple_type = computation_types.StructType([
        ('x', tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z.x + 1

    self.assertEqual(foo((10,)), 11)

  # This is the same as test_tensorflow_computation_with_tuple_of_one_constant
  # above, but does not have a name on the types. This behavior may change in
  # the future, this unittest
  def test_tensorflow_computation_with_tuple_of_one_unnamed_constant(self):
    tuple_type = computation_types.StructType([
        (None, tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z[0] + 1

    self.assertEqual(foo((10,)), 11)

  def test_tensorflow_computation_with_tuple_of_constants(self):
    tuple_type = computation_types.StructType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z.x + z.y

    self.assertEqual(foo((10, 20)), 30)
    self.assertEqual(foo((20, 10)), 30)

  def test_tensorflow_computation_with_tuple_of_empty_tuples(self):
    tuple_type = computation_types.StructType([])
    tuple_group_type = computation_types.StructType([
        ('a', tuple_type),
        ('b', tuple_type),
    ])

    @computations.tf_computation(tuple_group_type)
    def foo(z):
      self.assertEmpty(z.a)
      self.assertEmpty(z.b)
      return tf.constant(1)

    self.assertEqual(foo(((), ())), 1)

  def test_tensorflow_computation_with_tuple_of_tuples(self):
    tuple_type = computation_types.StructType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])
    tuple_group_type = computation_types.StructType([
        ('a', tuple_type),
        ('b', tuple_type),
    ])

    @computations.tf_computation(tuple_group_type)
    def foo(z):
      return z.a.x + z.a.y + z.b.x + z.b.y

    self.assertEqual(foo(((10, 20), (30, 40))), 100)
    self.assertEqual(foo(((40, 30), (20, 10))), 100)

  def test_tensorflow_computation_with_tuple_of_sequences(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    tuple_type = computation_types.StructType([
        ('a', sequence_type),
        ('b', sequence_type),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      value1 = z.a.reduce(0, lambda x, y: x + y)
      value2 = z.b.reduce(0, lambda x, y: x + y)
      return value1 + value2

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    # pylint: disable=too-many-function-args
    self.assertEqual(foo(ds1, ds2), 100)
    # pylint: enable=too-many-function-args
    self.assertEqual(foo([ds1, ds2]), 100)
    self.assertEqual(foo((ds1, ds2)), 100)

  def test_tensorflow_computation_output_nested_structure(self):
    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a'])

    @computations.tf_computation
    def foo():
      return test_named_tuple(tf.constant(10.0))

    result = foo()
    self.assertIsInstance(result, test_named_tuple)
    self.assertEqual(result, test_named_tuple(10.0))

  def test_computation_with_batched_federated_int_sequence(self):
    ds1_shape = tf.TensorShape([None])
    sequence_type = computation_types.SequenceType(
        computation_types.TensorType(tf.int32, ds1_shape))
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placement_literals.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + tf.reduce_sum(y))
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20]).batch(1)
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40]).batch(1)

    self.assertEqual(bar([ds1, ds2]), [30, 70])

  def test_computation_with_int_sequence_raises(self):
    batch_size = 1
    ds1_shape = tf.TensorShape([batch_size])
    sequence_type = computation_types.SequenceType(
        computation_types.TensorType(tf.int32, ds1_shape))
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placement_literals.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + tf.reduce_sum(y))
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20]).batch(batch_size)
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40]).batch(batch_size)

    with self.assertRaisesRegex(ValueError, 'Please pass a list'):
      bar(ds1)
    with self.assertRaisesRegex(ValueError, 'Please pass a list'):
      bar(ds2)

  def test_batching_namedtuple_dataset(self):
    batch_type = collections.namedtuple('Batch', ['x', 'y'])
    federated_sequence_type = computation_types.FederatedType(
        computation_types.SequenceType(
            batch_type(
                x=computation_types.TensorType(tf.float32, [None, 2]),
                y=computation_types.TensorType(tf.float32, [None, 1]))),
        placement_literals.CLIENTS,
        all_equal=False)

    @computations.tf_computation(federated_sequence_type.member)
    def test_batch_select_and_reduce(z):
      i = z.map(lambda x: x.y)
      return i.reduce(0., lambda x, y: x + tf.reduce_sum(y))

    @computations.federated_computation(federated_sequence_type)
    def map_y_sum(x):
      return intrinsics.federated_map(test_batch_select_and_reduce, x)

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [[1., 2.], [3., 4.]]),
            ('y', [[5.], [6.]]),
        ])).batch(1)
    self.assertEqual(map_y_sum([ds] * 5), [np.array([[11.]])] * 5)

  def test_batching_ordereddict_dataset(self):
    odict_type = collections.OrderedDict([
        ('x', computation_types.TensorType(tf.float32, [None, 2])),
        ('y', computation_types.TensorType(tf.float32, [None, 1])),
    ])

    @computations.tf_computation(odict_type)
    def test_foo(z):
      return tf.reduce_sum(z['x']) + z['y']

    self.assertEqual(
        test_foo(
            collections.OrderedDict([
                ('x', np.ones(shape=[1, 2], dtype=np.float32)),
                ('y', np.ones(shape=[1, 1], dtype=np.float32)),
            ])), 3.0)

  def test_helpful_failure_federated_int_sequence(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placement_literals.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + y)
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    with self.assertRaisesRegex(TypeError,
                                'only with a single positional argument'):
      # pylint: disable=too-many-function-args
      _ = bar(ds1, ds2)
      # pylint: enable=too-many-function-args

  def test_graph_mode_dataset_fails_well(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placement_literals.CLIENTS)

    with tf.Graph().as_default():

      @computations.tf_computation(sequence_type)
      def foo(z):
        value1 = z.reduce(0, lambda x, y: x + y)
        return value1

      @computations.federated_computation(federated_type)
      def bar(x):
        return intrinsics.federated_map(foo, x)

      ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
      ds2 = tf.data.Dataset.from_tensor_slices([30, 40])
      with self.assertRaisesRegex(
          ValueError, 'outside of eager mode is not currently supported.'):
        bar([ds1, ds2])

  def test_tensorflow_computation_with_tuples_of_constants(self):
    tuple_type = computation_types.StructType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type, tuple_type)
    def foo(a, b):
      return a.x + a.y + b.x + b.y

    self.assertEqual(foo((10, 20), (30, 40)), 100)
    self.assertEqual(foo((40, 30), (20, 10)), 100)

  def test_tensorflow_computation_with_result_sequence_anon_tuple(self):
    input_type = computation_types.SequenceType(
        computation_types.StructType([('a', tf.int64)]))

    @computations.tf_computation(input_type)
    def foo(dataset):
      return dataset.map(lambda x: tf.nest.map_structure(lambda v: v * 2, x))

    # Note: if `collections.OrderedDict` changes here, that is completely okay.
    # This test merely serves as documentation of the current state of the
    # world. The `collections.OrderedDict` addition is the result of recording
    # the Python container that carries the result of `foo`.
    self.assertEqual(
        foo.type_signature.result,
        computation_types.SequenceType(
            computation_types.StructWithPythonType([('a', tf.int64)],
                                                   collections.OrderedDict)))
    input_value = tf.data.Dataset.range(5).map(
        lambda x: collections.OrderedDict(a=x))
    self.assertAllEqual([i for i in foo(input_value)],
                        [collections.OrderedDict(a=i * 2) for i in range(5)])

  def test_tensorflow_computation_with_result_sequence_py_container(self):

    @computations.tf_computation()
    def foo():
      return tf.data.Dataset.range(5).map(
          lambda x: collections.OrderedDict(a=x))

    self.assertEqual(
        foo.type_signature.result,
        computation_types.SequenceType(
            computation_types.StructWithPythonType(
                collections.OrderedDict(a=tf.int64), collections.OrderedDict)))
    self.assertAllEqual([i for i in foo()],
                        [collections.OrderedDict(a=i) for i in range(5)])

  def test_tensorflow_computation_with_arg_empty_sequence(self):
    sequence_type = computation_types.SequenceType(tf.float32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      del ds  # Unused.
      return 1

    ds = tf.data.Dataset.from_tensor_slices([])
    self.assertEqual(foo(ds), 1)

  def test_tensorflow_computation_with_arg_sequence_of_one_constant(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(1, lambda x, y: x + y)

    ds = tf.data.Dataset.from_tensor_slices([10])

    self.assertEqual(foo(ds), 11)

  def test_tensorflow_computation_with_arg_sequence_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(0, lambda x, y: x + y)

    ds = tf.data.Dataset.from_tensor_slices([10, 20])
    self.assertEqual(foo(ds), 30)

  def test_tensorflow_computation_with_arg_sequence_of_tuples(self):
    tuple_type = computation_types.StructType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])
    sequence_type = computation_types.SequenceType(tuple_type)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(0, lambda x, y: x + y['x'] + y['y'])

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [10, 30]),
            ('y', [20, 40]),
        ]))

    self.assertEqual(foo(ds), 100)

  def test_tensorflow_computation_with_arg_sequences_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type, sequence_type)
    def foo(ds1, ds2):
      value1 = ds1.reduce(0, lambda x, y: x + y)
      value2 = ds2.reduce(0, lambda x, y: x + y)
      return value1 + value2

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    self.assertEqual(foo(ds1, ds2), 100)

  def test_tensorflow_computation_with_simple_lambda(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(tf.int32)
    def add_two(x):
      return add_one(add_one(x))

    self.assertEqual(add_two(5), 7)

  def test_tensorflow_computation_with_lambda_and_selection(self):

    @computations.federated_computation(tf.int32,
                                        computation_types.FunctionType(
                                            tf.int32, tf.int32))
    def apply_twice(x, f):
      return f(f(x))

    add_one = computations.tf_computation(lambda x: x + 1, tf.int32)

    self.assertEqual(apply_twice(5, add_one), 7)

  def test_multiply_by_scalar_with_float_and_float(self):
    self.assertEqual(
        reference_context.multiply_by_scalar(
            reference_context.ComputedValue(10.0, tf.float32), 0.5).value, 5.0)
    self.assertAlmostEqual(
        reference_context.multiply_by_scalar(
            reference_context.ComputedValue(np.float32(1.0), tf.float32),
            0.333333333333).value,
        0.3333333,
        places=3)

  def test_multiply_by_scalar_with_tuple_and_float(self):
    self.assertEqual(
        str(
            reference_context.multiply_by_scalar(
                reference_context.ComputedValue(
                    structure.Struct([
                        ('A', 10.0),
                        ('B', structure.Struct([('C', 20.0)])),
                    ]), [('A', tf.float32), ('B', [('C', tf.float32)])]),
                0.5).value), '<A=5.0,B=<C=10.0>>')

  def test_fit_argument(self):
    old_arg = reference_context.ComputedValue(
        structure.Struct([('A', 10)]),
        [('A', computation_types.at_clients(tf.int32, all_equal=True))])
    new_arg = reference_context.fit_argument(
        old_arg, [('A', computation_types.at_clients(tf.int32))],
        reference_context.ComputationContext(
            cardinalities={placement_literals.CLIENTS: 3}))
    self.assertEqual(str(new_arg.type_signature), '<A={int32}@CLIENTS>')
    self.assertEqual(new_arg.value.A, [10, 10, 10])

  def test_execute_with_nested_lambda(self):
    int32_add = bb.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(
                lambda a, b: tf.add(a, b),  # pylint: disable=unnecessary-lambda
                [tf.int32, tf.int32])))

    curried_int32_add = bb.Lambda(
        'x', tf.int32,
        bb.Lambda(
            'y', tf.int32,
            bb.Call(
                int32_add,
                bb.Struct([('a', bb.Reference('x', tf.int32)),
                           ('b', bb.Reference('y', tf.int32))]))))

    make_10 = bb.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda: tf.constant(10))))

    add_10 = bb.Call(curried_int32_add, bb.Call(make_10))

    add_10_computation = computation_impl.ComputationImpl(
        add_10.proto, context_stack_impl.context_stack)

    self.assertEqual(add_10_computation(5), 15)

  def test_execute_with_block(self):
    add_one = bb.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda x: x + 1, tf.int32)))

    make_10 = bb.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda: tf.constant(10))))

    make_13 = bb.Lambda(
        None, None,
        bb.Block([('x', bb.Call(make_10)),
                  ('x', bb.Call(add_one, bb.Reference('x', tf.int32))),
                  ('x', bb.Call(add_one, bb.Reference('x', tf.int32))),
                  ('x', bb.Call(add_one, bb.Reference('x', tf.int32)))],
                 bb.Reference('x', tf.int32)))

    make_13_computation = computation_impl.ComputationImpl(
        make_13.proto, context_stack_impl.context_stack)

    self.assertEqual(make_13_computation(), 13)

  def test_sequence_sum_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo(x):
      return intrinsics.sequence_sum(x)

    self.assertEqual(str(foo.type_signature), '(int32* -> int32)')
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_sequence_sum_with_list_of_tuples(self):

    @computations.federated_computation(
        computation_types.SequenceType([tf.int32, tf.int32]))
    def foo(x):
      return intrinsics.sequence_sum(x)

    self.assertEqual(
        str(foo.type_signature), '(<int32,int32>* -> <int32,int32>)')
    self.assertEqual(str(foo([[1, 2], [3, 4], [5, 6]])), '[9, 12]')

  def test_federated_collect_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_collect(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32*@SERVER)')
    self.assertEqual(foo([1, 2, 3]), [1, 2, 3])

  def test_federated_map_with_list_of_integers(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def bar(x):
      return intrinsics.federated_map(foo, x)

    self.assertEqual(
        str(bar.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')
    self.assertEqual(bar([1, 10, 3, 7, 2]), [2, 11, 4, 8, 3])

  def test_federated_map_all_equal_with_int(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FederatedType(
            tf.int32, placement_literals.CLIENTS, all_equal=True))
    def bar(x):
      factory = intrinsic_factory.IntrinsicFactory(
          context_stack_impl.context_stack)
      return factory.federated_map_all_equal(foo, x)

    self.assertEqual(
        str(bar.type_signature), '(int32@CLIENTS -> int32@CLIENTS)')
    self.assertEqual(bar(10), 11)

  def test_federated_map_with_int(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True))
    def bar(x):
      return intrinsics.federated_map(foo, x)

    self.assertEqual(str(bar.type_signature), '(int32@SERVER -> int32@SERVER)')
    self.assertEqual(bar(10), 11)

  def test_federated_map_with_int_sequence(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(z):
      return intrinsics.sequence_map(foo, z)

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placement_literals.SERVER,
            True))
    def baz(x):
      return intrinsics.federated_map(bar, x)

    self.assertEqual(
        str(baz.type_signature), '(int32*@SERVER -> int32*@SERVER)')
    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    self.assertEqual(baz(ds1), [11, 21])

  def test_federated_sum_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_federated_secure_sum_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_secure_sum(x, 8)

    self.assertEqual(foo.type_signature.compact_representation(),
                     '({int32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_federated_value_at_clients_and_at_server(self):

    @computations.federated_computation(tf.int32)
    def foo(x):
      return [
          intrinsics.federated_value(x, placement_literals.CLIENTS),
          intrinsics.federated_value(x, placement_literals.SERVER)
      ]

    self.assertEqual(
        str(foo.type_signature), '(int32 -> <int32@CLIENTS,int32@SERVER>)')
    self.assertEqual(str(foo(11)), '[11, 11]')

  def test_federated_computation_returns_named_tuple(self):
    test_named_tuple = collections.namedtuple('_', ['sum', 'n'])

    @computations.federated_computation()
    def foo():
      return test_named_tuple(sum=10.0, n=2)

    self.assertEqual(str(foo.type_signature), '( -> <sum=float32,n=int32>)')
    self.assertEqual(foo(), test_named_tuple(10.0, 2))

  def test_federated_computation_returns_ordered_dict(self):

    @computations.federated_computation()
    def foo():
      return collections.OrderedDict([('A', 1.0), ('B', 2)])

    self.assertEqual(str(foo.type_signature), '( -> <A=float32,B=int32>)')
    result = foo()
    self.assertIsInstance(result, collections.OrderedDict)
    self.assertDictEqual(result, {'A': 1.0, 'B': 2})

  def test_generic_zero_with_scalar_int32_tensor_type(self):

    @computations.federated_computation
    def foo():
      return zero_for(tf.int32, context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> int32)')
    self.assertEqual(foo(), 0)

  def test_generic_zero_with_two_dimensional_float32_tensor_type(self):

    @computations.federated_computation
    def foo():
      return zero_for(
          computation_types.TensorType(tf.float32, [2, 3]),
          context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> float32[2,3])')
    foo_result = foo()
    self.assertEqual(type(foo_result), np.ndarray)
    self.assertTrue(np.array_equal(foo_result, [[0., 0., 0.], [0., 0., 0.]]))

  def test_generic_zero_with_tuple_type(self):

    @computations.federated_computation
    def foo():
      return zero_for([('A', tf.int32), ('B', tf.float32)],
                      context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> <A=int32,B=float32>)')
    self.assertEqual(str(foo()), '<A=0,B=0.0>')

  def test_generic_zero_with_federated_int_on_server(self):

    @computations.federated_computation
    def foo():
      return zero_for(
          computation_types.FederatedType(
              tf.int32, placement_literals.SERVER, all_equal=True),
          context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> int32@SERVER)')
    self.assertEqual(foo(), 0)

  def test_generic_plus_with_integers(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(tf.int32, tf.int32)
    def foo(x, y):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, y])

    self.assertEqual(str(foo.type_signature), '(<x=int32,y=int32> -> int32)')
    self.assertEqual(foo(2, 3), 5)

  def test_generic_plus_with_tuples(self):
    type_spec = [('A', tf.int32), ('B', tf.float32)]
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(type_spec, type_spec)
    def foo(x, y):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, y])

    self.assertEqual(
        str(foo.type_signature),
        '(<x=<A=int32,B=float32>,y=<A=int32,B=float32>> -> <A=int32,B=float32>)'
    )
    foo_result = foo([2, 0.1], [3, 0.2])
    self.assertIsInstance(foo_result, structure.Struct)
    self.assertSameElements(dir(foo_result), ['A', 'B'])
    self.assertEqual(foo_result.A, 5)
    self.assertAlmostEqual(foo_result.B, 0.3, places=2)

  def test_sequence_map_with_list_of_integers(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(x):
      return intrinsics.sequence_map(foo, x)

    self.assertEqual(str(bar.type_signature), '(int32* -> int32*)')
    self.assertEqual(bar([1, 10, 3]), [2, 11, 4])

  def test_sequence_reduce_with_integers(self):

    @computations.tf_computation(tf.int32, tf.float32)
    def foo(x, y):
      return x + tf.cast(y > 0.5, tf.int32)

    @computations.federated_computation(
        computation_types.SequenceType(tf.float32))
    def bar(x):
      return intrinsics.sequence_reduce(x, 0, foo)

    self.assertEqual(str(bar.type_signature), '(float32* -> int32)')
    self.assertEqual(bar([0.1, 0.6, 0.2, 0.4, 0.8]), 2)

  def test_sequence_reduce_with_namedtuples(self):
    accumulator_type = collections.namedtuple('_', ['sum', 'product'])(
        sum=tf.int32, product=tf.int32)

    @computations.tf_computation(accumulator_type, tf.int32)
    def foo(accumulator, x):
      return collections.OrderedDict([('sum', accumulator.sum + x),
                                      ('product', accumulator.product * x)])

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(x):
      zero = collections.OrderedDict([('sum', 0), ('product', 1)])
      return intrinsics.sequence_reduce(x, zero, foo)

    self.assertEqual(
        str(bar.type_signature), '(int32* -> <sum=int32,product=int32>)')
    self.assertEqual(
        str(bar([1, 2, 3, 4, 5])),
        "OrderedDict([('sum', 15), ('product', 120)])")

  def test_federated_reduce_with_integers(self):

    @computations.tf_computation(tf.int32, tf.float32)
    def foo(x, y):
      return x + tf.cast(y > 0.5, tf.int32)

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    def bar(x):
      return intrinsics.federated_reduce(x, 0, foo)

    self.assertEqual(
        str(bar.type_signature), '({float32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(bar([0.1, 0.6, 0.2, 0.4, 0.8]), 2)

  def test_federated_mean_with_floats(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x)

    self.assertEqual(
        str(foo.type_signature), '({float32}@CLIENTS -> float32@SERVER)')
    self.assertEqual(foo([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

  def test_federated_mean_with_tuples(self):

    @computations.federated_computation(
        computation_types.FederatedType([('A', tf.float32), ('B', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x)

    self.assertEqual(
        str(foo.type_signature),
        '({<A=float32,B=float32>}@CLIENTS -> <A=float32,B=float32>@SERVER)')
    self.assertEqual(
        str(
            foo([{
                'A': 1.0,
                'B': 5.0
            }, {
                'A': 2.0,
                'B': 6.0
            }, {
                'A': 3.0,
                'B': 7.0
            }])), '<A=2.0,B=6.0>')

  def test_federated_zip_at_server(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True),
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True)
    ])
    def foo(x):
      return intrinsics.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature),
        '(<int32@SERVER,int32@SERVER> -> <int32,int32>@SERVER)')

    self.assertEqual(str(foo(5, 6)), '[5, 6]')  # pylint: disable=too-many-function-args

  def assert_list(self, value, expected: str):
    """Assert that a value is a list with a given string representation."""
    self.assertIsInstance(value, list)
    value_str = ','.join(str(x) for x in value).replace(' ', '')
    self.assertEqual(value_str, expected)

  def test_federated_zip_at_clients(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS),
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS)
    ])
    def foo(x):
      return intrinsics.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,{int32}@CLIENTS> -> {<int32,int32>}@CLIENTS)')
    foo_result = foo([[1, 2, 3], [4, 5, 6]])
    self.assert_list(foo_result, '[1,4],[2,5],[3,6]')

  def test_federated_aggregate_with_integers(self):
    test_named_tuple = collections.namedtuple('_', ['sum', 'n'])
    accu_type = computation_types.to_type(
        test_named_tuple(sum=tf.int32, n=tf.int32))

    @computations.tf_computation(accu_type, tf.int32)
    def accumulate(a, x):
      return collections.OrderedDict([('sum', a.sum + x), ('n', a.n + 1)])

    @computations.tf_computation(accu_type, accu_type)
    def merge(a, b):
      return collections.OrderedDict([('sum', a.sum + b.sum), ('n', a.n + b.n)])

    @computations.tf_computation(accu_type)
    def report(a):
      return tf.cast(a.sum, tf.float32) / tf.cast(a.n, tf.float32)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_aggregate(
          x, collections.OrderedDict([('sum', 0), ('n', 0)]), accumulate, merge,
          report)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> float32@SERVER)')
    self.assertEqual(foo([1, 2, 3, 4, 5, 6, 7]), 4.0)

  def test_federated_weighted_average_with_floats(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS),
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    def foo(v, w):
      return intrinsics.federated_mean(v, w)

    self.assertEqual(
        str(foo.type_signature),
        '(<v={float32}@CLIENTS,w={float32}@CLIENTS> -> float32@SERVER)')
    self.assertEqual(foo([5.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 3.0)

  def test_federated_broadcast_without_data_on_clients(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True))
    def foo(x):
      return intrinsics.federated_broadcast(x)

    self.assertEqual(str(foo.type_signature), '(int32@SERVER -> int32@CLIENTS)')
    self.assertEqual(foo(10), 10)

  def test_federated_broadcast_zipped_with_client_data(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS),
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True))
    def foo(x, y):
      return intrinsics.federated_zip([x, intrinsics.federated_broadcast(y)])

    self.assertEqual(
        str(foo.type_signature),
        '(<x={int32}@CLIENTS,y=int32@SERVER> -> {<int32,int32>}@CLIENTS)')

    foo_result = foo([1, 2, 3], 10)
    self.assert_list(foo_result, '[1,10],[2,10],[3,10]')

  @parameterized.named_parameters(
      ('federated', computations.federated_computation),
      ('tf', computations.tf_computation),
  )
  def test_federated_eval_at_clients_simple_number(self, comp_wrapper):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      del x
      return_five = comp_wrapper(lambda: 5)
      return intrinsics.federated_eval(return_five, placement_literals.CLIENTS)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')
    foo_result = foo([0, 0, 0])
    self.assert_list(foo_result, '5,5,5')

  @parameterized.named_parameters(
      ('federated', computations.federated_computation),
      ('tf', computations.tf_computation),
  )
  def test_federated_eval_at_server_simple_number(self, comp_wrapper):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      del x
      return_five = comp_wrapper(lambda: 5)
      return intrinsics.federated_eval(return_five, placement_literals.SERVER)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(foo([0, 0, 0]), 5)

  def test_federated_eval_at_clients_random(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      del x
      rand = computations.tf_computation(lambda: tf.random.normal([]))
      return intrinsics.federated_eval(rand, placement_literals.CLIENTS)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {float32}@CLIENTS)')
    foo_result = foo([0, 0, 0])
    self.assertIsInstance(foo_result, list)
    self.assertLen(foo_result, 3)
    self.assertNotEqual(foo_result[0], foo_result[1])
    self.assertNotEqual(foo_result[1], foo_result[2])

  def test_with_unequal_tensor_types(self):

    @computations.tf_computation
    def foo():
      return tf.data.Dataset.range(5).map(lambda _: tf.constant(10.0)).batch(1)

    self.assertEqual(str(foo.type_signature), '( -> float32[?]*)')
    foo_result = foo()
    self.assert_list(foo_result, '[10.],[10.],[10.],[10.],[10.]')

  def test_numpy_cast(self):
    self.assertEqual(
        reference_context.numpy_cast(True, tf.bool, tf.TensorShape([])),
        np.bool_(True))
    self.assertEqual(
        reference_context.numpy_cast(10, tf.int32, tf.TensorShape([])),
        np.int32(10))
    self.assertEqual(
        reference_context.numpy_cast(0.3333333333333333333333333, tf.float32,
                                     tf.TensorShape([])),
        np.float32(0.3333333333333333333333333))
    self.assertTrue(
        np.array_equal(
            reference_context.numpy_cast([[1, 2], [3, 4]], tf.int32,
                                         tf.TensorShape([2, 2])),
            np.array([[1, 2], [3, 4]])))

  def test_sum_of_squares(self):
    int32_sequence = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(tf.int32, tf.int32)
    def square_error(x, y):
      return tf.pow(x - y, 2)

    @computations.federated_computation(tf.int32, int32_sequence)
    def sum_of_square_errors(x, y):

      @computations.federated_computation(tf.int32)
      def mapping_fn(v):
        return square_error(x, v)

      return intrinsics.sequence_sum(intrinsics.sequence_map(mapping_fn, y))

    self.assertEqual(sum_of_square_errors(10, [11, 8, 13]), 14)

  def test_variable_input(self):
    assert tf.executing_eagerly()
    v = tf.Variable(10.0)

    @computations.tf_computation
    def add_one(x):
      return x + 1.0

    self.assertEqual(add_one(v), 11.0)


class UnwrapPlacementIntegrationTest(test_case.TestCase):

  def test_unwrap_placement_with_federated_map_executes_correctly(self):
    int_ref = bb.Reference('x', tf.int32)
    int_id = bb.Lambda('x', tf.int32, int_ref)
    fed_ref = bb.Reference(
        'x',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)
    lambda_wrapping_id = bb.Lambda('x', fed_ref.type_signature,
                                   second_applied_id)
    lambda_wrapping_placement_unwrapped = bb.Lambda('x', fed_ref.type_signature,
                                                    placement_unwrapped)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_id)
    executable_unwrapped = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(executable_identity([k]), executable_unwrapped([k]))

  def test_unwrap_placement_with_federated_apply_executes_correctly(self):
    int_ref = bb.Reference('x', tf.int32)
    int_id = bb.Lambda('x', tf.int32, int_ref)
    fed_ref = bb.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)
    lambda_wrapping_id = bb.Lambda('x', fed_ref.type_signature,
                                   second_applied_id)
    lambda_wrapping_placement_unwrapped = bb.Lambda('x', fed_ref.type_signature,
                                                    placement_unwrapped)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_id)
    executable_unwrapped = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(executable_identity(k), executable_unwrapped(k))

  def test_unwrap_placement_with_federated_zip_at_server_executes_correctly(
      self):
    fed_tuple = bb.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placement_literals.SERVER))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)

    lambda_wrapping_zip = bb.Lambda('tup', fed_tuple.type_signature, zipped)
    lambda_wrapping_placement_unwrapped = bb.Lambda('tup',
                                                    fed_tuple.type_signature,
                                                    placement_unwrapped)
    executable_zip = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_zip)
    executable_unwrapped = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(
          executable_zip([k, k * 1., k, k * 1.]),
          executable_unwrapped([k, k * 1., k, k * 1.]))

  def test_unwrap_placement_with_federated_zip_at_clients_executes_correctly(
      self):
    fed_tuple = bb.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placement_literals.CLIENTS))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)
    lambda_wrapping_zip = bb.Lambda('tup', fed_tuple.type_signature, zipped)
    lambda_wrapping_placement_unwrapped = bb.Lambda('tup',
                                                    fed_tuple.type_signature,
                                                    placement_unwrapped)
    executable_zip = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_zip)
    executable_unwrapped = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(
          executable_zip([[k, k * 1., k, k * 1.]]),
          executable_unwrapped([[k, k * 1., k, k * 1.]]))


class MergeTupleIntrinsicsIntegrationTest(test_case.TestCase):

  def test_merge_tuple_intrinsics_executes_with_federated_aggregate(self):
    value_type = computation_types.FederatedType(tf.int32,
                                                 placement_literals.CLIENTS)
    ref_type = computation_types.StructType(
        (value_type, tf.float32, tf.float32, tf.float32, tf.bool))
    ref = bb.Reference('a', ref_type)
    value = bb.Selection(ref, index=0)
    zero = bb.Selection(ref, index=1)
    accumulate_type = computation_types.StructType((tf.float32, tf.int32))
    accumulate_result = bb.Selection(ref, index=2)
    accumulate = bb.Lambda('b', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((tf.float32, tf.float32))
    merge_result = bb.Selection(ref, index=3)
    merge = bb.Lambda('c', merge_type, merge_result)
    report_result = bb.Selection(ref, index=4)
    report = bb.Lambda('d', tf.float32, report_result)
    called_intrinsic = building_block_factory.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    tup = bb.Struct((called_intrinsic, called_intrinsic))
    comp = bb.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    comp_impl = computation_wrapper_instances.building_block_to_computation(
        comp)
    transformed_comp_impl = computation_wrapper_instances.building_block_to_computation(
        transformed_comp)

    self.assertEqual(
        comp_impl(((1,), 1.0, 2.0, 3.0, True)),
        transformed_comp_impl(((2,), 4.0, 5.0, 6.0, True)))

  def test_merge_tuple_intrinsics_executes_with_federated_apply(self):
    ref_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.SERVER)
    ref = bb.Reference('a', ref_type)
    fn = compiler_test_utils.create_identity_function('b')
    arg = ref
    called_intrinsic = building_block_factory.create_federated_apply(fn, arg)
    tup = bb.Struct((called_intrinsic, called_intrinsic))
    comp = bb.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_APPLY.uri)

    comp_impl = computation_wrapper_instances.building_block_to_computation(
        comp)
    transformed_comp_impl = computation_wrapper_instances.building_block_to_computation(
        transformed_comp)

    self.assertEqual(comp_impl(1), transformed_comp_impl(1))

  def test_merge_tuple_intrinsics_executes_with_federated_broadcast(self):
    self.skipTest('b/135279151')
    ref_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.SERVER)
    ref = bb.Reference('a', ref_type)
    called_intrinsic = building_block_factory.create_federated_broadcast(ref)
    tup = bb.Struct((called_intrinsic, called_intrinsic))
    comp = bb.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    comp_impl = computation_wrapper_instances.building_block_to_computation(
        comp)
    transformed_comp_impl = computation_wrapper_instances.building_block_to_computation(
        transformed_comp)

    self.assertEqual(comp_impl(10), transformed_comp_impl(10))

  def test_merge_tuple_intrinsics_executes_with_federated_map(self):
    ref_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    ref = bb.Reference('a', ref_type)
    fn = compiler_test_utils.create_identity_function('b')
    arg = ref
    called_intrinsic = building_block_factory.create_federated_map(fn, arg)
    tup = bb.Struct((called_intrinsic, called_intrinsic))
    comp = bb.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    comp_impl = computation_wrapper_instances.building_block_to_computation(
        comp)
    transformed_comp_impl = computation_wrapper_instances.building_block_to_computation(
        transformed_comp)

    self.assertEqual(comp_impl((1,)), transformed_comp_impl((1,)))

  def test_cardinalities_inferred_before_function_ingested(self):

    @computations.federated_computation(
        computation_types.FederatedType(
            (computation_types.SequenceType(tf.string)),
            placement_literals.CLIENTS))
    def compute_clients(examples):
      del examples  # Unused.
      return intrinsics.federated_sum(
          intrinsics.federated_value(1, placement_literals.CLIENTS))

    self.assertEqual(compute_clients([['a', 'b', 'c'], ['a'], ['a', 'b']]), 3)


if __name__ == '__main__':
  reference_context.set_reference_context()
  test_case.main()
