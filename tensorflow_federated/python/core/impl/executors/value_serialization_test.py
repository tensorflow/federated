# Copyright 2021, The TensorFlow Federated Authors.
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
import sys

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_test_utils

# Convenience aliases.
TensorType = computation_types.TensorType

TENSOR_SERIALIZATION_TEST_PARAMS = [
    ('numpy_scalar', np.float32(25.0), TensorType(tf.float32)),
    ('numpy_1d_tensor', np.asarray([1.0, 2.0]), TensorType(tf.float32, [2])),
    ('python_scalar', 25.0, TensorType(tf.float32)),
    ('python_1d_list', [1.0, 2.0], TensorType(tf.float32, [2])),
    ('python_2d_list', [[1.0], [2.0]], TensorType(tf.float32, [2, 1])),
]


class ValueSerializationtest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_without_hint(
      self, x, serialize_type_spec):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec)
    type_test_utils.assert_types_identical(value_type, serialize_type_spec)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype.as_numpy_dtype)
    self.assertAllEqual(x, y)

  def test_serialize_deserialize_tensor_value_unk_shape_without_hint(self):
    x = np.asarray([1., 2.])
    serialize_type_spec = TensorType(tf.float32, [None])
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec)
    type_test_utils.assert_type_assignable_from(value_type, serialize_type_spec)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_type_assignable_from(serialize_type_spec, type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype.as_numpy_dtype)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_without_hint_graph_mode(
      self, x, serialize_type_spec):
    # Test serializing non-eager tensors.
    with tf.Graph().as_default():
      tensor_x = tf.convert_to_tensor(x)
      value_proto, value_type = value_serialization.serialize_value(
          tensor_x, serialize_type_spec)
    type_test_utils.assert_types_identical(value_type, serialize_type_spec)
    y, deserialize_type_spec = value_serialization.deserialize_value(
        value_proto)
    type_test_utils.assert_types_identical(deserialize_type_spec,
                                           serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype.as_numpy_dtype)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_with_hint(self, x,
                                                        serialize_type_spec):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec)
    type_test_utils.assert_types_identical(value_type, serialize_type_spec)
    y, deserialize_type_spec = value_serialization.deserialize_value(
        value_proto, type_hint=serialize_type_spec)
    type_test_utils.assert_types_identical(deserialize_type_spec,
                                           serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype.as_numpy_dtype)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(
      ('int32_array', [1, 2, 3], np.int32),
      ('int64_array', [1, 2, 3], np.int64),
      ('float16_array', [1.0, 2.0, 3.0], np.float16),
      ('float32_array', [1.0, 2.0, 3.0], np.float32),
      ('float64_array', [1.0, 2.0, 3.0], np.float64),
      ('str_array', ['a', 'b', 'c'], np.dtype('<U1')),
  )
  def test_serialization_does_not_increase_numpy_refcount(self, value, dtype):
    # TensorFlow's internal conversion `NdarraytoTensor` adds a "delayed
    # refcount decrement" logic to the input numpy array and must be cleared
    # manually later. This test ensures that serialization doesn't cause
    # dangling references.
    array = np.asarray(value, dtype=dtype)
    type_spec = TensorType(dtype=tf.dtypes.as_dtype(dtype), shape=[3])
    # `tensorflow::NdarrayToTensor` will increase the refcount on `array` each
    # time it is called. If `TensorFlow::ClearDecrefCache` is not called this
    # will grow linearly with calls. Correct behavior is for the refcount to
    # increase by no more than one on the first use, and it should remain the
    # same or one more than the refcount before any of the calls (not refcount +
    # N).
    before_refcount = sys.getrefcount(array)
    value_serialization._serialize_tensor_value(array, type_spec)
    value_serialization._serialize_tensor_value(array, type_spec)
    value_serialization._serialize_tensor_value(array, type_spec)
    value_serialization._serialize_tensor_value(array, type_spec)
    value_serialization._serialize_tensor_value(array, type_spec)
    after_refcount = sys.getrefcount(array)
    self.assertBetween(
        after_refcount, minv=before_refcount, maxv=before_refcount + 1)

  @parameterized.named_parameters(
      ('numpy', np.str_('abc')),
      ('tensorflow', tf.constant('abc')),
  )
  def test_serialize_deserialize_string_value(self, x):
    tf_type = tf.as_dtype(x.dtype)
    type_spec = TensorType(tf_type, x.shape)
    value_proto, value_type = value_serialization.serialize_value(x, type_spec)
    type_test_utils.assert_types_identical(value_type,
                                           TensorType(tf_type, x.shape))
    y, type_spec = value_serialization.deserialize_value(
        value_proto, type_hint=type_spec)
    type_test_utils.assert_types_identical(type_spec,
                                           TensorType(tf_type, x.shape))
    self.assertIsInstance(y, bytes)
    self.assertAllEqual(x, y)

  def test_serialize_deserialize_variable_as_tensor_value(self):
    x = tf.Variable(10.0)
    type_spec = TensorType(tf.as_dtype(x.dtype), x.shape)
    value_proto, value_type = value_serialization.serialize_value(x, type_spec)
    type_test_utils.assert_types_identical(value_type, TensorType(tf.float32))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, TensorType(tf.float32))
    self.assertAllEqual(x, y)

  def test_serialize_raises_on_incompatible_dtype_float_to_int(self):
    x = tf.constant(10.0)
    with self.assertRaisesRegex(TypeError, 'Failed to serialize value'):
      value_serialization.serialize_value(x, TensorType(tf.int32))

  def test_serialize_deserialize_tensor_value_with_different_dtype(self):
    x = tf.constant(10)
    value_proto, value_type = value_serialization.serialize_value(
        x, TensorType(tf.float32))
    type_test_utils.assert_types_identical(value_type, TensorType(tf.float32))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, TensorType(tf.float32))
    self.assertEqual(y, 10.0)

  def test_serialize_deserialize_tensor_value_with_nontrivial_shape(self):
    x = tf.constant([10, 20, 30])
    value_proto, value_type = value_serialization.serialize_value(
        x, TensorType(tf.int32, [3]))
    type_test_utils.assert_types_identical(value_type,
                                           TensorType(tf.int32, [3]))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, TensorType(tf.int32, [3]))
    self.assertAllEqual(x, y)

  def test_serialize_struct_with_type_element_mismatch(self):
    x = {'a': 1}
    with self.assertRaisesRegex(
        TypeError, ('Cannot serialize a struct value of 1 elements to a struct '
                    'type requiring 2 elements.')):
      value_serialization.serialize_value(
          x, computation_types.StructType([('a', tf.int32), ('b', tf.int32)]))

  def test_serialize_sequence_bad_element_type(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize dataset .* int64\* .* float32\*.*'):
      _ = value_serialization.serialize_value(
          x, computation_types.SequenceType(tf.float32))

  def test_serialize_sequence_bad_container_type(self):
    # A Python non-ODict container whose names are not in alphabetical order.
    bad_container = collections.namedtuple('BadContainer', 'y x')
    x = tf.data.Dataset.range(5).map(lambda x: bad_container(x, x * 2))
    with self.assertRaisesRegex(ValueError, r'non-OrderedDict container'):
      _ = value_serialization.serialize_value(
          x, computation_types.SequenceType(bad_container(tf.int64, tf.int64)))

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize Python type int as .* float32\*'):
      _ = value_serialization.serialize_value(
          5, computation_types.SequenceType(tf.float32))

  @parameterized.named_parameters(('as_dataset', lambda x: x),
                                  ('as_list', list))
  def test_serialize_deserialize_sequence_of_scalars(self, dataset_fn):
    ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, computation_types.SequenceType(tf.int64))
    type_test_utils.assert_types_identical(
        value_type, computation_types.SequenceType(tf.int64))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.SequenceType(tf.int64))
    self.assertAllEqual(list(y), [x * 2 for x in range(5)])

  @parameterized.named_parameters(('as_dataset', lambda x: x),
                                  ('as_list', list))
  def test_serialize_deserialize_sequence_of_namedtuples_alphabetical_order(
      self, dataset_fn):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, tf.int32), c=tf.cast(x - 1, tf.float32))

    ds = tf.data.Dataset.range(5).map(make_test_tuple)
    ds_repr = dataset_fn(ds)
    element_type = computation_types.to_type(
        test_tuple_type(tf.int64, tf.int32, tf.float32))
    sequence_type = computation_types.SequenceType(element=element_type)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, sequence_type)
    self.assertEqual(value_type, sequence_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_equivalent(type_spec, sequence_type)
    actual_values = list(y)
    expected_values = [
        test_tuple_type(a=x * 2, b=x, c=x - 1.) for x in range(5)
    ]
    for actual, expected in zip(actual_values, expected_values):
      self.assertAllClose(actual, expected)

  def test_serialize_deserialize_sequence_of_scalars_graph_mode(self):
    with tf.Graph().as_default():
      ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
      value_proto, value_type = value_serialization.serialize_value(
          ds, computation_types.SequenceType(tf.int64))
    type_test_utils.assert_types_identical(
        value_type, computation_types.SequenceType(tf.int64))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.SequenceType(tf.int64))
    self.assertAllEqual(list(y), [x * 2 for x in range(5)])

  @parameterized.named_parameters(('as_dataset', lambda x: x),
                                  ('as_list', list))
  def test_serialize_deserialize_sequence_of_tuples(self, dataset_fn):
    ds = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, tf.int32), tf.cast(x - 1, tf.float32)))
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr,
        computation_types.SequenceType(
            element=(tf.int64, tf.int32, tf.float32)))
    expected_type = computation_types.SequenceType(
        (tf.int64, tf.int32, tf.float32))
    type_test_utils.assert_types_identical(value_type, expected_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # Only checking for equivalence, we don't have the Python container
    # after deserialization.
    type_test_utils.assert_types_equivalent(type_spec, expected_type)
    self.assertAllEqual(list(y), [(x * 2, x, x - 1.) for x in range(5)])

  @parameterized.named_parameters(('as_dataset', lambda x: x))
  def test_serialize_deserialize_sequence_of_sparse_tensors(self, dataset_fn):
    # TODO(b/258037897): Add back the as-list test case here, once we fix
    # data_set_from_elements.
    sparse_tensor = tf.SparseTensor([[0, i] for i in range(5)], list(range(5)),
                                    [1, 10])
    ds = tf.data.Dataset.from_tensor_slices(sparse_tensor)
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, computation_types.SequenceType(element=ds.element_spec))
    expected_type = computation_types.SequenceType(element=ds.element_spec)
    type_test_utils.assert_types_identical(value_type, expected_type)
    y, type_spec = value_serialization.deserialize_value(
        value_proto, type_hint=expected_type)
    # SparseTensor types lose the size of the first dimension, so check for
    # assignability.
    type_test_utils.assert_type_assignable_from(expected_type, type_spec)

    deserialized_sparse_tensor = tf.concat(
        list(y.map(tf.sparse.to_dense)), axis=0)
    original_sparse_tensor = tf.concat(list(ds.map(tf.sparse.to_dense)), axis=0)
    self.assertAllEqual(deserialized_sparse_tensor, original_sparse_tensor)

  def test_serialize_deserialize_dataset_of_ragged_tensors_with_type_hint(self):
    ragged_tensor = tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4], row_splits=[0, 2, 2, 3])
    ds = tf.data.Dataset.from_tensors(ragged_tensor)
    value_proto, value_type = value_serialization.serialize_value(
        ds, computation_types.SequenceType(element=ds.element_spec))
    expected_type = computation_types.SequenceType(ds.element_spec)
    type_test_utils.assert_types_identical(value_type, expected_type)
    # We remove the `element_type` field from the proto in order to ensure
    # that the type hint we pass to deserialize_value is used, as this contains
    # the information that the dataset uses RaggedTensors as python containers
    value_proto.sequence.ClearField('element_type')
    _, type_spec = value_serialization.deserialize_value(
        value_proto, value_type)
    type_test_utils.assert_types_identical(type_spec, expected_type)

  def test_serialize_deserialize_dataset_of_sparse_tensors_with_type_hint(self):
    sparse_tensor = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    ds = tf.data.Dataset.from_tensors(sparse_tensor)
    value_proto, value_type = value_serialization.serialize_value(
        ds, computation_types.SequenceType(element=ds.element_spec))
    expected_type = computation_types.SequenceType(ds.element_spec)
    type_test_utils.assert_types_identical(value_type, expected_type)
    # We remove the `element_type` field from the proto in order to ensure
    # that the type hint we pass to deserialize_value is used, as this contains
    # the information that the dataset uses SparseTensors as python containers
    value_proto.sequence.ClearField('element_type')
    _, type_spec = value_serialization.deserialize_value(
        value_proto, value_type)
    type_test_utils.assert_types_identical(type_spec, expected_type)

  @parameterized.named_parameters(('as_dataset', lambda x: x),
                                  ('as_list', list))
  def test_serialize_deserialize_sequence_of_ragged_tensors(self, dataset_fn):
    self.skipTest('b/237423976,b/258038191')
    ds = tf.data.Dataset.from_tensor_slices(tf.strings.split(['a b c', 'd e']))
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, computation_types.SequenceType(element=ds.element_spec))
    expected_type = computation_types.SequenceType(ds.element_spec)
    type_test_utils.assert_types_identical(value_type, expected_type)
    _, type_spec = value_serialization.deserialize_value(value_proto)
    # Only checking for equivalence, we don't have the Python container
    # after deserialization.
    type_test_utils.assert_types_equivalent(type_spec, expected_type)

  @parameterized.named_parameters(('as_dataset', lambda x: x),
                                  ('as_list', list))
  def test_serialize_deserialize_sequence_of_nested_structures(
      self, dataset_fn):

    def _make_nested_tf_structure(x):
      return collections.OrderedDict(
          b=tf.cast(x, tf.int32),
          a=tuple([
              x,
              collections.OrderedDict(u=x * 2, v=x * 3),
              collections.OrderedDict(x=x**2, y=x**3)
          ]))

    ds = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    ds_repr = dataset_fn(ds)
    element_type = computation_types.to_type(
        collections.OrderedDict(
            b=tf.int32,
            a=tuple([
                tf.int64,
                collections.OrderedDict(u=tf.int64, v=tf.int64),
                collections.OrderedDict(x=tf.int64, y=tf.int64),
            ])))
    sequence_type = computation_types.SequenceType(element=element_type)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, sequence_type)
    type_test_utils.assert_types_identical(value_type, sequence_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # These aren't the same because ser/de destroys the PyContainer
    type_test_utils.assert_types_equivalent(type_spec, sequence_type)

    def _build_expected_structure(x):
      return collections.OrderedDict(
          b=x,
          a=tuple([
              x,
              collections.OrderedDict(u=x * 2, v=x * 3),
              collections.OrderedDict(x=x**2, y=x**3)
          ]))

    actual_values = list(y)
    expected_values = [_build_expected_structure(x) for x in range(5)]
    for actual, expected in zip(actual_values, expected_values):
      self.assertEqual(type(actual), type(expected))
      self.assertAllClose(actual, expected)

  def test_serialize_deserialize_tensor_value_with_bad_shape(self):
    x = tf.constant([10, 20, 30])
    with self.assertRaises(TypeError):
      value_serialization.serialize_value(x, tf.int32)

  def test_serialize_deserialize_computation_value(self):

    @tensorflow_computation.tf_computation
    def comp():
      return tf.constant(10)

    value_proto, value_type = value_serialization.serialize_value(comp)
    self.assertEqual(value_proto.WhichOneof('value'), 'computation')
    type_test_utils.assert_types_identical(
        value_type,
        computation_types.FunctionType(parameter=None, result=tf.int32))
    comp, type_spec = value_serialization.deserialize_value(value_proto)
    # self.assertIsInstance(comp, computation_pb2.Computation)
    type_test_utils.assert_types_identical(
        type_spec,
        computation_types.FunctionType(parameter=None, result=tf.int32))

  def test_serialize_deserialize_nested_tuple_value_with_names(self):
    x = collections.OrderedDict(
        a=10, b=[20, 30], c=collections.OrderedDict(d=40))
    x_type = computation_types.to_type(
        collections.OrderedDict(
            a=tf.int32,
            b=[tf.int32, tf.int32],
            c=collections.OrderedDict(d=tf.int32)))
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # Don't assert on the Python container since it is lost in serialization.
    type_test_utils.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container(x, recursive=True))

  def test_serialize_deserialize_nested_tuple_value_without_names(self):
    x = (10, 20)
    x_type = computation_types.to_type((tf.int32, tf.int32))
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container((10, 20)))

  def test_serialize_deserialize_federated_at_clients(self):
    x = [10, 20]
    x_type = computation_types.at_clients(tf.int32)
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(
        value_type, computation_types.at_clients(tf.int32))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.at_clients(tf.int32))
    self.assertEqual(y, [10, 20])

  def test_deserialize_federated_value_with_unset_member_type(self):
    x = 10
    x_type = computation_types.to_type(tf.int32)
    member_proto, _ = value_serialization.serialize_value(x, x_type)
    fully_specified_type_at_clients = type_serialization.serialize_type(
        computation_types.at_clients(tf.int32))

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=fully_specified_type_at_clients.federated.all_equal)

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type, value=[member_proto])
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    deserialized_federated_value, deserialized_type_spec = value_serialization.deserialize_value(
        federated_value_proto)
    type_test_utils.assert_types_identical(
        deserialized_type_spec, computation_types.at_clients(tf.int32))
    self.assertEqual(deserialized_federated_value, [10])

  def test_deserialize_federated_value_with_incompatible_member_types_raises(
      self):
    x = 10
    x_type = computation_types.to_type(tf.int32)
    int_member_proto, _ = value_serialization.serialize_value(x, x_type)
    y = 10.
    y_type = computation_types.to_type(tf.float32)
    float_member_proto, _ = value_serialization.serialize_value(y, y_type)
    fully_specified_type_at_clients = type_serialization.serialize_type(
        computation_types.at_clients(tf.int32))

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=False)

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[int_member_proto, float_member_proto])
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    with self.assertRaises(TypeError):
      value_serialization.deserialize_value(federated_value_proto)

  def test_deserialize_federated_all_equal_value_takes_first_element(self):
    tensor_value_pb, _ = value_serialization.serialize_value(
        10, TensorType(tf.int32))
    num_clients = 5
    value_pb = executor_pb2.Value(
        federated=executor_pb2.Value.Federated(
            value=[tensor_value_pb] * num_clients,
            type=computation_pb2.FederatedType(
                placement=computation_pb2.PlacementSpec(
                    value=computation_pb2.Placement(
                        uri=placements.CLIENTS.uri)))))
    all_equal_clients_type_hint = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    deserialized_value, deserialized_type = value_serialization.deserialize_value(
        value_pb, all_equal_clients_type_hint)
    type_test_utils.assert_types_identical(deserialized_type,
                                           all_equal_clients_type_hint)
    self.assertAllEqual(deserialized_value, 10)

  def test_deserialize_federated_value_promotes_types(self):
    x = [10]
    smaller_type = computation_types.StructType([
        (None, computation_types.to_type(tf.int32))
    ])
    smaller_type_member_proto, _ = value_serialization.serialize_value(
        x, smaller_type)
    larger_type = computation_types.StructType([
        ('a', computation_types.to_type(tf.int32))
    ])
    larger_type_member_proto, _ = value_serialization.serialize_value(
        x, larger_type)
    type_at_clients = type_serialization.serialize_type(
        computation_types.at_clients(tf.int32))

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=type_at_clients.federated.placement, all_equal=False)

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[larger_type_member_proto, smaller_type_member_proto])
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    _, deserialized_type_spec = value_serialization.deserialize_value(
        federated_value_proto)
    type_test_utils.assert_types_identical(
        deserialized_type_spec, computation_types.at_clients(larger_type))

  def test_serialize_deserialize_federated_at_server(self):
    x = 10
    x_type = computation_types.at_server(tf.int32)
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(
        value_type, computation_types.at_server(tf.int32))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, x_type)
    self.assertEqual(y, 10)


class DatasetSerializationTest(tf.test.TestCase):

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*Dataset.* found int'):
      _ = value_serialization._serialize_dataset(5)

  def test_serialize_sequence_bytes_too_large(self):
    with self.assertRaisesRegex(ValueError,
                                r'Serialized size .* exceeds maximum allowed'):
      _ = value_serialization._serialize_dataset(
          tf.data.Dataset.range(5), max_serialized_size_bytes=0)

  def test_roundtrip_sequence_of_scalars(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    serialized_bytes = value_serialization._serialize_dataset(x)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes,
        element_type=computation_types.to_type(x.element_spec))
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(list(y), [x * 2 for x in range(5)])

  def test_coerce_odict_with_extra_elements_as_sparse_tensor_fails(self):

    tensors = (tf.constant([(1, 1)], dtype=tf.int64), (1.,),
               tf.constant((3, 3), dtype=tf.int64))

    ds = tf.data.Dataset.from_tensors(tensors)

    def _make_sparsetensor(i, v, s):
      return tf.SparseTensor(i, v, s)

    sparsetensor_ds = ds.map(_make_sparsetensor)

    def _make_odict_with_extra_element(i, v, s):
      return collections.OrderedDict(
          indices=i, values=v, dense_shape=s, extra_element=0.0)

    ds = ds.map(_make_odict_with_extra_element)

    with self.assertRaises(TypeError):
      value_serialization._coerce_deserialized_dataset_elements_to_tff_type_spec(
          ds, computation_types.to_type(sparsetensor_ds.element_spec))

  def test_roundtrip_sequence_of_sparse_tensors(self):

    tensors = (tf.constant([(1, 1)], dtype=tf.int64), (1.,),
               tf.constant((3, 3), dtype=tf.int64))

    ds = tf.data.Dataset.from_tensors(tensors)
    ds = ds.map(lambda i, v, s: tf.SparseTensor(i, v, s))  # pylint: disable=unnecessary-lambda

    serialized_value, _ = value_serialization.serialize_value(
        ds, computation_types.SequenceType(ds.element_spec))
    deserialized_ds, _ = value_serialization.deserialize_value(
        serialized_value, computation_types.SequenceType(ds.element_spec))
    self.assertEqual(ds.element_spec, deserialized_ds.element_spec)
    self.assertAllEqual(
        # We convert to dense for the comparison, since comparison on
        # SparseTensors always fails.
        list(deserialized_ds.map(tf.sparse.to_dense)),
        [tf.sparse.to_dense(tf.sparse.SparseTensor([(1, 1)], (1.,), (3, 3)))])

  def test_roundtrip_sequence_of_tuples(self):
    x = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, tf.int32), tf.cast(x - 1, tf.float32)))
    serialized_bytes = value_serialization._serialize_dataset(x)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes,
        element_type=computation_types.to_type(x.element_spec))
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(list(y), [(x * 2, x, x - 1.) for x in range(5)])

  def test_roundtrip_sequence_of_singleton_tuples(self):
    x = tf.data.Dataset.range(5).map(lambda x: (x,))
    serialized_bytes = value_serialization._serialize_dataset(x)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes,
        element_type=computation_types.to_type(x.element_spec))
    self.assertEqual(x.element_spec, y.element_spec)
    expected_values = [(x,) for x in range(5)]
    actual_values = list(y)
    self.assertAllEqual(expected_values, actual_values)

  def test_roundtrip_sequence_of_namedtuples(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, tf.int32), c=tf.cast(x - 1, tf.float32))

    x = tf.data.Dataset.range(5).map(make_test_tuple)
    serialized_bytes = value_serialization._serialize_dataset(x)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes,
        element_type=computation_types.to_type(x.element_spec))
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(
        list(y), [test_tuple_type(a=x * 2, b=x, c=x - 1.) for x in range(5)])

  def test_roundtrip_sequence_of_nested_structures(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return collections.OrderedDict(
          b=tf.cast(x, tf.int32),
          a=(
              x,
              test_tuple_type(x * 2, x * 3),
              collections.OrderedDict(x=x**2, y=x**3),
          ))

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    serialzied_bytes = value_serialization._serialize_dataset(x)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialzied_bytes,
        element_type=computation_types.to_type(x.element_spec))
    # Note: TF loses the `OrderedDict` during serialization, so the expectation
    # here is for a `dict` in the result.
    expected_element_spec = collections.OrderedDict(
        b=tf.TensorSpec([], tf.int32),
        a=(tf.TensorSpec([], tf.int64),
           test_tuple_type(
               tf.TensorSpec([], tf.int64), tf.TensorSpec([], tf.int64)),
           collections.OrderedDict(
               x=tf.TensorSpec([], tf.int64), y=tf.TensorSpec([], tf.int64))))
    self.assertEqual(y.element_spec, expected_element_spec)

    def _build_expected_structure(x):
      return collections.OrderedDict(
          b=x,
          a=(
              x,
              test_tuple_type(x * 2, x * 3),
              collections.OrderedDict(x=x**2, y=x**3),
          ))

    expected_values = (_build_expected_structure(x) for x in range(5))
    for actual, expected in zip(y, expected_values):
      self.assertAllClose(actual, expected)


class SerializeCardinalitiesTest(tf.test.TestCase):

  def test_serialize_deserialize_clients_and_server_cardinalities_roundtrip(
      self):
    client_and_server_cardinalities = {
        placements.CLIENTS: 10,
        placements.SERVER: 1
    }
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_and_server_cardinalities)
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list)
    self.assertEqual(client_and_server_cardinalities,
                     reconstructed_cardinalities)

  def test_serialize_deserialize_clients_alone(self):
    client_cardinalities = {placements.CLIENTS: 10}
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_cardinalities)
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list)
    self.assertEqual(client_cardinalities, reconstructed_cardinalities)


class SerializeNpArrayTest(tf.test.TestCase, parameterized.TestCase):

  def test_serialize_deserialize_value_proto_for_array_undefined_shape(self):
    ndarray = np.array([10])
    type_spec = computation_types.TensorType(dtype=ndarray.dtype, shape=[None])
    value_proto = value_serialization._value_proto_for_np_array(
        ndarray, type_spec)
    deserialized_array, deserialized_type = value_serialization.deserialize_value(
        value_proto)
    type_test_utils.assert_type_assignable_from(type_spec, deserialized_type)
    self.assertAllEqual(ndarray, deserialized_array)

  @parameterized.named_parameters(('str_generic', np.str_('abc')),
                                  ('bytes_generic', np.bytes_('def')))
  def test_serialize_deserialize_value_proto_for_generic(self, np_generic):
    type_spec = computation_types.TensorType(
        dtype=np_generic.dtype, shape=np_generic.shape)
    value_proto = value_serialization._value_proto_for_np_array(
        np_generic, type_spec)
    deserialized_array, deserialized_type = value_serialization.deserialize_value(
        value_proto)
    type_test_utils.assert_type_assignable_from(type_spec, deserialized_type)
    self.assertAllEqual(np_generic, deserialized_array)


class CoerceDeserializedDatasetElementsTest(tf.test.TestCase):

  def test_coerce_dataset_elements_noop(self):
    x = tf.data.Dataset.range(5)
    y = value_serialization._coerce_deserialized_dataset_elements_to_tff_type_spec(
        x, computation_types.TensorType(tf.int64))
    self.assertEqual(x.element_spec, y.element_spec)

  def test_coerce_dataset_elements_nested_structure(self):

    def _make_nested_tf_structure(x):
      return collections.OrderedDict(
          b=tf.cast(x, tf.int32),
          a=(x, collections.OrderedDict(u=x * 2, v=x * 3),
             collections.OrderedDict(x=x**2, y=x**3)),
          c=tf.cast(x, tf.float32),
      )

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)

    element_type = computation_types.StructType([
        ('a',
         computation_types.StructType([
             (None, tf.int64),
             (None,
              computation_types.to_type(
                  collections.OrderedDict(u=tf.int64, v=tf.int64))),
             (None,
              computation_types.to_type(
                  collections.OrderedDict(x=tf.int64, y=tf.int64))),
         ])),
        ('b', tf.int32),
        ('c', tf.float32),
    ])

    y = value_serialization._coerce_deserialized_dataset_elements_to_tff_type_spec(
        x, element_type)

    computation_types.to_type(y.element_spec).check_equivalent_to(element_type)


if __name__ == '__main__':
  tf.test.main()
