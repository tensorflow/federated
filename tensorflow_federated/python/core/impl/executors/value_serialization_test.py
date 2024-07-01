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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_test_utils

# Convenience aliases.
TensorType = computation_types.TensorType

TENSOR_SERIALIZATION_TEST_PARAMS = [
    ('numpy_scalar', np.float32(25.0), TensorType(np.float32)),
    ('numpy_1d_tensor', np.asarray([1.0, 2.0]), TensorType(np.float32, [2])),
    ('python_scalar', 25.0, TensorType(np.float32)),
    ('python_1d_list', [1.0, 2.0], TensorType(np.float32, [2])),
    ('python_2d_list', [[1.0], [2.0]], TensorType(np.float32, [2, 1])),
]


class ValueSerializationtest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_without_hint(
      self, x, serialize_type_spec
  ):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    type_test_utils.assert_types_identical(value_type, serialize_type_spec)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    self.assertAllEqual(x, y)

  def test_serialize_deserialize_tensor_value_unk_shape_without_hint(self):
    x = np.asarray([1.0, 2.0])
    serialize_type_spec = TensorType(np.float32, [None])
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    type_test_utils.assert_type_assignable_from(value_type, serialize_type_spec)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_type_assignable_from(serialize_type_spec, type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_with_hint(
      self, x, serialize_type_spec
  ):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    type_test_utils.assert_types_identical(value_type, serialize_type_spec)
    y, deserialize_type_spec = value_serialization.deserialize_value(
        value_proto, type_hint=serialize_type_spec
    )
    type_test_utils.assert_types_identical(
        deserialize_type_spec, serialize_type_spec
    )
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(
      ('str', 'abc', TensorType(np.str_), b'abc'),
      ('bytes', b'abc', TensorType(np.bytes_), b'abc'),
      (
          'bytes_null_terminated',
          b'abc\x00\x00',
          TensorType(np.bytes_),
          b'abc\x00\x00',
      ),
      ('numpy_str', np.str_('abc'), TensorType(np.str_), b'abc'),
      ('numpy_bytes', np.bytes_(b'abc'), TensorType(np.bytes_), b'abc'),
      (
          'numpy_bytes_null_termianted',
          np.bytes_(b'abc\x00\x00'),
          TensorType(np.bytes_),
          b'abc\x00\x00',
      ),
  )
  def test_serialize_deserialize_string_value(
      self, value, type_spec, expected_value
  ):
    value_proto, value_type = value_serialization.serialize_value(
        value, type_spec
    )
    type_test_utils.assert_types_identical(value_type, type_spec)
    result, result_type = value_serialization.deserialize_value(
        value_proto, type_spec
    )
    type_test_utils.assert_types_identical(result_type, type_spec)
    self.assertEqual(result, expected_value)

  def test_serialize_deserialize_variable_as_tensor_value(self):
    x = tf.Variable(10.0)
    type_spec = TensorType(np.float32)
    value_proto, value_type = value_serialization.serialize_value(x, type_spec)
    type_test_utils.assert_types_identical(value_type, TensorType(np.float32))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, TensorType(np.float32))
    self.assertEqual(x, y)

  def test_serialize_raises_on_incompatible_dtype_float_to_int(self):
    x = np.float32(10.0)
    with self.assertRaises(ValueError):
      value_serialization.serialize_value(x, TensorType(np.int32))

  # def test_serialize_deserialize_tensor_value_with_different_dtype(self):
  #   x = np.int32(10)
  #   value_proto, value_type = value_serialization.serialize_value(
  #       x, TensorType(np.float32)
  #   )
  #   type_test_utils.assert_types_identical(value_type, TensorType(np.float32))
  #   y, type_spec = value_serialization.deserialize_value(value_proto)
  #   type_test_utils.assert_types_identical(type_spec, TensorType(np.float32))
  #   self.assertEqual(y, 10.0)

  def test_serialize_deserialize_tensor_value_with_nontrivial_shape(self):
    x = np.int32([10, 20, 30])
    value_proto, value_type = value_serialization.serialize_value(
        x, TensorType(np.int32, [3])
    )
    type_test_utils.assert_types_identical(
        value_type, TensorType(np.int32, [3])
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, TensorType(np.int32, [3]))
    self.assertAllEqual(x, y)

  def test_serialize_struct_with_type_element_mismatch(self):
    x = {'a': 1}
    with self.assertRaisesRegex(
        TypeError,
        (
            'Cannot serialize a struct value of 1 elements to a struct '
            'type requiring 2 elements.'
        ),
    ):
      value_serialization.serialize_value(
          x, computation_types.StructType([('a', np.int32), ('b', np.int32)])
      )

  def test_serialize_sequence_bad_element_type(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize dataset .* int64\* .* float32\*.*'
    ):
      _ = value_serialization.serialize_value(
          x, computation_types.SequenceType(np.float32)
      )

  def test_serialize_sequence_bad_container_type(self):
    # A Python non-ODict container whose names are not in alphabetical order.
    bad_container = collections.namedtuple('BadContainer', 'y x')
    x = tf.data.Dataset.range(5).map(lambda x: bad_container(x, x * 2))
    with self.assertRaisesRegex(ValueError, r'non-OrderedDict container'):
      _ = value_serialization.serialize_value(
          x, computation_types.SequenceType(bad_container(np.int64, np.int64))
      )

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize Python type int as .* float32\*'
    ):
      _ = value_serialization.serialize_value(
          5, computation_types.SequenceType(np.float32)
      )

  @parameterized.named_parameters(
      ('as_dataset', lambda x: x), ('as_list', list)
  )
  def test_serialize_deserialize_sequence_of_scalars(self, dataset_fn):
    ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, computation_types.SequenceType(np.int64)
    )
    type_test_utils.assert_types_identical(
        value_type, computation_types.SequenceType(np.int64)
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.SequenceType(np.int64)
    )
    self.assertEqual(list(y), [x * 2 for x in range(5)])

  @parameterized.named_parameters(
      ('as_dataset', lambda x: x), ('as_list', list)
  )
  def test_serialize_deserialize_sequence_of_namedtuples_alphabetical_order(
      self, dataset_fn
  ):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, np.int32), c=tf.cast(x - 1, np.float32)
      )

    ds = tf.data.Dataset.range(5).map(make_test_tuple)
    ds_repr = dataset_fn(ds)
    element_type = computation_types.to_type(
        test_tuple_type(np.int64, np.int32, np.float32)
    )
    sequence_type = computation_types.SequenceType(element_type)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, sequence_type
    )
    self.assertEqual(value_type, sequence_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_equivalent(type_spec, sequence_type)
    actual_values = list(y)
    expected_values = [
        test_tuple_type(a=x * 2, b=x, c=x - 1.0) for x in range(5)
    ]
    for actual, expected in zip(actual_values, expected_values):
      self.assertAllClose(actual, expected)

  def test_serialize_deserialize_sequence_of_scalars_graph_mode(self):
    with tf.Graph().as_default():
      ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
      value_proto, value_type = value_serialization.serialize_value(
          ds, computation_types.SequenceType(np.int64)
      )
    type_test_utils.assert_types_identical(
        value_type, computation_types.SequenceType(np.int64)
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.SequenceType(np.int64)
    )
    self.assertEqual(list(y), [x * 2 for x in range(5)])

  @parameterized.named_parameters(
      ('as_dataset', lambda x: x), ('as_list', list)
  )
  def test_serialize_deserialize_sequence_of_tuples(self, dataset_fn):
    ds = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, np.int32), tf.cast(x - 1, np.float32))
    )
    ds_repr = dataset_fn(ds)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr,
        computation_types.SequenceType(
            element=(np.int64, np.int32, np.float32)
        ),
    )
    expected_type = computation_types.SequenceType(
        (np.int64, np.int32, np.float32)
    )
    type_test_utils.assert_types_identical(value_type, expected_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # Only checking for equivalence, we don't have the Python container
    # after deserialization.
    type_test_utils.assert_types_equivalent(type_spec, expected_type)
    self.assertEqual(list(y), [(x * 2, x, x - 1.0) for x in range(5)])

  @parameterized.named_parameters(
      ('as_dataset', lambda x: x), ('as_list', list)
  )
  def test_serialize_deserialize_sequence_of_nested_structures(
      self, dataset_fn
  ):
    def _make_nested_tf_structure(x):
      return collections.OrderedDict(
          b=tf.cast(x, np.int32),
          a=tuple([
              x,
              collections.OrderedDict(u=x * 2, v=x * 3),
              collections.OrderedDict(x=x**2, y=x**3),
          ]),
      )

    ds = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    ds_repr = dataset_fn(ds)
    element_type = computation_types.to_type(
        collections.OrderedDict(
            b=np.int32,
            a=tuple([
                np.int64,
                collections.OrderedDict(u=np.int64, v=np.int64),
                collections.OrderedDict(x=np.int64, y=np.int64),
            ]),
        )
    )
    sequence_type = computation_types.SequenceType(element=element_type)
    value_proto, value_type = value_serialization.serialize_value(
        ds_repr, sequence_type
    )
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
              collections.OrderedDict(x=x**2, y=x**3),
          ]),
      )

    actual_values = list(y)
    expected_values = [_build_expected_structure(x) for x in range(5)]
    for actual, expected in zip(actual_values, expected_values):
      self.assertEqual(type(actual), type(expected))
      self.assertAllClose(actual, expected)

  def test_serialize_deserialize_tensor_value_with_bad_shape(self):
    x = tf.constant([10, 20, 30])
    with self.assertRaises(TypeError):
      value_serialization.serialize_value(x, np.int32)

  def test_serialize_deserialize_computation_value(self):

    proto = computation_factory.create_lambda_identity(
        computation_types.TensorType(np.int32)
    )
    comp = computation_impl.ConcreteComputation(
        proto, context_stack_impl.context_stack
    )

    value_proto, value_type = value_serialization.serialize_value(comp)
    self.assertEqual(value_proto.WhichOneof('value'), 'computation')
    type_test_utils.assert_types_identical(
        value_type,
        computation_types.FunctionType(parameter=np.int32, result=np.int32),
    )
    _, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec,
        computation_types.FunctionType(parameter=np.int32, result=np.int32),
    )

  def test_serialize_deserialize_nested_tuple_value_with_names(self):
    x = collections.OrderedDict(
        a=10, b=[20, 30], c=collections.OrderedDict(d=40)
    )
    x_type = computation_types.to_type(
        collections.OrderedDict(
            a=np.int32,
            b=[np.int32, np.int32],
            c=collections.OrderedDict(d=np.int32),
        )
    )
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # Don't assert on the Python container since it is lost in serialization.
    type_test_utils.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container(x, recursive=True))

  def test_serialize_deserialize_nested_tuple_value_without_names(self):
    x = (10, 20)
    x_type = computation_types.to_type((np.int32, np.int32))
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container((10, 20)))

  def test_serialize_deserialize_federated_at_clients(self):
    x = [10, 20]
    x_type = computation_types.FederatedType(np.int32, placements.CLIENTS)
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(
        value_type,
        computation_types.FederatedType(np.int32, placements.CLIENTS),
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(
        type_spec, computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    self.assertEqual(y, [10, 20])

  def test_deserialize_federated_value_with_unset_member_type(self):
    x = 10
    x_type = computation_types.to_type(np.int32)
    member_proto, _ = value_serialization.serialize_value(x, x_type)
    fully_specified_type_at_clients = type_serialization.serialize_type(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=fully_specified_type_at_clients.federated.all_equal,
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type, value=[member_proto]
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    deserialized_federated_value, deserialized_type_spec = (
        value_serialization.deserialize_value(federated_value_proto)
    )
    type_test_utils.assert_types_identical(
        deserialized_type_spec,
        computation_types.FederatedType(np.int32, placements.CLIENTS),
    )
    self.assertEqual(deserialized_federated_value, [10])

  def test_deserialize_federated_value_with_incompatible_member_types_raises(
      self,
  ):
    x = 10
    x_type = computation_types.to_type(np.int32)
    int_member_proto, _ = value_serialization.serialize_value(x, x_type)
    y = 10.0
    y_type = computation_types.to_type(np.float32)
    float_member_proto, _ = value_serialization.serialize_value(y, y_type)
    fully_specified_type_at_clients = type_serialization.serialize_type(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=False,
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[int_member_proto, float_member_proto],
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    with self.assertRaises(TypeError):
      value_serialization.deserialize_value(federated_value_proto)

  def test_deserialize_federated_all_equal_value_takes_first_element(self):
    tensor_value_pb, _ = value_serialization.serialize_value(
        10, TensorType(np.int32)
    )
    num_clients = 5
    value_pb = executor_pb2.Value(
        federated=executor_pb2.Value.Federated(
            value=[tensor_value_pb] * num_clients,
            type=computation_pb2.FederatedType(
                placement=computation_pb2.PlacementSpec(
                    value=computation_pb2.Placement(uri=placements.CLIENTS.uri)
                )
            ),
        )
    )
    all_equal_clients_type_hint = computation_types.FederatedType(
        np.int32, placements.CLIENTS, all_equal=True
    )
    deserialized_value, deserialized_type = (
        value_serialization.deserialize_value(
            value_pb, all_equal_clients_type_hint
        )
    )
    type_test_utils.assert_types_identical(
        deserialized_type, all_equal_clients_type_hint
    )
    self.assertEqual(deserialized_value, 10)

  def test_deserialize_federated_value_promotes_types(self):
    x = [10]
    smaller_type = computation_types.StructType(
        [(None, computation_types.to_type(np.int32))]
    )
    smaller_type_member_proto, _ = value_serialization.serialize_value(
        x, smaller_type
    )
    larger_type = computation_types.StructType(
        [('a', computation_types.to_type(np.int32))]
    )
    larger_type_member_proto, _ = value_serialization.serialize_value(
        x, larger_type
    )
    type_at_clients = type_serialization.serialize_type(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=type_at_clients.federated.placement, all_equal=False
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[larger_type_member_proto, smaller_type_member_proto],
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    _, deserialized_type_spec = value_serialization.deserialize_value(
        federated_value_proto
    )
    type_test_utils.assert_types_identical(
        deserialized_type_spec,
        computation_types.FederatedType(larger_type, placements.CLIENTS),
    )

  def test_serialize_deserialize_federated_at_server(self):
    x = 10
    x_type = computation_types.FederatedType(np.int32, placements.SERVER)
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    type_test_utils.assert_types_identical(
        value_type, computation_types.FederatedType(np.int32, placements.SERVER)
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    type_test_utils.assert_types_identical(type_spec, x_type)
    self.assertEqual(y, 10)


class DatasetSerializationTest(absltest.TestCase):

  def test_serialize_sequence_bytes_too_large(self):
    with self.assertRaisesRegex(
        ValueError, r'Serialized size .* exceeds maximum allowed'
    ):
      _ = value_serialization._serialize_dataset(
          tf.data.Dataset.range(5), max_serialized_size_bytes=0
      )

  def test_roundtrip_sequence_of_scalars(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    serialized_bytes = value_serialization._serialize_dataset(x)
    element_type = computation_types.tensorflow_to_type(x.element_spec)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes, element_type
    )
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertEqual(list(y), [x * 2 for x in range(5)])

  def test_roundtrip_sequence_of_tuples(self):
    x = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, np.int32), tf.cast(x - 1, np.float32))
    )
    serialized_bytes = value_serialization._serialize_dataset(x)
    element_type = computation_types.tensorflow_to_type(x.element_spec)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes, element_type
    )
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertEqual(list(y), [(x * 2, x, x - 1.0) for x in range(5)])

  def test_roundtrip_sequence_of_singleton_tuples(self):
    x = tf.data.Dataset.range(5).map(lambda x: (x,))
    serialized_bytes = value_serialization._serialize_dataset(x)
    element_type = computation_types.tensorflow_to_type(x.element_spec)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes, element_type
    )
    self.assertEqual(x.element_spec, y.element_spec)
    expected_values = [(x,) for x in range(5)]
    actual_values = list(y)
    self.assertEqual(expected_values, actual_values)

  def test_roundtrip_sequence_of_namedtuples(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, np.int32), c=tf.cast(x - 1, np.float32)
      )

    x = tf.data.Dataset.range(5).map(make_test_tuple)
    serialized_bytes = value_serialization._serialize_dataset(x)
    element_type = computation_types.tensorflow_to_type(x.element_spec)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialized_bytes, element_type
    )
    self.assertEqual(x.element_spec, y.element_spec)
    self.assertEqual(
        list(y), [test_tuple_type(a=x * 2, b=x, c=x - 1.0) for x in range(5)]
    )

  def test_roundtrip_sequence_of_nested_structures(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return collections.OrderedDict(
          b=tf.cast(x, np.int32),
          a=(
              x,
              test_tuple_type(x * 2, x * 3),
              collections.OrderedDict(x=x**2, y=x**3),
          ),
      )

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    serialzied_bytes = value_serialization._serialize_dataset(x)
    element_type = computation_types.tensorflow_to_type(x.element_spec)
    y = value_serialization._deserialize_dataset_from_graph_def(
        serialzied_bytes, element_type
    )
    # Note: TF loses the `OrderedDict` during serialization, so the expectation
    # here is for a `dict` in the result.
    expected_element_spec = collections.OrderedDict(
        b=tf.TensorSpec([], np.int32),
        a=(
            tf.TensorSpec([], np.int64),
            test_tuple_type(
                tf.TensorSpec([], np.int64), tf.TensorSpec([], np.int64)
            ),
            collections.OrderedDict(
                x=tf.TensorSpec([], np.int64), y=tf.TensorSpec([], np.int64)
            ),
        ),
    )
    self.assertEqual(y.element_spec, expected_element_spec)

    def _build_expected_structure(x):
      return collections.OrderedDict(
          b=x,
          a=(
              x,
              test_tuple_type(x * 2, x * 3),
              collections.OrderedDict(x=x**2, y=x**3),
          ),
      )

    expected_values = (_build_expected_structure(x) for x in range(5))
    for actual, expected in zip(y, expected_values):
      self.assertAlmostEqual(actual, expected)


class SerializeCardinalitiesTest(absltest.TestCase):

  def test_serialize_deserialize_clients_and_server_cardinalities_roundtrip(
      self,
  ):
    client_and_server_cardinalities = {
        placements.CLIENTS: 10,
        placements.SERVER: 1,
    }
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_and_server_cardinalities
    )
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list
    )
    self.assertEqual(
        client_and_server_cardinalities, reconstructed_cardinalities
    )

  def test_serialize_deserialize_clients_alone(self):
    client_cardinalities = {placements.CLIENTS: 10}
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_cardinalities
    )
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list
    )
    self.assertEqual(client_cardinalities, reconstructed_cardinalities)


if __name__ == '__main__':
  absltest.main()
