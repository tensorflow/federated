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
import portpicker
import tensorflow as tf
import tree

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import tensorflow_executor_bindings
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils


# Creating logical devices should be done only once before TF runtime startup
# Thus, perform it during setUpModule method.
def setUpModule():
  devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(
      devices[0],
      [
          tf.config.LogicalDeviceConfiguration(),
      ]
      * 8,
  )


def _to_python_value(value):
  def _fn(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return None

  return tree.traverse(_fn, value)


def get_executor(use_tf_executor):
  if use_tf_executor:
    return tensorflow_executor_bindings.create_tensorflow_executor()
  else:
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:%d' % i for i in range(8)], mesh_dims=[('batch', 8)]
    )
    return tensorflow_executor_bindings.create_dtensor_executor(
        tf.experimental.dtensor.device_name(), mesh.to_string(), -1
    )


class ReferenceResolvingExecutorBindingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create(self, use_tf_executor):
    try:
      executor_bindings.create_reference_resolving_executor(
          get_executor(use_tf_executor)
      )
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Exception: {e}')

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_value(self, use_tf_executor):
    executor = executor_bindings.create_reference_resolving_executor(
        get_executor(use_tf_executor)
    )
    # 1. Test a simple tensor.
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        [1, 2, 3], expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    self.assertEqual(value.ref, 0)
    self.assertEqual(str(value), '0')
    self.assertEqual(repr(value), r'<OwnedValueId: 0>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    type_test_utils.assert_types_identical(type_spec, expected_type_spec)
    self.assertEqual(list(deserialized_value), [1, 2, 3])
    # 2. Test a struct of tensors, ensure that we get a different ID.
    expected_type_spec = computation_types.StructType([
        ('a', computation_types.TensorType(np.int64, [3])),
        ('b', computation_types.TensorType(np.float32, [])),
    ])
    value_pb, _ = value_serialization.serialize_value(
        collections.OrderedDict(a=np.array([1, 2, 3]), b=np.array(42.0)),
        expected_type_spec,
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented.
    self.assertEqual(value.ref, 1)
    self.assertEqual(str(value), '1')
    self.assertEqual(repr(value), r'<OwnedValueId: 1>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    # Note: here we've lost the names `a` and `b` in the output. The output
    # is a more _strict_ type.
    self.assertTrue(expected_type_spec.is_assignable_from(type_spec))
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, expected_type_spec
    )

    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(
        deserialized_value, collections.OrderedDict(a=[1, 2, 3], b=42.0)
    )

    # 3. Test creating a value from a computation.
    foo = computation_factory.create_lambda_identity(
        computation_types.TensorType(np.int64)
    )

    value_pb = executor_pb2.Value(computation=foo)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented again.
    self.assertEqual(value.ref, 2)
    self.assertEqual(str(value), '2')
    self.assertEqual(repr(value), '<OwnedValueId: 2>')
    # Note: functions are not materializable, no addition assertions.

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_struct(self, use_tf_executor):
    executor = executor_bindings.create_reference_resolving_executor(
        get_executor(use_tf_executor)
    )
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        np.array([1, 2, 3]), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, [[1, 2, 3], [1, 2, 3]])
    # 2. Create a struct from the struct and another value.
    new_struct_value = executor.create_struct([struct_value.ref, value.ref])
    materialized_value = executor.materialize(new_struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [struct_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, [[[1, 2, 3], [1, 2, 3]], [1, 2, 3]])

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_selection(self, use_tf_executor):
    executor = executor_bindings.create_reference_resolving_executor(
        get_executor(use_tf_executor)
    )
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        np.array([1, 2, 3]), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, [[1, 2, 3], [1, 2, 3]])
    # 2. Select the first value out of the struct.
    new_value = executor.create_selection(struct_value.ref, 0)
    materialized_value = executor.materialize(new_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    type_test_utils.assert_types_equivalent(type_spec, expected_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, [1, 2, 3])

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_call_with_arg(self, use_tf_executor):
    executor = executor_bindings.create_reference_resolving_executor(
        get_executor(use_tf_executor)
    )

    foo = computation_factory.create_lambda_identity(
        computation_types.TensorType(np.int64)
    )
    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    value_pb, _ = value_serialization.serialize_value(
        np.array([1, 2, 3]),
        computation_types.TensorType(np.int64, [3]),
    )
    arg = executor.create_value(value_pb)
    result = executor.create_call(comp.ref, arg.ref)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    result_tensor = _to_python_value(result_tensor)
    self.assertEqual(result_tensor, [1, 2, 3])

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_call_no_arg(self, use_tf_executor):
    executor = executor_bindings.create_reference_resolving_executor(
        get_executor(use_tf_executor)
    )

    foo = computation_factory.create_lambda_empty_struct()
    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, None)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertEmpty(result_tensor, ())


class FederatingExecutorBindingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('server_client_both_tf_executor', True, True),
      # ('server_client_both_dtensor_executor', False, False),
      # ('server_tf_client_dtensor_executor', True, False),
      # ('server_dtensor_client_tf_executor', False, True),
  )
  def test_construction_placements_casters(
      self, use_tf_executor_for_server, use_tf_executor_for_client
  ):
    server_executor = get_executor(use_tf_executor_for_server)
    client_executor = get_executor(use_tf_executor_for_client)
    with self.subTest('placement_literal_keys'):
      try:
        executor_bindings.create_federating_executor(
            server_executor,
            client_executor,
            {placements.CLIENTS: 10},
        )
      except Exception as e:  # pylint: disable=broad-except
        self.fail(f'Exception: {e}')
    with self.subTest('fails_non_dict'):
      with self.assertRaisesRegex(TypeError, 'must be a `Mapping`'):
        executor_bindings.create_federating_executor(
            server_executor,
            client_executor,
            [(placements.CLIENTS, 10)],
        )
    with self.subTest('fails_non_placement_keys'):
      with self.assertRaisesRegex(TypeError, '`PlacementLiteral`'):
        executor_bindings.create_federating_executor(
            server_executor, client_executor, {'clients': 10}
        )
      with self.assertRaisesRegex(TypeError, '`PlacementLiteral`'):
        executor_bindings.create_federating_executor(
            server_executor, client_executor, {10: 10}
        )
    with self.subTest('fails_non_int_value'):
      with self.assertRaisesRegex(TypeError, r'`int` values'):
        executor_bindings.create_federating_executor(
            server_executor,
            client_executor,
            {placements.CLIENTS: 0.5},
        )


class RemoteExecutorBindingsTest(absltest.TestCase):

  def test_insecure_channel_construction(self):
    remote_ex = executor_bindings.create_remote_executor(
        executor_bindings.create_insecure_grpc_channel(
            'localhost:{}'.format(portpicker.pick_unused_port())
        ),
        cardinalities={placements.CLIENTS: 10},
    )
    self.assertIsInstance(remote_ex, executor_bindings.Executor)


class ComposingExecutorBindingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('server_client_both_tf_executor', True, True),
      # ('server_client_both_dtensor_executor', False, False),
      # ('server_tf_client_dtensor_executor', True, False),
      # ('server_dtensor_client_tf_executor', False, True),
  )
  def test_construction(
      self, use_tf_executor_for_server, use_tf_executor_for_client
  ):
    server = get_executor(use_tf_executor_for_server)
    children = [
        executor_bindings.create_composing_child(
            get_executor(use_tf_executor_for_client),
            {placements.CLIENTS: 0},
        )
    ]
    composing_ex = executor_bindings.create_composing_executor(server, children)
    self.assertIsInstance(composing_ex, executor_bindings.Executor)


class SerializeTensorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_int32', 1, np.int32),
      ('scalar_float64', 2.0, np.float64),
      ('scalar_string', b'abc', np.str_),
      ('tensor_int32', [1, 2, 3], np.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], np.float64),
      ('tensor_string', [[b'abc', b'xyz']], np.str_),
  )
  def test_serialize(self, input_value, dtype):
    serialized_value = executor_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype)
    )
    tensor_proto = tf.make_tensor_proto(values=0)
    self.assertTrue(serialized_value.tensor.Unpack(tensor_proto))
    actual_value = tf.make_ndarray(tensor_proto)
    actual_value = _to_python_value(actual_value)
    self.assertEqual(actual_value, input_value)

  @parameterized.named_parameters(
      ('scalar_int32', 1, np.int32),
      ('scalar_float64', 2.0, np.float64),
      ('scalar_string', b'abc', np.str_),
      ('tensor_int32', [1, 2, 3], np.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], np.float64),
      ('tensor_string', [[b'abc', b'xyz']], np.str_),
  )
  def test_roundtrip(self, input_value, dtype):
    serialized_value = executor_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype)
    )
    deserialized_value = executor_bindings.deserialize_tensor_value(
        serialized_value
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, input_value)


class SequenceExecutorBindingsTest(absltest.TestCase):

  def test_create(self):
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    try:
      executor_bindings.create_sequence_executor(executor)
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')

  def test_materialize_on_unkown_fails(self):
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    executor_bindings.create_sequence_executor(executor)
    with self.assertRaisesRegex(Exception, 'NOT_FOUND'):
      executor.materialize(0)


if __name__ == '__main__':
  absltest.main()
