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

import asyncio
import collections

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import federated_composing_strategy
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization


def _create_bottom_stack():
  executor = eager_tf_executor.EagerTFExecutor()
  return reference_resolving_executor.ReferenceResolvingExecutor(executor)


def _create_worker_stack():
  factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
      placements.SERVER: _create_bottom_stack(),
      placements.CLIENTS: [_create_bottom_stack() for _ in range(2)],
  })
  return federating_executor.FederatingExecutor(factory, _create_bottom_stack())


def _create_middle_stack(children):
  factory = federated_composing_strategy.FederatedComposingStrategy.factory(
      _create_bottom_stack(), children)
  executor = federating_executor.FederatingExecutor(factory,
                                                    _create_bottom_stack())
  return reference_resolving_executor.ReferenceResolvingExecutor(executor)


def _create_test_executor():
  executor = _create_middle_stack([
      _create_middle_stack([_create_worker_stack() for _ in range(3)]),
      _create_middle_stack([_create_worker_stack() for _ in range(3)]),
  ])
  # 2 clients per worker stack * 3 worker stacks * 2 middle stacks
  num_clients = 12
  return executor, num_clients


def _invoke(ex, comp, arg=None):
  v1 = asyncio.run(ex.create_value(comp))
  if arg is not None:
    type_spec = v1.type_signature.parameter
    v2 = asyncio.run(ex.create_value(arg, type_spec))
  else:
    v2 = None
  v3 = asyncio.run(ex.create_call(v1, v2))
  return asyncio.run(v3.compute())


class FederatedComposingStrategyTest(parameterized.TestCase):

  def test_recovers_from_raising(self):

    class _RaisingExecutor(eager_tf_executor.EagerTFExecutor):
      """An executor which can be configured to raise on `create_value`."""

      def __init__(self):
        self._should_raise = True
        super().__init__()

      def stop_raising(self):
        self._should_raise = False

      async def create_value(self, *args, **kwargs):
        if self._should_raise:
          raise AssertionError
        return await super().create_value(*args, **kwargs)

    raising_executors = [_RaisingExecutor() for _ in range(2)]

    factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
        placements.SERVER: _create_worker_stack(),
        placements.CLIENTS: raising_executors,
    })
    federating_ex = federating_executor.FederatingExecutor(
        factory, _create_worker_stack())

    raising_stacks = [federating_ex for _ in range(3)]

    executor = _create_middle_stack([
        _create_middle_stack(raising_stacks),
        _create_middle_stack(raising_stacks),
    ])

    @computations.federated_computation(
        computation_types.at_clients(tf.float32))
    def comp(x):
      return intrinsics.federated_mean(x)

    # 2 clients per worker stack * 3 worker stacks * 2 middle stacks
    num_clients = 12
    arg = [float(x + 1) for x in range(num_clients)]

    with self.assertRaises(AssertionError):
      _invoke(executor, comp, arg)

    for ex in raising_executors:
      ex.stop_raising()

    result = _invoke(executor, comp, arg)
    self.assertEqual(result, 6.5)

  def test_federated_value_at_server(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.SERVER)

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 10)

  def test_federated_value_at_clients(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.CLIENTS)

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 10)

  def test_federated_eval_at_server(self):

    @computations.federated_computation
    def comp():
      return_five = computations.tf_computation(lambda: 5)
      return intrinsics.federated_eval(return_five, placements.SERVER)

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 5)

  def test_federated_eval_at_server_then_map(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return_five = computations.tf_computation(lambda: 5)
      five_at_server = intrinsics.federated_eval(return_five, placements.SERVER)
      six_at_server = intrinsics.federated_map(add_one, five_at_server)
      return six_at_server

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 6)

  def test_federated_eval_at_clients(self):

    @computations.federated_computation
    def comp():
      return_five = computations.tf_computation(lambda: 5)
      return intrinsics.federated_eval(return_five, placements.CLIENTS)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertIsInstance(result, list)
    self.assertLen(result, num_clients)
    for x in result:
      self.assertEqual(x, 5)

  def test_federated_aggregate(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def add_int(x, y):
      return x + y

    @computations.tf_computation(tf.int32)
    def add_five(x):
      return x + 5

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_aggregate(value, 0, add_int, add_int,
                                            add_five)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 10 * num_clients + 5)

  def test_federated_aggregate_of_nested_tuple(self):
    test_type = computation_types.StructType([
        ('a', (tf.int32, tf.float32)),
    ])

    @computations.tf_computation(test_type, test_type)
    def add_test_type(x, y):
      return collections.OrderedDict([
          ('a', (x.a[0] + y.a[0], x.a[1] + y.a[1])),
      ])

    @computations.tf_computation(test_type)
    def add_five_and_three(x):
      return collections.OrderedDict([('a', (x['a'][0] + 5, x['a'][1] + 3.0))])

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(
          collections.OrderedDict([('a', (10, 2.0))]), placements.CLIENTS)
      zero = collections.OrderedDict([('a', (0, 0.0))])
      return intrinsics.federated_aggregate(value, zero, add_test_type,
                                            add_test_type, add_five_and_three)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    expected_result = structure.Struct([
        ('a',
         structure.Struct([
             (None, 10 * num_clients + 5),
             (None, 2.0 * num_clients + 3.0),
         ])),
    ])
    self.assertEqual(result, expected_result)

  def test_federated_broadcast(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      value_at_server = intrinsics.federated_value(10, placements.SERVER)
      value_at_clients = intrinsics.federated_broadcast(value_at_server)
      return intrinsics.federated_map(add_one, value_at_clients)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, [10 + 1] * num_clients)

  def test_federated_map_at_server(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(10, placements.SERVER)
      return intrinsics.federated_map(add_one, value)

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 10 + 1)

  def test_federated_map(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_map(add_one, value)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, [10 + 1] * num_clients)

  def test_federated_map_all_equal(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_map_all_equal(add_one, value)

    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    for value in result:
      self.assertEqual(value.numpy(), 10 + 1)

  def test_federated_select(self):

    @computations.tf_computation
    def get_keys():
      return tf.constant([1, 2, 5])

    @computations.tf_computation(tf.string, tf.int32)
    def select_fn(database, key):
      return collections.OrderedDict(database=database, key=key)

    @computations.federated_computation
    def comp():
      client_keys = intrinsics.federated_eval(get_keys, placements.CLIENTS)
      max_key = intrinsics.federated_value(5, placements.SERVER)
      server_val = intrinsics.federated_value('db', placements.SERVER)
      return intrinsics.federated_select(client_keys, max_key, server_val,
                                         select_fn)

    executor, num_clients = _create_test_executor()
    results = _invoke(executor, comp)
    self.assertIsInstance(results, list)
    self.assertLen(results, num_clients)
    for client_result in results:
      self.assertIsInstance(client_result, tf.data.Dataset)
      for actual, expected_key in zip(client_result, [1, 2, 5]):
        expected = collections.OrderedDict(database='db', key=expected_key)
        self.assertEqual(actual, expected)

  def test_federated_zip_at_server_unnamed(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip([
          intrinsics.federated_value(10, placements.SERVER),
          intrinsics.federated_value(20, placements.SERVER),
      ])

    self.assertEqual(comp.type_signature.compact_representation(),
                     '( -> <int32,int32>@SERVER)')
    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    expected_result = structure.Struct([(None, 10), (None, 20)])
    self.assertEqual(result, expected_result)

  def test_federated_zip_at_server_named(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip(
          collections.OrderedDict([
              ('A', intrinsics.federated_value(10, placements.SERVER)),
              ('B', intrinsics.federated_value(20, placements.SERVER)),
          ]))

    self.assertEqual(comp.type_signature.compact_representation(),
                     '( -> <A=int32,B=int32>@SERVER)')
    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    expected_result = structure.Struct([('A', 10), ('B', 20)])
    self.assertEqual(result, expected_result)

  def test_federated_zip_at_clients_unnamed(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip([
          intrinsics.federated_value(10, placements.CLIENTS),
          intrinsics.federated_value(20, placements.CLIENTS),
      ])

    self.assertEqual(comp.type_signature.compact_representation(),
                     '( -> {<int32,int32>}@CLIENTS)')
    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    for value in result:
      excepted_value = structure.Struct([(None, 10), (None, 20)])
      self.assertEqual(value, excepted_value)

  def test_federated_zip_at_clients_named(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip(
          collections.OrderedDict([
              ('A', intrinsics.federated_value(10, placements.CLIENTS)),
              ('B', intrinsics.federated_value(20, placements.CLIENTS)),
          ]))

    self.assertEqual(comp.type_signature.compact_representation(),
                     '( -> {<A=int32,B=int32>}@CLIENTS)')
    executor, _ = _create_test_executor()
    result = _invoke(executor, comp)
    for value in result:
      excepted_value = structure.Struct([('A', 10), ('B', 20)])
      self.assertEqual(value, excepted_value)

  @parameterized.named_parameters([
      ('at_clients', placements.CLIENTS),
      ('at_server', placements.SERVER),
  ])
  def test_federated_zip_nested(self, placement):

    @computations.federated_computation()
    def comp():
      server_val = intrinsics.federated_value(10, placement)
      return intrinsics.federated_zip((server_val, (server_val, server_val)))

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    if placement.is_clients():
      self.assertIsInstance(result, list)
      self.assertLen(result, num_clients)
      result = result[0]
    expected_result = structure.from_container((10, (10, 10)), recursive=True)
    self.assertEqual(result, expected_result)

  def test_federated_sum(self):

    @computations.federated_computation
    def comp():
      value = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_sum(value)

    executor, num_clients = _create_test_executor()
    result = _invoke(executor, comp)
    self.assertEqual(result, 10 * num_clients)

  def test_federated_mean(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32))
    def comp(x):
      return intrinsics.federated_mean(x)

    executor, num_clients = _create_test_executor()
    arg = [float(x + 1) for x in range(num_clients)]
    result = _invoke(executor, comp, arg)
    self.assertEqual(result, 6.5)

  def test_federated_weighted_mean(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32),
        computation_types.at_clients(tf.float32))
    def comp(x, y):
      return intrinsics.federated_mean(x, y)

    executor, num_clients = _create_test_executor()
    arg = structure.Struct([('x', [float(x + 1) for x in range(num_clients)]),
                            ('y', [1.0, 2.0, 3.0] * 4)])
    result = _invoke(executor, comp, arg)
    self.assertAlmostEqual(result, 6.83333333333, places=3)

  def test_executor_call_unsupported_intrinsic(self):
    # `whimsy_intrinsic` definition is needed to allow successful lookup.
    whimsy_intrinsic = intrinsic_defs.IntrinsicDef(
        'WHIMSY_INTRINSIC', 'whimsy_intrinsic',
        computation_types.AbstractType('T'))
    type_signature = computation_types.TensorType(tf.int32)
    comp = pb.Computation(
        type=type_serialization.serialize_type(type_signature),
        intrinsic=pb.Intrinsic(uri='whimsy_intrinsic'))
    del whimsy_intrinsic

    factory = federated_composing_strategy.FederatedComposingStrategy.factory(
        _create_bottom_stack(), [_create_worker_stack()])
    executor = federating_executor.FederatingExecutor(factory,
                                                      _create_bottom_stack())

    v1 = asyncio.run(executor.create_value(comp))
    with self.assertRaises(NotImplementedError):
      asyncio.run(executor.create_call(v1))


if __name__ == '__main__':
  absltest.main()
