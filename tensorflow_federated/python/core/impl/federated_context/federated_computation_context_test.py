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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class FederatedComputationContextTest(absltest.TestCase):

  def test_invoke_returns_value_with_correct_type(self):
    tensor_type = computation_types.TensorType(np.int32)
    computation_proto = computation_factory.create_lambda_identity(tensor_type)
    computation = computation_impl.ConcreteComputation(
        computation_proto, context_stack_impl.context_stack
    )
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )

    with context_stack_impl.context_stack.install(context):
      result = context.invoke(computation, 1)

    self.assertIsInstance(result, value_impl.Value)
    self.assertEqual(result.type_signature, tensor_type)

  def test_ingest_zips_value_when_necessary_to_match_federated_type(self):
    # Expects `{<int, int>}@C`
    @federated_computation.federated_computation(
        computation_types.FederatedType(
            (np.int32, np.int32), placements.CLIENTS
        )
    )
    def fn(_):
      return ()

    # This thing will be <{int}@C, {int}@C>
    arg = building_blocks.Struct([
        building_blocks.Reference(
            'x',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
        building_blocks.Reference(
            'y',
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
    ])

    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    with context_stack_impl.context_stack.install(context):
      fn(arg)

  def test_ingest_zips_federated_under_struct(self):

    @federated_computation.federated_computation(
        computation_types.StructType([(
            None,
            collections.OrderedDict(
                x=computation_types.FederatedType(np.int32, placements.CLIENTS),
                y=computation_types.FederatedType(np.int32, placements.CLIENTS),
            ),
        )])
    )
    def fn(_):
      return ()

    arg = building_blocks.Struct(
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

    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    with context_stack_impl.context_stack.install(context):
      fn(arg)

  def test_construction_populates_name(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    self.assertEqual(context.name, 'FEDERATED')

  def test_suggested_name_populates_name_attribute(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO'
    )
    self.assertEqual(context.name, 'FOO')

  def test_child_name_doesnt_conflict(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO'
    )
    self.assertEqual(context.name, 'FOO')
    context2 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO', parent=context
    )
    self.assertEqual(context2.name, 'FOO_1')
    context3 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO', parent=context2
    )
    self.assertEqual(context3.name, 'FOO_2')

  def test_parent_populated_correctly(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    context2 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, parent=context
    )
    self.assertIs(context2.parent, context)

  def test_bind_single_computation_to_reference(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    ref = context.bind_computation_to_reference(data)
    symbol_bindings = context.symbol_bindings
    bound_symbol_name = symbol_bindings[0][0]

    self.assertIsInstance(ref, building_blocks.Reference)
    self.assertEqual(ref.type_signature, data.type_signature)
    self.assertLen(symbol_bindings, 1)
    self.assertEqual(bound_symbol_name, ref.name)

  def test_bind_two_computations_to_reference(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    lit = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    float_data = building_blocks.Literal(
        2.0, computation_types.TensorType(np.float32)
    )
    ref1 = context.bind_computation_to_reference(lit)
    ref2 = context.bind_computation_to_reference(float_data)
    symbol_bindings = context.symbol_bindings

    self.assertIsInstance(ref1, building_blocks.Reference)
    self.assertIsInstance(ref2, building_blocks.Reference)

    self.assertEqual(ref1.type_signature, lit.type_signature)
    self.assertEqual(ref2.type_signature, float_data.type_signature)
    self.assertLen(symbol_bindings, 2)
    self.assertEqual(symbol_bindings[0][0], ref1.name)
    self.assertEqual(symbol_bindings[1][0], ref2.name)


if __name__ == '__main__':
  absltest.main()
