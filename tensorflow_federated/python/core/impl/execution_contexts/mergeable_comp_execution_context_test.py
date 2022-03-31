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
"""Tests for mergeable_comp_execution_context."""
import collections

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions


def build_sum_client_arg_computation(
    server_arg_type: computation_types.FederatedType,
    clients_arg_type: computation_types.FederatedType
) -> computation_base.Computation:

  @computations.federated_computation(server_arg_type, clients_arg_type)
  def up_to_merge(server_arg, client_arg):
    del server_arg  # Unused
    return intrinsics.federated_sum(client_arg)

  return up_to_merge


def build_noarg_count_clients_computation() -> computation_base.Computation:

  @computations.federated_computation()
  def up_to_merge():
    return intrinsics.federated_sum(
        intrinsics.federated_value(1, placements.CLIENTS))

  return up_to_merge


def build_whimsy_merge_computation(
    arg_type: computation_types.Type) -> computation_base.Computation:

  @computations.federated_computation(arg_type, arg_type)
  def merge(arg0, arg1):
    del arg1  # Unused
    return arg0

  return merge


def build_sum_merge_computation(
    arg_type: computation_types.Type) -> computation_base.Computation:

  @computations.tf_computation(arg_type, arg_type)
  def merge(arg0, arg1):
    return arg0 + arg1

  return merge


def build_whimsy_after_merge_computation(
    original_arg_type: computation_types.Type,
    merge_result_type: computation_types.Type) -> computation_base.Computation:

  if original_arg_type is not None:

    @computations.federated_computation(
        original_arg_type, computation_types.at_server(merge_result_type))
    def after_merge(original_arg, merge_result):
      del merge_result  # Unused
      return original_arg

  else:

    @computations.federated_computation(
        computation_types.at_server(merge_result_type))
    def after_merge(merge_result):
      return merge_result

  return after_merge


def build_return_merge_result_computation(
    original_arg_type: computation_types.Type,
    merge_result_type: computation_types.Type) -> computation_base.Computation:

  @computations.federated_computation(
      original_arg_type, computation_types.at_server(merge_result_type))
  def after_merge(original_arg, merge_result):
    del original_arg  # Unused
    return merge_result

  return after_merge


def build_return_merge_result_with_no_first_arg_computation(
    merge_result_type: computation_types.Type) -> computation_base.Computation:

  @computations.federated_computation(
      computation_types.at_server(merge_result_type))
  def after_merge(merge_result):
    return merge_result

  return after_merge


def build_sum_merge_with_first_arg_computation(
    original_arg_type: computation_types.Type,
    merge_result_type: computation_types.Type) -> computation_base.Computation:
  """Assumes original_arg_type is federated, and compatible with summing with merge_result_type."""

  @computations.tf_computation(original_arg_type['server_arg'].member,
                               merge_result_type)
  def add(x, y):
    return x + y

  @computations.federated_computation(
      original_arg_type, computation_types.at_server(merge_result_type))
  def after_merge(original_arg, merge_result):
    return intrinsics.federated_map(add,
                                    (original_arg['server_arg'], merge_result))

  return after_merge


class MergeableCompFormTest(absltest.TestCase):

  def test_raises_mismatched_up_to_merge_and_merge(self):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    bad_merge = build_whimsy_merge_computation(tf.float32)

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            bad_merge.type_signature.result))
    def after_merge(x, y):
      return (x, y)

    with self.assertRaises(
        mergeable_comp_execution_context.MergeTypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=bad_merge, after_merge=after_merge)

  def test_raises_merge_computation_not_assignable_result(self):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    @computations.tf_computation(tf.int32, tf.int32)
    def bad_merge(x, y):
      del x, y  # Unused
      return 1.  # of type float.

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            bad_merge.type_signature.result))
    def after_merge(x, y):
      return (x, y)

    with self.assertRaises(
        mergeable_comp_execution_context.MergeTypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=bad_merge, after_merge=after_merge)

  def test_raises_no_top_level_argument_in_after_agg(self):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    merge = build_whimsy_merge_computation(tf.int32)

    @computations.federated_computation(
        computation_types.at_server(merge.type_signature.result))
    def bad_after_merge(x):
      return x

    with self.assertRaises(computation_types.TypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=merge, after_merge=bad_after_merge)

  def test_raises_up_to_merge_returns_non_server_placed_result(self):

    @computations.federated_computation(computation_types.at_server(tf.int32))
    def bad_up_to_merge(x):
      # Returns non SERVER-placed result.
      return x, x

    merge = build_whimsy_merge_computation(tf.int32)

    after_merge = build_whimsy_after_merge_computation(
        bad_up_to_merge.type_signature.parameter, merge.type_signature.result)

    with self.assertRaises(mergeable_comp_execution_context.UpToMergeTypeError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=bad_up_to_merge, merge=merge, after_merge=after_merge)

  def test_raises_with_aggregation_in_after_agg(self):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    merge = build_whimsy_merge_computation(tf.int32)

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            merge.type_signature.result))
    def after_merge_with_sum(original_arg, merged_arg):
      del merged_arg  # Unused
      return intrinsics.federated_sum(original_arg['client_arg'])

    with self.assertRaisesRegex(
        mergeable_comp_execution_context.AfterMergeStructureError,
        'federated_sum'):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge,
          merge=merge,
          after_merge=after_merge_with_sum)

  def test_passes_with_correct_signatures(self):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    merge = build_whimsy_merge_computation(tf.int32)
    after_merge = build_whimsy_after_merge_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)
    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)

    self.assertIsInstance(mergeable_comp_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_passes_with_noarg_top_level_computation(self):
    up_to_merge = build_noarg_count_clients_computation()
    merge = build_whimsy_merge_computation(tf.int32)
    after_merge = build_whimsy_after_merge_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)
    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)
    self.assertIsInstance(mergeable_comp_form,
                          mergeable_comp_execution_context.MergeableCompForm)


class PartitionValueTest(absltest.TestCase):

  def test_partitions_value_with_no_clients_arguments(self):
    value = 0
    type_signature = computation_types.at_server(tf.int32)
    num_desired_subrounds = 2
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    self.assertEqual(partitioned_value, [0, 0])

  def test_wraps_value_with_empty_client_argument(self):
    value = (0, [])
    type_signature = computation_types.StructType([
        (None, computation_types.at_server(tf.int32)),
        (None, computation_types.at_clients(tf.int32))
    ])
    num_desired_subrounds = 2
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    self.assertEqual(partitioned_value, [(0, [])])

  def test_replicates_all_equal_clients_argument(self):
    value = (0, 1)
    type_signature = computation_types.StructType([
        (None, computation_types.at_server(tf.int32)),
        (None, computation_types.at_clients(tf.int32, all_equal=True))
    ])
    num_desired_subrounds = 2
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    self.assertEqual(partitioned_value, [(0, 1), (0, 1)])

  def test_partitions_client_placed_value_into_subrounds(self):
    value = list(range(10))
    type_signature = computation_types.at_clients(tf.int32)
    num_desired_subrounds = 5
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    expected_partitioning = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    self.assertEqual(partitioned_value, expected_partitioning)

  def test_partitions_clients_placed_struct_elem_into_subrounds(self):
    value = (0, list(range(10)))
    server_placed_name = 'a'
    clients_placed_name = 'b'
    type_signature = computation_types.StructType([
        (server_placed_name, computation_types.at_server(tf.int32)),
        (clients_placed_name, computation_types.at_clients(tf.int32))
    ])

    num_desired_subrounds = 5
    expected_partitioning = []
    for j in range(0, 10, 2):
      expected_struct_partition = structure.Struct([(server_placed_name, 0),
                                                    (clients_placed_name,
                                                     [j, j + 1])])
      expected_partitioning.append(expected_struct_partition)
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    self.assertEqual(partitioned_value, expected_partitioning)

  def test_partitions_fewer_clients_than_rounds_into_nonempty_rounds(self):
    value = [0, 1]
    type_signature = computation_types.at_clients(tf.int32)
    num_desired_subrounds = 5
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    expected_partitioning = [[0], [1]]
    self.assertEqual(partitioned_value, expected_partitioning)


class RepackageResultsTest(absltest.TestCase):

  def assertRoundTripEqual(self, value, type_signature,
                           expected_round_trip_value):
    num_desired_subrounds = 2
    partitioned_value = mergeable_comp_execution_context._split_value_into_subrounds(
        value, type_signature, num_desired_subrounds)
    self.assertEqual(
        mergeable_comp_execution_context._repackage_partitioned_values(
            partitioned_value, type_signature), expected_round_trip_value)

  def test_roundtrip_with_no_clients_argument(self):
    value = 0
    type_signature = computation_types.at_server(tf.int32)
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_named_struct(self):
    value = collections.OrderedDict(a=0)
    type_signature = computation_types.StructType([
        ('a', computation_types.at_server(tf.int32))
    ])
    self.assertRoundTripEqual(value, type_signature,
                              structure.Struct([('a', 0)]))

  def test_roundtrip_with_empty_clients_argument(self):
    value = (0, [])
    type_signature = computation_types.StructType([
        (None, computation_types.at_server(tf.int32)),
        (None, computation_types.at_clients(tf.int32))
    ])
    self.assertRoundTripEqual(value, type_signature,
                              structure.from_container(value))

  def test_roundtrip_with_nonempty_clients_argument(self):
    value = list(range(10))
    type_signature = computation_types.at_clients(tf.int32)
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_nonempty_tuple_clients_argument(self):
    value = tuple(range(10))
    type_signature = computation_types.at_clients(tf.int32)
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_all_equal_clients_argument(self):
    value = (0, 1)
    type_signature = computation_types.StructType([
        (None, computation_types.at_server(tf.int32)),
        (None, computation_types.at_clients(tf.int32, all_equal=True))
    ])
    self.assertRoundTripEqual(value, type_signature,
                              structure.from_container(value))


class MergeableCompExecutionContextTest(parameterized.TestCase):

  def test_invoke_raises_computation_no_compiler(self):

    @computations.tf_computation()
    def return_one():
      return 1

    ex_factories = [executor_stacks.local_executor_factory() for _ in range(1)]
    no_compiler_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories)

    with self.assertRaises(ValueError):
      no_compiler_context.invoke(return_one)

  def test_invoke_raises_computation_not_compiled_to_mergeable_comp_form(self):

    @computations.tf_computation()
    def return_one():
      return 1

    ex_factories = [executor_stacks.local_executor_factory() for _ in range(1)]
    context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories, compiler_fn=lambda x: x)

    with self.assertRaises(ValueError):
      context.invoke(return_one)

  @parameterized.named_parameters(('100_clients', (0, list(range(100)))),
                                  ('fewer_clients_than_subrounds', (0, [0])))
  def test_runs_computation_with_clients_placed_return_values(self, arg):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    merge = build_whimsy_merge_computation(tf.int32)
    after_merge = build_whimsy_after_merge_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)

    # Simply returns the original argument
    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)
    ex_factories = [executor_stacks.local_executor_factory() for _ in range(5)]
    mergeable_comp_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories)
    # We preemptively package as a struct to work around shortcircuiting in
    # type_to_py_container in a non-Struct argument case.
    arg = structure.Struct.unnamed(*arg)
    expected_result = type_conversions.type_to_py_container(
        arg, after_merge.type_signature.result)
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      ('500_clients', (0, list(range(500))), sum(range(500))),
      ('500_clients_nonzero_server_values',
       (1, list(range(500))), sum(range(500))),
      ('fewer_clients_than_subrounds', (0, [1]), 1))
  def test_computes_sum_of_clients_values(self, arg, expected_sum):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_return_merge_result_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)

    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)
    ex_factories = [executor_stacks.local_executor_factory() for _ in range(5)]
    mergeable_comp_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories)

    expected_result = type_conversions.type_to_py_container(
        expected_sum, after_merge.type_signature.result)
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      ('100_clients', (100, list(range(100))), sum(range(101))),
      ('fewer_clients_than_subrounds', (1, [1]), 2))
  def test_computes_sum_of_all_values(self, arg, expected_sum):
    up_to_merge = build_sum_client_arg_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_sum_merge_with_first_arg_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)

    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)
    ex_factories = [executor_stacks.local_executor_factory() for _ in range(5)]
    mergeable_comp_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories)

    expected_result = type_conversions.type_to_py_container(
        expected_sum, after_merge.type_signature.result)
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)
    self.assertEqual(expected_result, result)

  def test_counts_clients_with_noarg_computation(self):
    num_clients = 100
    num_executors = 5
    up_to_merge = build_noarg_count_clients_computation()
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_return_merge_result_with_no_first_arg_computation(
        merge.type_signature.result)

    mergeable_comp_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)
    ex_factories = [
        executor_stacks.local_executor_factory(
            default_num_clients=int(num_clients / num_executors))
        for _ in range(num_executors)
    ]
    mergeable_comp_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        ex_factories)

    expected_result = num_clients
    result = mergeable_comp_context.invoke(mergeable_comp_form, None)
    self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
