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
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_utils
from tensorflow_federated.python.core.impl.types import computation_types

TestNamedTuple = collections.namedtuple('TestTuple', ['x', 'y'])


class ZeroOrOneArgFnToBuildingBlockTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'nested_fn_same',
          lambda f, x: f(f(x)),
          computation_types.StructType([
              ('f', computation_types.FunctionType(np.int32, np.int32)),
              ('x', np.int32),
          ]),
          (
              '(FEDERATED_foo -> (let '
              'fc_FEDERATED_symbol_0=FEDERATED_foo.f(FEDERATED_foo.x),'
              'fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0)'
              ' in fc_FEDERATED_symbol_1))'
          ),
      ),
      (
          'nested_fn_different',
          lambda f, g, x: f(g(x)),
          computation_types.StructType([
              ('f', computation_types.FunctionType(np.int32, np.int32)),
              ('g', computation_types.FunctionType(np.int32, np.int32)),
              ('x', np.int32),
          ]),
          (
              '(FEDERATED_foo -> (let '
              'fc_FEDERATED_symbol_0=FEDERATED_foo.g(FEDERATED_foo.x),'
              'fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0)'
              ' in fc_FEDERATED_symbol_1))'
          ),
      ),
      (
          'selection',
          lambda x: (x[1], x[0]),
          computation_types.StructType([np.int32, np.int32]),
          '(FEDERATED_foo -> <FEDERATED_foo[1],FEDERATED_foo[0]>)',
      ),
      (
          'constant',
          lambda: 'stuff',
          None,
          (
              '( -> (let '
              "fc_FEDERATED_symbol_0=( -> b'stuff')()"
              ' in fc_FEDERATED_symbol_0))'
          ),
      ),
  )
  def test_returns_result(self, fn, parameter_type, fn_str):
    parameter_name = 'foo' if parameter_type is not None else None
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    result, _ = (
        federated_computation_utils.zero_or_one_arg_fn_to_building_block(
            fn, parameter_name, parameter_type, context_stack_impl.context_stack
        )
    )
    self.assertEqual(str(result), fn_str)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tuple',
       lambda x: (x[1], x[0]),
       computation_types.StructType([np.int32, np.float32]),
       computation_types.StructWithPythonType([
           (None, np.float32), (None, np.int32)], tuple)),
      ('list',
       lambda x: [x[1], x[0]],
       computation_types.StructType([np.int32, np.float32]),
       computation_types.StructWithPythonType([
           (None, np.float32), (None, np.int32)], list)),
      ('odict',
       lambda x: collections.OrderedDict([('A', x[1]), ('B', x[0])]),
       computation_types.StructType([np.int32, np.float32]),
       computation_types.StructWithPythonType([
           ('A', np.float32), ('B', np.int32)], collections.OrderedDict)),
      ('namedtuple',
       lambda x: TestNamedTuple(x=x[1], y=x[0]),
       computation_types.StructType([np.int32, np.float32]),
       computation_types.StructWithPythonType([
           ('x', np.float32), ('y', np.int32)], TestNamedTuple)),
  )
  # pyformat: enable
  def test_returns_result_with_py_container(
      self, fn, parameter_type, expected_result_type
  ):
    parameter_name = 'foo'
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    _, type_signature = (
        federated_computation_utils.zero_or_one_arg_fn_to_building_block(
            fn, parameter_name, parameter_type, context_stack_impl.context_stack
        )
    )
    self.assertIs(type(type_signature.result), type(expected_result_type))
    self.assertIs(
        type_signature.result.python_container,
        expected_result_type.python_container,
    )
    self.assertEqual(type_signature.result, expected_result_type)

  def test_raises_value_error_with_none_result(self):
    fn = lambda: None
    parameter_type = None
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)

    with self.assertRaises(computation_wrapper.ComputationReturnedNoneError):
      federated_computation_utils.zero_or_one_arg_fn_to_building_block(
          fn, None, parameter_type, context_stack_impl.context_stack
      )


if __name__ == '__main__':
  absltest.main()
