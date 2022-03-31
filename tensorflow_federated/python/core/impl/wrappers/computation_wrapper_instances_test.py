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
import tensorflow as tf

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


class TensorflowWrapperTest(test_case.TestCase):

  def test_invoke_with_typed_lambda(self):
    foo = lambda x: x > 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_invoke_with_polymorphic_lambda(self):
    foo = lambda x: x > 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_invoke_with_no_arg_lambda(self):
    foo = lambda: 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo)
    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_invoke_with_typed_fn(self):

    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_invoke_with_polymorphic_fn(self):

    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_invoke_with_no_arg_fn(self):

    def foo():
      return 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)
    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_decorate_as_typed_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_decorate_as_polymorphic_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper
    def foo(x):
      return x > 10

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_decorate_as_no_arg_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper
    def foo():
      return 10

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_invoke_with_typed_tf_function(self):

    @tf.function
    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_invoke_with_polymorphic_tf_function(self):

    @tf.function
    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_invoke_with_no_arg_tf_function(self):

    @tf.function
    def foo():
      return 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)
    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_takes_tuple_typed(self):

    @tf.function
    def foo(t):
      return t[0] + t[1]

    foo = computation_wrapper_instances.tensorflow_wrapper(
        foo, (tf.int32, tf.int32))
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(<int32,int32> -> int32)')

  def test_takes_tuple_polymorphic(self):

    def foo(t):
      return t[0] + t[1]

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructType([
            computation_types.TensorType(tf.int32),
            computation_types.TensorType(tf.int32),
        ]))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(<int32,int32> -> int32)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructType([
            computation_types.TensorType(tf.float32),
            computation_types.TensorType(tf.float32),
        ]))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(<float32,float32> -> float32)')

  def test_takes_structured_tuple_typed(self):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = computation_wrapper_instances.tensorflow_wrapper(
        foo, [
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict([('foo', tf.int32), ('bar', tf.int32)]),
            MyType(tf.int32, tf.int32),
        ])
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '(<x=int32,t=<int32,int32>,l=<int32,int32>,odict=<foo=int32,bar=int32>,my_type=<x=int32,y=int32>> -> int32)'
    )

  def test_takes_structured_tuple_polymorphic(self):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.to_type([
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict([('foo', tf.int32), ('bar', tf.int32)]),
            MyType(tf.int32, tf.int32),
        ]))
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<int32,<int32,int32>,<int32,int32>,<foo=int32,bar=int32>,<x=int32,y=int32>> -> int32)'
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.to_type([
            tf.float32,
            (tf.float32, tf.float32),
            [tf.float32, tf.float32],
            collections.OrderedDict([('foo', tf.float32), ('bar', tf.float32)]),
            MyType(tf.float32, tf.float32),
        ]))
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<float32,<float32,float32>,<float32,float32>,<foo=float32,bar=float32>,<x=float32,y=float32>> -> float32)'
    )

  def test_returns_tuple_structured(self):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo():
      return (
          1,
          (2, 3.0),
          [4, 5.0],
          collections.OrderedDict([('foo', 6), ('bar', 7.0)]),
          MyType(True, False),
      )

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    # pyformat: disable
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '( -> <int32,<int32,float32>,<int32,float32>,<foo=int32,bar=float32>,<x=bool,y=bool>>)'
    )
    # pyformat: enable

  def test_takes_namedtuple_typed(self):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo(x):
      self.assertIsInstance(x, MyType)
      return x.x + x.y

    foo = computation_wrapper_instances.tensorflow_wrapper(
        foo, MyType(tf.int32, tf.int32))
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(<x=int32,y=int32> -> int32)')

  def test_takes_namedtuple_polymorphic(self):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo(t):
      self.assertIsInstance(t, MyType)
      return t.x + t.y

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType([('x', tf.int32),
                                                ('y', tf.int32)], MyType))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(<x=int32,y=int32> -> int32)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType([('x', tf.float32),
                                                ('y', tf.float32)], MyType))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(<x=float32,y=float32> -> float32)')

  def test_with_variable(self):
    v_slot = []

    @tf.function(autograph=False)
    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      v.assign(1)
      return v + x

    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> int32)')

  def test_does_not_raise_type_error_with_sequence_inputs_and_outputs(self):
    try:

      @computation_wrapper_instances.tensorflow_wrapper(
          computation_types.SequenceType(tf.int32))
      def foo(x):  # pylint: disable=unused-variable
        return x

    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_fails_with_bad_types(self):
    function = computation_types.FunctionType(
        None, computation_types.TensorType(tf.int32))
    federated = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    tuple_on_function = computation_types.StructType([federated, function])

    def foo(x):  # pylint: disable=unused-variable
      del x  # Unused.

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type {int32}@CLIENTS'):
      computation_wrapper_instances.tensorflow_wrapper(foo, federated)

    # pylint: disable=anomalous-backslash-in-string
    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type \( -> int32\)'):
      computation_wrapper_instances.tensorflow_wrapper(foo, function)

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type placement'):
      computation_wrapper_instances.tensorflow_wrapper(
          foo, computation_types.PlacementType())

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type T'):
      computation_wrapper_instances.tensorflow_wrapper(
          foo, computation_types.AbstractType('T'))

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type <{int32}@CLIENTS,\( '
        '-> int32\)>'):
      computation_wrapper_instances.tensorflow_wrapper(foo, tuple_on_function)
    # pylint: enable=anomalous-backslash-in-string

  def test_stackframes_in_errors(self):

    class DummyError(RuntimeError):
      pass

    with golden.check_raises_traceback('tensorflow_wrapper_traceback.expected',
                                       DummyError):

      @computation_wrapper_instances.tensorflow_wrapper
      def _():
        raise DummyError()

  def test_error_on_non_callable_non_type(self):
    with golden.check_raises_traceback(
        'non_callable_non_type_traceback.expected', TypeError):
      computation_wrapper_instances.tensorflow_wrapper(5)

  def test_stack_resets_on_none_returned(self):
    stack = get_context_stack.get_context_stack()
    self.assertIsInstance(stack.current,
                          runtime_error_context.RuntimeErrorContext)

    try:

      @computation_wrapper_instances.tensorflow_wrapper()
      def _():
        pass

    except computation_wrapper.ComputationReturnedNoneError:
      self.assertIsInstance(  # pylint: disable=g-assert-in-except
          stack.current, runtime_error_context.RuntimeErrorContext)


class FederatedComputationWrapperTest(test_case.TestCase):

  def test_federated_computation_wrapper(self):

    @computation_wrapper_instances.federated_computation_wrapper(
        (computation_types.FunctionType(tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      return f(f(x))

    self.assertIsInstance(foo, computation_impl.ConcreteComputation)
    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')

    self.assertEqual(
        str(foo.to_building_block()),
        '(foo_arg -> (let fc_foo_symbol_0=foo_arg[0](foo_arg[1]),fc_foo_symbol_1=foo_arg[0](fc_foo_symbol_0) in fc_foo_symbol_1))'
    )

  def test_stackframes_in_errors(self):

    class DummyError(RuntimeError):
      pass

    with golden.check_raises_traceback(
        'federated_computation_wrapper_traceback.expected', DummyError):

      @computation_wrapper_instances.federated_computation_wrapper
      def _():
        raise DummyError()

  def test_empty_tuple_arg(self):

    @computation_wrapper_instances.federated_computation_wrapper(
        computation_types.StructType([]))
    def foo(x):
      return x

    self.assertIsInstance(foo, computation_impl.ConcreteComputation)
    self.assertEqual(str(foo.type_signature), '(<> -> <>)')

    self.assertEqual(str(foo.to_building_block()), '(foo_arg -> <>)')

  def test_stack_resets_on_none_returned(self):
    stack = get_context_stack.get_context_stack()
    self.assertIsInstance(stack.current,
                          runtime_error_context.RuntimeErrorContext)

    try:

      @computation_wrapper_instances.federated_computation_wrapper()
      def _():
        pass

    except computation_wrapper.ComputationReturnedNoneError:
      self.assertIsInstance(  # pylint: disable=g-assert-in-except
          stack.current, runtime_error_context.RuntimeErrorContext)

  def test_arguments_have_python_structural_types(self):

    @computation_wrapper_instances.federated_computation_wrapper(
        collections.OrderedDict(a=(tf.int32,)))
    def _(dict_holding_tuple):
      self.assertIsInstance(dict_holding_tuple, collections.OrderedDict)
      self.assertIsInstance(dict_holding_tuple['a'], tuple)
      value = dict_holding_tuple['a'][0]
      self.assertIsInstance(value, value_impl.Value)
      self.assert_types_identical(value.type_signature,
                                  computation_types.to_type(tf.int32))
      self.assertIsInstance(dict_holding_tuple, collections.OrderedDict)
      return ()

  def test_results_have_python_structural_types(self):

    @computation_wrapper_instances.federated_computation_wrapper
    def make_dict_holding_tuple():
      return collections.OrderedDict(a=(5,))

    @computation_wrapper_instances.federated_computation_wrapper
    def _():
      dict_holding_tuple = make_dict_holding_tuple()
      self.assertIsInstance(dict_holding_tuple, collections.OrderedDict)
      self.assertIsInstance(dict_holding_tuple['a'], tuple)
      value = dict_holding_tuple['a'][0]
      self.assertIsInstance(value, value_impl.Value)
      self.assert_types_identical(value.type_signature,
                                  computation_types.to_type(tf.int32))
      return ()


class AssertReturnsTest(test_case.TestCase):

  def test_basic_non_tff_function_as_decorator_succeeds(self):

    @computation_wrapper_instances.check_returns_type(tf.int32)
    def f():
      return 5

    self.assertEqual(f(), 5)

  def test_basic_non_tff_function_as_decorator_fails(self):

    @computation_wrapper_instances.check_returns_type(tf.int32)
    def f():
      return [5]

    with self.assertRaises(TypeError):
      f()

  def test_basic_non_tff_function_as_nondecorator_succeeds(self):

    def f():
      return 5

    f_wrapped = computation_wrapper_instances.check_returns_type(f, tf.int32)
    self.assertEqual(f_wrapped(), 5)

  def test_basic_non_tff_function_as_nondecorator_fails(self):

    def f():
      return [5]

    f_wrapped = computation_wrapper_instances.check_returns_type(f, tf.int32)
    with self.assertRaises(TypeError):
      f_wrapped()

  def test_with_tensorflow_computation_succeeds(self):

    @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
    @computation_wrapper_instances.check_returns_type(tf.int32)
    def _(x):
      return x

  def test_with_tensorflow_computation_fails(self):
    with self.assertRaises(TypeError):  # pylint: disable=g-error-prone-assert-raises

      @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
      @computation_wrapper_instances.check_returns_type(tf.int32)
      def _(x):
        return (x, x)

  def test_with_tensorflow_computation_picking_up_named_parameters(self):

    @computation_wrapper_instances.tensorflow_wrapper(tf.int32, tf.int32)
    @computation_wrapper_instances.check_returns_type(tf.int32)
    def f(a, b):
      del b
      return a

    self.assertEqual(
        f.type_signature,
        computation_types.FunctionType(
            collections.OrderedDict(a=tf.int32, b=tf.int32), tf.int32))

  def test_fails_with_mismatched_container_type(self):
    with golden.check_raises_traceback(
        'returns_type_container_mismatch_traceback.expected', TypeError):
      # This test fails because it `check_returns_type` with a `tuple`,
      # but returns a `list`.
      @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
      @computation_wrapper_instances.check_returns_type((tf.int32, tf.int32))
      def _(a):
        return [a, a]

  def test_fails_with_more_general_tensorspec(self):
    type_with_known_shape = computation_types.TensorType(tf.int32, [1])
    type_with_unknown_shape = computation_types.TensorType(tf.int32, [None])

    with self.assertRaises(TypeError):  # pylint: disable=g-error-prone-assert-raises

      @computation_wrapper_instances.tensorflow_wrapper(type_with_known_shape)
      @computation_wrapper_instances.check_returns_type(type_with_unknown_shape)
      def _(a):
        return a

  def test_attrs_type(self):

    @attr.s(frozen=True, eq=False, slots=True)
    class MyAttrs:
      a = attr.ib()
      b = attr.ib()

    expected_return_type = MyAttrs(a=tf.int32, b=tf.int32)

    @computation_wrapper_instances.tensorflow_wrapper
    @computation_wrapper_instances.check_returns_type(expected_return_type)
    def _():
      return MyAttrs(a=0, b=0)


class ToComputationImplTest(test_case.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_wrapper_instances.building_block_to_computation(None)

  def test_converts_building_block_to_computation(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    computation_impl_lambda = computation_wrapper_instances.building_block_to_computation(
        lam)
    self.assertIsInstance(computation_impl_lambda,
                          computation_impl.ConcreteComputation)


if __name__ == '__main__':
  test_case.main()
