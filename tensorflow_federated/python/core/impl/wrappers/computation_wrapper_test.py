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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper

tffint32 = computation_types.TensorType(tf.int32)

tffstring = computation_types.TensorType(tf.string)


def build_zero_argument(parameter_type):
  if parameter_type is None:
    return None
  elif parameter_type.is_struct():
    return structure.map_structure(build_zero_argument, parameter_type)
  elif parameter_type == tffint32:
    return 0
  elif parameter_type == tffstring:
    return ''
  else:
    raise NotImplementedError(f'Unsupported type: {parameter_type}')


def _zero_tracer(parameter_type, name=None):
  del name
  zero_argument = build_zero_argument(parameter_type)
  zero_result = yield zero_argument
  yield ZeroTracedFunction(parameter_type, zero_result)


class ZeroTracedFunction(computation_impl.ConcreteComputation):
  """A class that represents a traced function for testing purposes."""

  def __init__(self, parameter_type, zero_result):
    self.zero_result = zero_result
    fn_type = computation_types.FunctionType(parameter_type, tf.string)
    test_proto = pb.Computation(type=type_serialization.serialize_type(fn_type))
    super().__init__(test_proto, context_stack_impl.context_stack, fn_type)


class ContextForTest(context_base.Context):

  def invoke(self, zero_traced_fn, arg):
    return Result(
        arg=arg,
        arg_type=zero_traced_fn.type_signature.parameter,
        zero_result=zero_traced_fn.zero_result)


@attr.s
class Result:
  arg = attr.ib()
  arg_type = attr.ib()
  zero_result = attr.ib()


test_wrap = computation_wrapper.ComputationWrapper(
    computation_wrapper.PythonTracingStrategy(_zero_tracer))


class ComputationWrapperTest(test_case.TestCase):

  # Note: Many tests below silence certain linter warnings. These warnings are
  # not applicable, since it's the wrapper code, not not the whimsy functions
  # that are being tested, so whether the specific function declarations used
  # here follow good practices is not really relevant. The purpose of the test
  # is to exercise various corner cases that the wrapper needs to be able to
  # correctly handle.

  def test_as_decorator_with_kwargs(self):
    with self.assertRaises(TypeError):

      @test_wrap(foo=1)
      def fn():  # pylint: disable=unused-variable
        pass

  def test_as_wrapper_with_kwargs(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda: 5, foo=1)  # pylint: disable=unnecessary-lambda

  def assert_is_return_ten_fn(self, fn):
    self.assertEqual(fn(), Result(arg=None, arg_type=None, zero_result=10))

  def assert_is_add_one_struct_arg_fn(self, fn):
    self.assertEqual(
        fn(10),
        Result(
            arg=structure.Struct([(None, 10)]),
            arg_type=computation_types.StructType([(None, tffint32)]),
            zero_result=1))

  def assert_is_add_one_unary_arg_fn(self, fn):
    self.assertEqual(fn(10), Result(arg=10, arg_type=tffint32, zero_result=1))

  def assert_is_add_two_unnamed_args_fn(self, fn):
    self.assertEqual(
        fn(10, 20),
        Result(
            arg=structure.Struct([(None, 10), (None, 20)]),
            arg_type=computation_types.StructType([(None, tffint32),
                                                   (None, tffint32)]),
            zero_result=0))

  def assert_is_add_two_named_args_fn(self, fn):
    self.assertEqual(
        fn(1, 2),
        Result(
            arg=structure.Struct([('x', 1), ('y', 2)]),
            arg_type=computation_types.StructType([('x', tffint32),
                                                   ('y', tffint32)]),
            zero_result=0))

  def assert_is_add_two_implied_name_args_fn(self, fn):
    expected = Result(
        arg=structure.Struct([('x', 10), ('y', 20)]),
        arg_type=computation_types.to_type(
            collections.OrderedDict(x=tffint32, y=tffint32)),
        zero_result=0,
    )

    self.assertEqual(fn(10, 20), expected, 'without names')
    self.assertEqual(fn(x=10, y=20), expected, 'with names')
    self.assertEqual(fn(y=20, x=10), expected, 'with names reversed')
    self.assertEqual(fn(10, y=20), expected, 'with only one name')

  def test_raises_on_none_returned(self):
    with self.assertRaises(computation_wrapper.ComputationReturnedNoneError):

      @test_wrap
      def _():
        pass

  def test_as_decorator_without_arguments_on_no_parameter_py_fn(self):

    @test_wrap
    def my_fn():
      """This is my fn."""
      return 10

    self.assert_is_return_ten_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_without_arguments_on_no_parameter_lambda(self):
    self.assert_is_return_ten_fn(test_wrap(lambda: 10))

  def test_as_decorator_with_empty_arguments_on_no_parameter_py_fn(self):

    @test_wrap()
    def my_fn():
      """This is my fn."""
      return 10

    self.assert_is_return_ten_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_one_argument_on_no_parameter_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap(tf.int32)
      def my_fn():  # pylint: disable=unused-variable
        pass

  def test_as_wrapper_with_one_argument_on_no_parameter_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda: None, tf.int32)

  def test_as_decorator_with_one_argument_on_one_parameter_py_fn(self):

    @test_wrap(tf.int32)
    def my_fn(x):
      """This is my fn."""
      return x + 1

    self.assert_is_add_one_unary_arg_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_one_argument_on_one_parameter_lambda(self):
    self.assert_is_add_one_unary_arg_fn(test_wrap(lambda x: x + 1, tf.int32))

  def test_as_decorator_with_non_tuple_argument_on_two_parameter_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap(tf.int32)
      def my_fn(x, y):  # pylint: disable=unused-variable
        del x, y  # Unused.
        pass

  def test_as_wrapper_with_non_tuple_argument_on_two_parameter_lambda(self):
    with self.assertRaises(TypeError):

      def my_fn(x, y):
        del x, y  # Unused.
        pass

      test_wrap(my_fn, tf.int32)

  def test_as_decorator_with_two_tuple_argument_on_three_param_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap((tf.int32, tf.int32))
      def my_fn(x, y, z):  # pylint: disable=unused-variable
        del x, y, z  # Unused.
        pass

  def test_as_wrapper_with_two_tuple_argument_on_three_param_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, y, z: None, (tf.int32, tf.int32))

  def test_as_decorator_with_arg_name_mismatching_element_name_in_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap([('x', tf.int32), ('y', tf.int32)])
      def my_fn(x, z):  # pylint: disable=unused-variable
        del x, z  # Unused.
        pass

  def test_as_wrapper_with_arg_name_mismatching_element_name_in_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, z: None, [('x', tf.int32), ('y', tf.int32)])

  def test_as_decorator_with_tuple_params_on_two_parameter_py_fn(self):

    @test_wrap((tf.int32, tf.int32))
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assert_is_add_two_implied_name_args_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_tuple_params_on_two_parameter_py_fn(self):
    wrapped = test_wrap(lambda x, y: x + y, (tf.int32, tf.int32))
    self.assert_is_add_two_implied_name_args_fn(wrapped)

  def test_as_decorator_with_tuple_params_on_one_parameter_py_fn(self):
    # Computations only have a single parameter (or none), and we allow the
    # flexibility of feeding tuple-like parameters in pieces by specifying
    # tuple elementas as multiple arguments. This is independent of how the
    # backing Python function binds to the argument on the definition side.
    # Thus, the ordinary linter check is inapplicable, as there's exists no
    # direct connection between the signature of the call and that of the
    # Python definition. The TFF type decouples one from the other.
    @test_wrap([('x', tf.int32), ('y', tf.int32)])
    def my_fn(arg):
      """This is my fn."""
      return arg['x'] + arg['y']

    self.assertEqual(
        my_fn(1, 2),  # pylint: disable=too-many-function-args
        Result(
            arg=structure.Struct([('x', 1), ('y', 2)]),
            arg_type=computation_types.StructType([('x', tffint32),
                                                   ('y', tffint32)]),
            zero_result=0,
        ))
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_tuple_params_on_one_parameter_py_fn(self):
    self.assertEqual(
        test_wrap(lambda arg: arg[0] + arg[1], (tf.int32, tf.int32))(1, 2),
        Result(
            arg=structure.Struct([(None, 1), (None, 2)]),
            arg_type=computation_types.to_type((tffint32, tffint32)),
            zero_result=0,
        ))

  def test_as_decorator_with_named_tuple_params_on_two_param_py_fn(self):

    @test_wrap([('x', tf.int32), ('y', tf.int32)])
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assert_is_add_two_named_args_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_named_tuple_params_on_two_param_py_fn(self):
    wrapped = test_wrap(lambda x, y: x + y, [('x', tf.int32), ('y', tf.int32)])
    self.assert_is_add_two_named_args_fn(wrapped)

  def test_as_decorator_without_arguments_on_py_fn_with_one_param(self):

    @test_wrap
    def my_fn(x):
      """This is my fn."""
      return x + 1

    self.assert_is_add_one_struct_arg_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_without_arguments_on_py_fn_with_one_param(self):
    wrapped = test_wrap(lambda x: x + 1)
    self.assert_is_add_one_struct_arg_fn(wrapped)

  def test_as_decorator_without_arguments_on_py_fn_with_two_params(self):

    @test_wrap
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assert_is_add_two_unnamed_args_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_empty_arguments_on_py_fn_with_one_param(self):

    @test_wrap()
    def my_fn(x):
      """This is my fn."""
      return x + 1

    self.assert_is_add_one_struct_arg_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_empty_arguments_on_py_fn_with_two_params(self):

    @test_wrap()
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assert_is_add_two_unnamed_args_fn(my_fn)
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_with_integer_args(self):
    with self.assertRaises(TypeError):
      test_wrap(10, 20)

  def test_with_varargs_no_type(self):
    with self.assertRaises(TypeError):

      @test_wrap
      def _(*args):
        """This is my fn."""
        return sum(args)

  def test_with_varargs_scalar_type(self):
    with self.assertRaises(TypeError):

      @test_wrap(tf.int32)
      def _(*args):
        """This is my fn."""
        return sum(args)

  def test_with_varargs_tuple_type(self):
    with self.assertRaises(TypeError):

      @test_wrap([tf.int32, tf.int32, tf.int32, tf.int32])
      def _(x, y, *args):
        """This is my fn."""
        return x + y + sum(args)

  def test_with_kwargs_no_type(self):
    with self.assertRaises(TypeError):

      @test_wrap
      def _(**kwargs):
        """This is my fn."""
        return kwargs['x'] / kwargs['y']

  def test_as_decorator_with_unbundled_arguments(self):

    @test_wrap(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    self.assert_is_add_two_implied_name_args_fn(foo)

  def test_as_decorator_with_named_positional_arguments(self):

    @test_wrap(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    self.assert_is_add_two_implied_name_args_fn(foo)

  def test_as_decorator_with_optional_arguments(self):
    with self.assertRaisesRegex(TypeError, 'default'):

      @test_wrap(tf.int32, tf.int32)
      def _(x=10, y=20):
        return x + y

  def test_as_wrapper_with_unbundled_arguments(self):
    foo = test_wrap(lambda x, y: x + y, tf.int32, tf.int32)
    self.assert_is_add_two_implied_name_args_fn(foo)

  def test_as_wrapper_with_one_argument_instance_method(self):

    class IntWrapper:

      def __init__(self, x):
        self._x = x

      def multiply_by(self, y):
        return self._x * y

    five = IntWrapper(5)
    wrapped = test_wrap(five.multiply_by, tf.int32)
    self.assertEqual(
        wrapped(2), Result(
            arg=2,
            arg_type=tffint32,
            zero_result=0,
        ))

  def test_as_wrapper_with_no_argument_instance_method(self):

    class C:

      def __init__(self, x):
        self._x = x

      def my_method(self):
        return self._x

    c = C(99)
    wrapped = test_wrap(c.my_method)
    self.assertEqual(wrapped(), Result(arg=None, arg_type=None, zero_result=99))

  def test_as_wrapper_with_class_property(self):

    class C:

      @property
      def x(self):
        return 99

    c = C()
    with self.assertRaises(TypeError):
      test_wrap(c.x)

  def test_as_wrapper_with_classmethod(self):

    class C:

      @classmethod
      def prefix(cls, msg):
        return f'{cls.__name__}_{msg}'

    wrapped = test_wrap(C.prefix)
    self.assertEqual(
        wrapped('foo'),
        Result(
            arg=structure.Struct([(None, 'foo')]),
            arg_type=computation_types.StructType([None, tffstring]),
            zero_result='C_',
        ))


if __name__ == '__main__':
  with context_stack_impl.context_stack.install(ContextForTest()):
    test_case.main()
