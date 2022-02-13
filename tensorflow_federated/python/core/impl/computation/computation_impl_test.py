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

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class ComputationImplTest(test_case.TestCase):

  def test_raises_when_invoked_from_tf_function(self):
    # TODO(b/201214708): Consider removing this error once it is possible to
    # invoke a `tff.tf_computation` from inside a `tf.function` without issue.
    computation_type = computation_types.FunctionType(tf.int32, tf.int32)
    computation_proto = pb.Computation(
        type=type_serialization.serialize_type(computation_type),
        data=pb.Data(uri='placeholder_data_uri'))
    computation = computation_impl.ConcreteComputation(
        computation_proto, context_stack_impl.context_stack)

    @tf.function
    def some_annotated_function():
      with self.assertRaises(computation_impl.InvokedInsideTfFunctionError):
        computation()

    some_annotated_function()

  def test_something(self):
    # TODO(b/113112108): Revise these tests after a more complete implementation
    # is in place.

    # At the moment, this should succeed, as both the computation body and the
    # type are well-formed.
    computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type':
                    type_serialization.serialize_type(
                        computation_types.FunctionType(tf.int32, tf.int32)),
                'intrinsic':
                    pb.Intrinsic(uri='whatever')
            }), context_stack_impl.context_stack)

    # This should fail, as the proto is not well-formed.
    self.assertRaises(TypeError, computation_impl.ConcreteComputation,
                      pb.Computation(), context_stack_impl.context_stack)

    # This should fail, as "10" is not an instance of pb.Computation.
    self.assertRaises(TypeError, computation_impl.ConcreteComputation, 10,
                      context_stack_impl.context_stack)

  def test_with_type_preserves_python_container(self):
    struct_return_type = computation_types.FunctionType(
        tf.int32, computation_types.StructType([(None, tf.int32)]))
    original_comp = computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type': type_serialization.serialize_type(struct_return_type),
                'intrinsic': pb.Intrinsic(uri='whatever')
            }), context_stack_impl.context_stack)

    list_return_type = computation_types.FunctionType(
        tf.int32,
        computation_types.StructWithPythonType([(None, tf.int32)], list))
    fn_with_annotated_type = computation_impl.ConcreteComputation.with_type(
        original_comp, list_return_type)
    self.assert_types_identical(list_return_type,
                                fn_with_annotated_type.type_signature)

  def test_with_type_raises_non_assignable_type(self):
    int_return_type = computation_types.FunctionType(tf.int32, tf.int32)
    original_comp = computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type': type_serialization.serialize_type(int_return_type),
                'intrinsic': pb.Intrinsic(uri='whatever')
            }), context_stack_impl.context_stack)

    list_return_type = computation_types.FunctionType(
        tf.int32,
        computation_types.StructWithPythonType([(None, tf.int32)], list))
    with self.assertRaises(computation_types.TypeNotAssignableError):
      computation_impl.ConcreteComputation.with_type(original_comp,
                                                     list_return_type)


if __name__ == '__main__':
  test_case.main()
