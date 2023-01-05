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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A context for execution based on an embedded executor instance."""

import asyncio
from collections.abc import Callable
import contextlib
from typing import Any, Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import retrying
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.execution_contexts import compiler_pipeline
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _unwrap(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  elif isinstance(value, structure.Struct):
    return structure.Struct(
        (k, _unwrap(v)) for k, v in structure.iter_elements(value))
  elif isinstance(value, list):
    return [_unwrap(v) for v in value]
  else:
    return value


def _is_retryable_error(exception):
  return isinstance(exception, executors_errors.RetryableError)


class AsyncExecutionContextValue(typed_object.TypedObject):
  """Wrapper class for values produced by `ExecutionContext`."""

  def __init__(self, value, type_spec):
    py_typecheck.check_type(type_spec, computation_types.Type)
    self._value = value
    self._type_spec = type_spec

  @property
  def type_signature(self):
    return self._type_spec

  @property
  def value(self):
    return self._value


async def _ingest(executor, val, type_spec):
  """A coroutine that handles ingestion.

  Args:
    executor: An instance of `executor_base.Executor`.
    val: The first argument to `AsyncExecutionContext.ingest()`.
    type_spec: The second argument to `AsyncExecutionContext.ingest()`.

  Returns:
    The result of the ingestion.

  Raises:
    TypeError: If the `val` is not a value of type `type_spec`.
  """
  if isinstance(val, executor_value_base.ExecutorValue):
    return val
  elif isinstance(val, ingestable_base.Ingestable):
    val_type = val.type_signature
    py_typecheck.check_type(val_type, computation_types.Type)
    type_spec.check_assignable_from(val_type)
    return await val.ingest(executor)
  elif type_spec.is_sequence():
    if not isinstance(val, tf.data.Dataset):
      val = tensorflow_utils.make_data_set_from_elements(
          graph=None, elements=val, element_type=type_spec.element)
    val = tensorflow_utils.coerce_dataset_to_yield_structures(val)
    # We may have performed a type coercion, e.g. in the case of a
    # SparseTensor-yielding dataset; pass the coerced type down, but only after
    # checking assignability with the original one.
    element_type = computation_types.to_type(val.element_spec)
    type_spec.element.check_assignable_from(element_type)
    return await executor.create_value(
        val, computation_types.SequenceType(element_type))
  elif (isinstance(val, structure.Struct) and not type_spec.is_federated()):
    type_spec.check_struct()
    v_elem = structure.to_elements(val)
    t_elem = structure.to_elements(type_spec)
    if len(v_elem) != len(t_elem):
      raise TypeError(
          'Value {} does not match type {}: mismatching tuple length.'.format(
              val, type_spec))
    for ((vk, _), (tk, _)) in zip(v_elem, t_elem):
      if vk not in [tk, None]:
        raise TypeError(
            'Value {} does not match type {}: mismatching tuple element '
            'names {} vs. {}.'.format(val, type_spec, vk, tk))
    ingested = []
    for (_, v), (_, t) in zip(v_elem, t_elem):
      ingested.append(_ingest(executor, v, t))
    ingested = await asyncio.gather(*ingested)
    return await executor.create_struct(
        structure.Struct(
            (name, val) for (name, _), val in zip(t_elem, ingested)))
  else:
    return await executor.create_value(val, type_spec)


async def _invoke(executor, comp, arg, result_type: computation_types.Type):
  """A coroutine that handles invocation.

  Args:
    executor: An instance of `executor_base.Executor`.
    comp: The first argument to `AsyncExecutionContext.invoke()`.
    arg: The optional second argument to `AsyncExecutionContext.invoke()`.
    result_type: The type signature of the result. This is used to convert the
      execution result into the proper container types.

  Returns:
    The result of the invocation.
  """
  if arg is not None:
    py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
  comp = await executor.create_value(comp)
  result = await executor.create_call(comp, arg)
  py_typecheck.check_type(result, executor_value_base.ExecutorValue)
  result_val = _unwrap(await result.compute())
  return type_conversions.type_to_py_container(result_val, result_type)


class AsyncExecutionContext(context_base.AsyncContext):
  """An asynchronous execution context backed by an `executor_base.Executor`.

  This context's `ingest` and `invoke` methods return Python coroutine objects
  which represent the actual work of ingestion and invocation in the backing
  executor.

  This context will support concurrent invocation of multiple computations if
  their arguments have the same cardinalities.
  """

  def __init__(
      self,
      executor_fn: executor_factory.ExecutorFactory,
      compiler_fn: Optional[Callable[[computation_base.Computation],
                                     Any]] = None,
      *,
      cardinality_inference_fn: cardinalities_utils
      .CardinalityInferenceFnType = cardinalities_utils.infer_cardinalities):
    """Initializes an execution context.

    Args:
      executor_fn: Instance of `executor_factory.ExecutorFactory`.
      compiler_fn: A Python function that will be used to compile a computation.
      cardinality_inference_fn: A Python function specifying how to infer
        cardinalities from arguments (and their associated types). The value
        returned by this function will be passed to the `create_executor` method
        of `executor_fn` to construct a `tff.framework.Executor` instance.
    """
    super().__init__()
    py_typecheck.check_type(executor_fn, executor_factory.ExecutorFactory)
    self._executor_factory = executor_fn
    if compiler_fn is not None:
      py_typecheck.check_callable(compiler_fn)
      self._compiler_pipeline = compiler_pipeline.CompilerPipeline(compiler_fn)
    else:
      self._compiler_pipeline = None
    py_typecheck.check_callable(cardinality_inference_fn)
    self._cardinality_inference_fn = cardinality_inference_fn

  @contextlib.contextmanager
  def _reset_factory_on_error(self, ex_factory, cardinalities):
    try:
      # We pass a copy down to prevent the caller from mutating.
      yield ex_factory.create_executor({**cardinalities})
    except Exception:
      ex_factory.clean_up_executor({**cardinalities})
      raise

  @property
  def executor_factory(self) -> executor_factory.ExecutorFactory:
    return self._executor_factory

  @retrying.retry(
      retry_on_exception_filter=_is_retryable_error,
      wait_max_ms=30 * 1000,
      wait_multiplier=2)
  async def invoke(self, comp, arg):
    if asyncio.iscoroutine(arg):
      # Awaiting if we are passed a coro allows us to install and use the async
      # context in conjunction with ConcreteComputations' implementation of
      # __call__.
      arg = await arg
    comp.type_signature.check_function()
    # Save the type signature before compiling. Compilation currently loses
    # container types, so we must remember them here so that they can be
    # restored in the output.
    result_type = comp.type_signature.result
    if self._compiler_pipeline is not None:
      with tracing.span('ExecutionContext', 'Compile', span=True):
        comp = self._compiler_pipeline.compile(comp)

    with tracing.span('ExecutionContext', 'Invoke', span=True):

      if arg is not None:
        cardinalities = self._cardinality_inference_fn(
            arg, comp.type_signature.parameter)
      else:
        cardinalities = {}

      with self._reset_factory_on_error(self._executor_factory,
                                        cardinalities) as executor:
        py_typecheck.check_type(executor, executor_base.Executor)

        if arg is not None:
          arg = await tracing.wrap_coroutine_in_current_trace_context(
              _ingest(executor, arg, comp.type_signature.parameter))

        return await tracing.wrap_coroutine_in_current_trace_context(
            _invoke(executor, comp, arg, result_type))
