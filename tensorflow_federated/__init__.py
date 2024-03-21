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
"""The TensorFlow Federated library."""

import ast as _ast
import inspect as _inspect
import sys as _sys

from absl import logging as _logging

# pylint: disable=g-importing-member
from tensorflow_federated.python import aggregators
from tensorflow_federated.python import analytics
from tensorflow_federated.python import learning
from tensorflow_federated.python import program
from tensorflow_federated.python import simulation
from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import deprecation as _deprecation
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing as profiler
from tensorflow_federated.python.core import backends
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core import templates
from tensorflow_federated.python.core import test
from tensorflow_federated.python.core.environments import jax
from tensorflow_federated.python.core.environments import tensorflow
from tensorflow_federated.python.core.impl import types
from tensorflow_federated.python.core.impl.computation.computation_base import Computation
from tensorflow_federated.python.core.impl.federated_context.data import data
from tensorflow_federated.python.core.impl.federated_context.federated_computation import federated_computation
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_aggregate
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_broadcast
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_eval
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_map
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_max
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_mean
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_min
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_modular_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_select
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_sum_bitwidth
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_select
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_value
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_zip
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_map
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_reduce
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_sum
from tensorflow_federated.python.core.impl.federated_context.value_impl import to_value
from tensorflow_federated.python.core.impl.federated_context.value_impl import Value
from tensorflow_federated.python.core.impl.types.computation_types import FederatedType
from tensorflow_federated.python.core.impl.types.computation_types import FunctionType
from tensorflow_federated.python.core.impl.types.computation_types import SequenceType
from tensorflow_federated.python.core.impl.types.computation_types import StructType
from tensorflow_federated.python.core.impl.types.computation_types import StructWithPythonType
from tensorflow_federated.python.core.impl.types.computation_types import TensorType
from tensorflow_federated.python.core.impl.types.computation_types import to_type
from tensorflow_federated.python.core.impl.types.computation_types import Type
from tensorflow_federated.python.core.impl.types.placements import CLIENTS
from tensorflow_federated.python.core.impl.types.placements import SERVER
from tensorflow_federated.python.core.impl.types.typed_object import TypedObject
from tensorflow_federated.version import __version__
# pylint: enable=g-importing-member

if _sys.version_info < (3, 9):
  raise RuntimeError('TFF only supports Python versions 3.9 or later.')

# TODO: b/305743962 - Remove deprecated API.
jax_computation = _deprecation.deprecated(
    '`tff.jax_computation` is deprecated, use `tff.jax.computation` instead.'
)(jax.computation)
tf_computation = _deprecation.deprecated(
    '`tff.tf_computation` is deprecated, use `tff.tensorflow.computation`'
    ' instead.'
)(tensorflow.computation)

# Initialize a default execution context. This is implicitly executed the
# first time a module in the `core` package is imported.
backends.native.set_sync_local_cpp_execution_context()

# Remove packages that are not part of the public API but are picked up due to
# the directory structure. The python import statements above implicitly add
# these to locals().
del python  # pylint: disable=undefined-variable
del proto  # pylint: disable=undefined-variable

# Update the __dir__ attribute on all TFF modules so that autocompletion tools
# that rely on __dir__ (such as JEDI, IPython, and Colab) only display the
# public APIs symbols shown on tensorflow.org/federated.
_self = _sys.modules[__name__]
_ModuleType = __import__('types').ModuleType


def _get_imported_symbols(module: _ModuleType) -> tuple[str, ...]:
  """Gets a list of only the symbols from explicit import statements."""

  class ImportNodeVisitor(_ast.NodeVisitor):
    """An `ast.Visitor` that collects the names of imported symbols."""

    def __init__(self):
      self.imported_symbols = []

    def _add_imported_symbol(self, node):
      for alias in node.names:
        name = alias.asname or alias.name
        if name == '*':
          continue
        if '.' in name:
          continue
        if name.startswith('_'):
          continue
        self.imported_symbols.append(name)

    def visit_Import(self, node):  # pylint: disable=invalid-name
      self._add_imported_symbol(node)

    def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
      self._add_imported_symbol(node)

  try:
    tree = _ast.parse(_inspect.getsource(module))
  except OSError:
    _logging.debug('Failed to get source code for: %s, skipping...', module)
    tree = None
  if tree is None:
    return ()

  visitor = ImportNodeVisitor()
  visitor.visit(tree)

  return tuple(sorted(visitor.imported_symbols))


def _update_dir_method(
    module: _ModuleType, seen_modules: set[_ModuleType]
) -> None:
  """Overwrites `__dir__` to only return the explicit public API.

  The "public API" is defined as:
    - modules, functions, and classes imported be packages (in __init__.py
      files)
    - functions and classes imported by modules (but not modules imported by
      modules)

  This definition matches the documentation generated at
  http://www.tensorflow.org/federated/api_docs/python/tff.

  To improve JEDI, IPython, and Colab autocomplete consistency with
  tensorflow.org/federated public API documentation, this method traverses
  the modules on import and replaces `__dir__` (the source of autocompletions)
  with only those symbols that were explicitly imported.

  Otherwise, Python imports will implicitly import any submodule in the package,
  exposing it via `__dir__`, which is undesirable.

  Args:
    module: A module to bind a new `__dir__` method to.
    seen_modules: A set of modules that have already been operated on, to reduce
      tree traversal .
  """
  public_attributes = tuple(
      getattr(module, attr, None)
      for attr in dir(module)
      if not attr.startswith('_')
  )

  def _is_tff_submodule(attribute):
    return (
        attribute is not None
        and _inspect.ismodule(attribute)
        and 'tensorflow_federated' in getattr(attribute, '__file__', '')
    )

  tff_submodules = tuple(
      a
      for a in public_attributes
      if _is_tff_submodule(a) and a not in seen_modules
  )
  for submodule in tff_submodules:
    _update_dir_method(submodule, seen_modules)
    seen_modules.add(submodule)
  imported_symbols = _get_imported_symbols(module)
  # Filter out imported modules from modules that are not themselves packages.
  is_package = hasattr(module, '__path__')

  def is_module_imported_by_module(symbol_name: str) -> bool:
    return not is_package and _inspect.ismodule(
        getattr(module, symbol_name, None)
    )

  imported_symbols = [
      symbol_name
      for symbol_name in imported_symbols
      if not is_module_imported_by_module(symbol_name)
  ]
  _logging.debug('Module %s had imported symbols %s', module, imported_symbols)
  module.__dir__ = lambda: imported_symbols


_update_dir_method(_self, seen_modules=set([]))
