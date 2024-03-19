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
"""A library of construction functions for building block structures."""

from collections.abc import Iterator, Sequence
import functools
import random
import string
from typing import Optional, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_transformations

Index = Union[str, int]
Path = Union[Index, tuple[Index, ...]]


def select_output_from_lambda(
    comp: building_blocks.Lambda, paths: Union[Path, list[Path]]
) -> building_blocks.Lambda:
  """Constructs a new function with result of selecting `paths` from `comp`.

  Args:
    comp: Lambda computation with result type `tff.StructType` from which we
      wish to select the sub-results at `paths`.
    paths: Either a `Path` or list of `Path`s specifying the indices we wish to
      select from the result of `comp`. Each path must be a `tuple` of `str` or
      `int` indices from which to select an output. If `paths` is a list, the
      returned computation will have a `tff.StructType` result holding each of
      the specified selections.

  Returns:
    A version of `comp` with result value the selection from the result of
    `comp` specified by `paths`.
  """
  comp.check_lambda()
  if not isinstance(comp.type_signature.result, computation_types.StructType):
    raise ValueError(
        f'Expected a `tff.StructType`, found {comp.type_signature.result}.'
    )

  def _select_path(result, path: Path):
    if not isinstance(path, tuple):
      path = (path,)
    for index in path:
      if isinstance(result, building_blocks.Struct):
        result = result[index]
      elif isinstance(index, str):
        result = building_blocks.Selection(result, name=index)
      elif isinstance(index, int):
        result = building_blocks.Selection(result, index=index)
      else:
        raise TypeError(
            'Invalid selection type: expected `str` or `int`, '
            f'found value `{index}` of type `{type(index)}`.'
        )
    return result

  if isinstance(paths, list):
    # Avoid duplicating `comp.result` by binding it to a local.
    result_name = next(unique_name_generator(comp))
    result_ref = building_blocks.Reference(
        result_name, comp.result.type_signature
    )
    elements = [_select_path(result_ref, path) for path in paths]
    result = building_blocks.Block(
        [(result_name, comp.result)], building_blocks.Struct(elements)
    )
  else:
    result = _select_path(comp.result, paths)
  return building_blocks.Lambda(
      comp.parameter_name, comp.parameter_type, result
  )


def unique_name_generator(
    comp: building_blocks.ComputationBuildingBlock, prefix: str = '_var'
) -> Iterator[str]:
  """Yields a new unique name that does not exist in `comp`.

  Args:
    comp: The computation building block to use as a reference.
    prefix: The prefix to use when generating unique names. If `prefix` is
      `None` or if `comp` contains any name with this prefix, then a unique
      prefix will be generated from random lowercase ascii characters.
  """
  if comp is not None:
    names = transformation_utils.get_unique_names(comp)
  else:
    names = set()
  while prefix is None or any(n.startswith(prefix) for n in names):
    characters = string.ascii_lowercase
    prefix = '_{}'.format(''.join(random.choice(characters) for _ in range(3)))
  index = 1
  while True:
    yield '{}{}'.format(prefix, index)
    index += 1


@functools.lru_cache()
def create_identity(
    type_signature: computation_types.Type,
) -> building_blocks.Lambda:
  return building_blocks.Lambda(
      'id_arg',
      type_signature,
      building_blocks.Reference('id_arg', type_signature),
  )


class SelectionSpec:
  """Data class representing map from input tuple to selection of result.

  Attributes:
    tuple_index: The index of the source of the selection sequence in the
      desired result of the generated TensorFlow. If this `SelectionSpec`
      appears at index i of a list of `SelectionSpec`s, index j is the source
      for the result of the generated function at index i.
    selection_sequence: A list or tuple representing the selections to make from
      `tuple_index`, so that the list `[0]` for example would represent the
      output is the 0th element of `tuple_index`, while `[0, 0]` would represent
      that the output is the 0th element of the 0th element of `tuple_index`.
  """

  def __init__(self, tuple_index: int, selection_sequence: Sequence[int]):
    self._tuple_index = tuple_index
    self._selection_sequence = selection_sequence

  @property
  def tuple_index(self):
    return self._tuple_index

  @property
  def selection_sequence(self):
    return self._selection_sequence

  def __str__(self):
    return 'SelectionSequence(tuple_index={},selection_sequence={}'.format(
        self._tuple_index, self._selection_sequence
    )

  def __repr__(self):
    return str(self)


def create_federated_getitem_call(
    arg: building_blocks.ComputationBuildingBlock, idx: Union[int, slice]
) -> building_blocks.Call:
  """Creates computation building block passing getitem to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.StructType` from which we wish to pick out item `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.StructType` underlying `arg`.

  Returns:
    Returns a `building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`, the result
    of applying or mapping the appropriate `__getitem__` function, as defined
    by `idx`.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(
      arg.type_signature.member,  # pytype: disable=attribute-error
      computation_types.StructType,
  )
  getitem_comp = create_federated_getitem_comp(arg, idx)
  return create_federated_map_or_apply(getitem_comp, arg)


def create_federated_getattr_call(
    arg: building_blocks.ComputationBuildingBlock, name: str
) -> building_blocks.Call:
  """Creates computation building block passing getattr to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.StructType` from which we wish to pick out item `name`.
    name: String name to address the `computation_types.StructType` underlying
      `arg`.

  Returns:
    Returns a `building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, str)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(
      arg.type_signature.member,  # pytype: disable=attribute-error
      computation_types.StructType,
  )
  getattr_comp = create_federated_getattr_comp(arg, name)
  return create_federated_map_or_apply(getattr_comp, arg)


def create_federated_getattr_comp(
    comp: building_blocks.ComputationBuildingBlock, name: str
) -> building_blocks.Lambda:
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.StructType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.StructType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(
      comp.type_signature.member,  # pytype: disable=attribute-error
      computation_types.StructType,
  )
  py_typecheck.check_type(name, str)
  element_names = [
      x for x, _ in structure.iter_elements(comp.type_signature.member)  # pytype: disable=attribute-error
  ]
  if name not in element_names:
    raise ValueError(
        'The federated value has no element of name `{}`. Value: {}'.format(
            name, comp.formatted_representation()
        )
    )
  apply_input = building_blocks.Reference('x', comp.type_signature.member)  # pytype: disable=attribute-error
  selected = building_blocks.Selection(apply_input, name=name)
  apply_lambda = building_blocks.Lambda(
      'x', apply_input.type_signature, selected
  )
  return apply_lambda


def create_federated_getitem_comp(
    comp: building_blocks.ComputationBuildingBlock, key: Union[int, slice]
) -> building_blocks.Lambda:
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.StructType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.StructType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `Value` objects with
      `type_signature` `computation_types.StructType`.

  Returns:
    Instance of `building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(
      comp.type_signature.member,  # pytype: disable=attribute-error
      computation_types.StructType,
  )
  py_typecheck.check_type(key, (int, slice))
  apply_input = building_blocks.Reference('x', comp.type_signature.member)  # pytype: disable=attribute-error
  if isinstance(key, int):
    selected = building_blocks.Selection(apply_input, index=key)
  else:
    elems = structure.to_elements(comp.type_signature.member)  # pytype: disable=attribute-error
    index_range = range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0], building_blocks.Selection(apply_input, index=k))
      )
    selected = building_blocks.Struct(elem_list)
  apply_lambda = building_blocks.Lambda(
      'x', apply_input.type_signature, selected
  )
  return apply_lambda


def _unname_fn_parameter(fn, unnamed_parameter_type):
  """Coerces `fn` to a comp whose parameter type is `unnamed_parameter_type`."""
  if structure.name_list(fn.type_signature.parameter):  # pytype: disable=attribute-error
    return building_blocks.Lambda(
        'a',
        unnamed_parameter_type,
        building_blocks.Call(
            fn,
            building_blocks.Reference('a', unnamed_parameter_type),
        ),
    )
  else:
    return fn


def create_federated_aggregate(
    value: building_blocks.ComputationBuildingBlock,
    zero: building_blocks.ComputationBuildingBlock,
    accumulate: building_blocks.ComputationBuildingBlock,
    merge: building_blocks.ComputationBuildingBlock,
    report: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated aggregate.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp, Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    zero: A `building_blocks.ComputationBuildingBlock` to use as the initial
      value.
    accumulate: A `building_blocks.ComputationBuildingBlock` to use as the
      accumulate function.
    merge: A `building_blocks.ComputationBuildingBlock` to use as the merge
      function.
    report: A `building_blocks.ComputationBuildingBlock` to use as the report
      function.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(accumulate, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(merge, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(report, building_blocks.ComputationBuildingBlock)
  # Its okay if the first argument of accumulate is assignable from the zero,
  # without being the exact type. This occurs when accumulate has a type like
  # (<int32[?], int32> -> int32[?]) but zero is int32[0].
  zero_arg_type = accumulate.type_signature.parameter[0]  # pytype: disable=attribute-error
  zero_arg_type.check_assignable_from(zero.type_signature)
  result_type = computation_types.FederatedType(
      report.type_signature.result,  # pytype: disable=attribute-error
      placements.SERVER,
  )

  accumulate_parameter_type = computation_types.StructType([
      zero_arg_type,
      value.type_signature.member,  # pytype: disable=attribute-error
  ])
  accumulate = _unname_fn_parameter(accumulate, accumulate_parameter_type)
  merge_parameter_type = computation_types.StructType(
      [zero_arg_type, zero_arg_type]
  )
  merge = _unname_fn_parameter(merge, merge_parameter_type)

  intrinsic_type = computation_types.FunctionType(
      (
          type_conversions.type_to_non_all_equal(value.type_signature),
          zero_arg_type,
          accumulate.type_signature,
          merge.type_signature,
          report.type_signature,
      ),
      result_type,
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_AGGREGATE.uri, intrinsic_type
  )
  values = building_blocks.Struct((value, zero, accumulate, merge, report))
  return building_blocks.Call(intrinsic, values)


def create_federated_apply(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated apply.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      fn.type_signature.result,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type
  )
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_broadcast(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated broadcast.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.CLIENTS,
      all_equal=True,
  )
  intrinsic_type = computation_types.FunctionType(
      value.type_signature, result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_BROADCAST.uri, intrinsic_type
  )
  return building_blocks.Call(intrinsic, value)


def create_federated_eval(
    fn: building_blocks.ComputationBuildingBlock,
    placement: placements.PlacementLiteral,
) -> building_blocks.Call:
  r"""Creates a called federated eval.

            Call
           /    \
  Intrinsic      Comp

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    placement: A `placements.PlacementLiteral` to use as the placement.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri
    all_equal = False
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(
      fn.type_signature.result,  # pytype: disable=attribute-error
      placement,
      all_equal=all_equal,
  )
  intrinsic_type = computation_types.FunctionType(
      fn.type_signature, result_type
  )
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, fn)


def create_federated_map(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  parameter_type = computation_types.FederatedType(
      arg.type_signature.member,  # pytype: disable=attribute-error
      placements.CLIENTS,
  )
  result_type = computation_types.FederatedType(
      fn.type_signature.result,  # pytype: disable=attribute-error
      placements.CLIENTS,
  )
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri, intrinsic_type
  )
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_all_equal(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated map of equal values.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Note: The `fn` is required to be deterministic and therefore should contain no
  `building_blocks.CompiledComputations`.

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  parameter_type = computation_types.FederatedType(
      arg.type_signature.member,  # pytype: disable=attribute-error
      placements.CLIENTS,
      all_equal=True,
  )
  result_type = computation_types.FederatedType(
      fn.type_signature.result,  # pytype: disable=attribute-error
      placements.CLIENTS,
      all_equal=True,
  )
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri, intrinsic_type
  )
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_or_apply(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated map or apply depending on `arg`s placement.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  if arg.type_signature.placement is placements.CLIENTS:  # pytype: disable=attribute-error
    if arg.type_signature.all_equal:  # pytype: disable=attribute-error
      return create_federated_map_all_equal(fn, arg)
    else:
      return create_federated_map(fn, arg)
  elif arg.type_signature.placement is placements.SERVER:  # pytype: disable=attribute-error
    return create_federated_apply(fn, arg)
  else:
    raise TypeError(
        'Unsupported placement {}.'.format(arg.type_signature.placement)  # pytype: disable=attribute-error
    )


def create_federated_mean(
    value: building_blocks.ComputationBuildingBlock,
    weight: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated mean.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    weight: A `building_blocks.ComputationBuildingBlock` to use as the weight or
      `None`.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  if weight is not None:
    py_typecheck.check_type(weight, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  if weight is not None:
    intrinsic_type = computation_types.FunctionType(
        (
            type_conversions.type_to_non_all_equal(value.type_signature),
            type_conversions.type_to_non_all_equal(weight.type_signature),
        ),
        result_type,
    )
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, intrinsic_type
    )
    values = building_blocks.Struct((value, weight))
    return building_blocks.Call(intrinsic, values)
  else:
    intrinsic_type = computation_types.FunctionType(
        type_conversions.type_to_non_all_equal(value.type_signature),
        result_type,
    )
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri, intrinsic_type
    )
    return building_blocks.Call(intrinsic, value)


def create_federated_min(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated min.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    ValueError: If any of the types do not match.
  """
  if not isinstance(value.type_signature, computation_types.FederatedType):
    raise ValueError('Expected a federated value.')
  result_type = computation_types.FederatedType(
      value.type_signature.member,
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      type_conversions.type_to_non_all_equal(value.type_signature), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MIN.uri, intrinsic_type
  )
  return building_blocks.Call(intrinsic, value)


def create_federated_max(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated max.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    ValueError: If any of the types do not match.
  """
  if not isinstance(value.type_signature, computation_types.FederatedType):
    raise ValueError('Expected a federated value.')
  result_type = computation_types.FederatedType(
      value.type_signature.member,
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      type_conversions.type_to_non_all_equal(value.type_signature), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAX.uri, intrinsic_type
  )
  return building_blocks.Call(intrinsic, value)


def create_federated_secure_modular_sum(
    value: building_blocks.ComputationBuildingBlock,
    modulus: building_blocks.ComputationBuildingBlock,
) -> building_blocks.ComputationBuildingBlock:
  r"""Creates a called secure modular sum.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    modulus: A `building_blocks.ComputationBuildingBlock` to use as the
      `modulus` value.

  Returns:
    A `building_blocks.Call`.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(modulus, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      [
          type_conversions.type_to_non_all_equal(value.type_signature),
          modulus.type_signature,
      ],
      result_type,
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM.uri, intrinsic_type
  )

  values = building_blocks.Struct([value, modulus])
  return building_blocks.Call(intrinsic, values)


def create_federated_secure_sum(
    value: building_blocks.ComputationBuildingBlock,
    max_input: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called secure sum.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    max_input: A `building_blocks.ComputationBuildingBlock` to use as the
      `max_input` value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(max_input, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      [
          type_conversions.type_to_non_all_equal(value.type_signature),
          max_input.type_signature,
      ],
      result_type,
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SECURE_SUM.uri, intrinsic_type
  )
  values = building_blocks.Struct([value, max_input])
  return building_blocks.Call(intrinsic, values)


def create_federated_secure_sum_bitwidth(
    value: building_blocks.ComputationBuildingBlock,
    bitwidth: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called secure sum using bitwidth.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    bitwidth: A `building_blocks.ComputationBuildingBlock` to use as the
      bitwidth value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(bitwidth, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      [
          type_conversions.type_to_non_all_equal(value.type_signature),
          bitwidth.type_signature,
      ],
      result_type,
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH.uri, intrinsic_type
  )
  values = building_blocks.Struct([value, bitwidth])
  return building_blocks.Call(intrinsic, values)


def create_federated_select(
    client_keys,
    max_key,
    server_val,
    select_fn,
    secure: bool,
) -> building_blocks.Call:
  """Creates a called `federated_select` or `federated_secure_select`."""
  py_typecheck.check_type(client_keys, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(max_key, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(server_val, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(select_fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(secure, bool)
  single_key_type = max_key.type_signature.member
  select_fn_unnamed_param_type = computation_types.StructType([
      (None, server_val.type_signature.member),
      (None, single_key_type),
  ])
  select_fn = _unname_fn_parameter(select_fn, select_fn_unnamed_param_type)
  result_type = computation_types.FederatedType(
      computation_types.SequenceType(select_fn.type_signature.result),
      placements.CLIENTS,
  )
  intrinsic_type = computation_types.FunctionType(
      [
          type_conversions.type_to_non_all_equal(client_keys.type_signature),
          max_key.type_signature,
          server_val.type_signature,
          select_fn.type_signature,
      ],
      result_type,
  )
  if secure:
    intrinsic_def = intrinsic_defs.FEDERATED_SECURE_SELECT
  else:
    intrinsic_def = intrinsic_defs.FEDERATED_SELECT
  intrinsic = building_blocks.Intrinsic(intrinsic_def.uri, intrinsic_type)
  values = building_blocks.Struct([client_keys, max_key, server_val, select_fn])
  return building_blocks.Call(intrinsic, values)


def create_federated_sum(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      type_conversions.type_to_non_all_equal(value.type_signature), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SUM.uri, intrinsic_type
  )
  return building_blocks.Call(intrinsic, value)


def create_federated_unzip(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Block:
  r"""Creates a tuple of called federated maps or applies.

                Block
               /     \
  [value=Comp]        Tuple
                      |
                      [Call,                        Call, ...]
                      /    \                       /    \
             Intrinsic      Tuple         Intrinsic      Tuple
                            |                            |
                [Lambda(arg), Ref(value)]    [Lambda(arg), Ref(value)]
                            \                            \
                             Sel(0)                       Sel(1)
                                   \                            \
                                    Ref(arg)                     Ref(arg)

  This function returns a tuple of federated values given a `value` with a
  federated tuple type signature.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType` containing at least one element.

  Returns:
    A `building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = structure.to_elements(value.type_signature.member)  # pytype: disable=attribute-error
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  value_ref = building_blocks.Reference('value', value.type_signature)
  elements = []
  fn_ref = building_blocks.Reference('arg', named_type_signatures)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = building_blocks.Selection(fn_ref, index=index)
    fn = building_blocks.Lambda(fn_ref.name, fn_ref.type_signature, sel)
    intrinsic = create_federated_map_or_apply(fn, value_ref)
    elements.append((name, intrinsic))
  result = building_blocks.Struct(
      elements,
      value.type_signature.member.python_container,  # pytype: disable=attribute-error
  )
  symbols = ((value_ref.name, value),)
  return building_blocks.Block(symbols, result)


def create_federated_value(
    value: building_blocks.ComputationBuildingBlock,
    placement: placements.PlacementLiteral,
) -> building_blocks.Call:
  r"""Creates a called federated value.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    placement: A `placements.PlacementLiteral` to use as the placement.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(
      value.type_signature, placement, all_equal=True
  )
  intrinsic_type = computation_types.FunctionType(
      value.type_signature, result_type
  )
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def _check_placements(placement_values: set[placements.PlacementLiteral]):
  """Checks if the placements of the values being zipped are compatible."""
  if not placement_values:
    raise TypeError(
        'federated_zip is only supported on nested structures '
        'containing at least one FederatedType, but none were '
        'found.'
    )
  elif len(placement_values) > 1:
    placement_list = ', '.join(placement.name for placement in placement_values)
    raise TypeError(
        'federated_zip requires all nested FederatedTypes to '
        'have the same placement, but values placed at '
        f'{placement_list} were found.'
    )


def create_federated_zip(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called federated zip.

  This function accepts a value whose type signature is a (potentially) nested
  tuple structure of federated values all with the same placement, and uses
  one of the federated_zip intrinsics (at client or at server) to promote the
  placement to the highest level. E.g., A value of type '<A@S, <<B@S>, C@S>>'
  would be mapped to a value of type '<A, <<B>, C>>@S'.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType` that may contain other nested
      `computation_types.StructTypes` bottoming out in at least one element of
      type `computation_Types.FederatedType`. These federated types must be at
      the same placement.

  Returns:
    A `building_blocks.Call` whose type signature is now a federated
      `computation_types.StructType`, placed at the same placement as the
      leaves of `value`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(value.type_signature, computation_types.StructType)

  all_placements = set()

  def _record_placements(type_signature: computation_types.Type):
    """Records the placements in `type_signature` to `all_placements`."""
    if isinstance(type_signature, computation_types.FederatedType):
      all_placements.add(type_signature.placement)
    elif isinstance(type_signature, computation_types.StructType):
      for i, _ in enumerate(type_signature):
        _record_placements(type_signature[i])
    else:
      raise TypeError(
          'Expected type signatures consisting of structures of StructType '
          'bottoming out in FederatedType, found: \n{}'.format(type_signature)
      )

  _record_placements(value.type_signature)
  _check_placements(all_placements)
  placement = all_placements.pop()
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))

  def normalize_all_equals(element_type):
    if (
        isinstance(element_type, computation_types.FederatedType)
        and element_type.placement is placements.CLIENTS
        and element_type.all_equal
    ):
      return (
          computation_types.FederatedType(
              element_type.member, placements.CLIENTS
          ),
          True,
      )
    return element_type, False

  normalized_input_type, _ = type_transformations.transform_type_postorder(
      value.type_signature, normalize_all_equals
  )

  unplaced_output_type = type_transformations.strip_placement(
      value.type_signature
  )
  output_type = computation_types.FederatedType(unplaced_output_type, placement)
  intrinsic_type = computation_types.FunctionType(
      normalized_input_type, output_type
  )
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_sequence_map(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called sequence map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.SequenceType(fn.type_signature.result)  # pytype: disable=attribute-error
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_MAP.uri, intrinsic_type
  )
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_sequence_reduce(
    value: building_blocks.ComputationBuildingBlock,
    zero: building_blocks.ComputationBuildingBlock,
    op: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called sequence reduce.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    zero: A `building_blocks.ComputationBuildingBlock` to use as the initial
      value.
    op: A `building_blocks.ComputationBuildingBlock` to use as the op function.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(op, building_blocks.ComputationBuildingBlock)
  op_parameter_type = computation_types.StructType([
      zero.type_signature,
      value.type_signature.element,  # pytype: disable=attribute-error
  ])
  op = _unname_fn_parameter(op, op_parameter_type)
  intrinsic_type = computation_types.FunctionType(
      (
          value.type_signature,
          zero.type_signature,
          op.type_signature,
      ),
      op.type_signature.result,  # pytype: disable=attribute-error
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_REDUCE.uri, intrinsic_type
  )
  values = building_blocks.Struct((value, zero, op))
  return building_blocks.Call(intrinsic, values)


def create_sequence_sum(
    value: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Call:
  r"""Creates a called sequence sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  intrinsic_type = computation_types.FunctionType(
      value.type_signature,
      value.type_signature.element,  # pytype: disable=attribute-error
  )
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_SUM.uri, intrinsic_type
  )
  return building_blocks.Call(intrinsic, value)


def _create_naming_function(tuple_type_to_name, names_to_add, container_type):
  """Private function to construct lambda naming a given tuple type.

  Args:
    tuple_type_to_name: Instance of `computation_types.StructType`, the type of
      the argument which we wish to name.
    names_to_add: Python `list` or `tuple`, the names we wish to give to
      `tuple_type_to_name`.
    container_type: Optional Python container type to associate with the
      resulting tuple.

  Returns:
    An instance of `building_blocks.Lambda` representing a function
    which will take an argument of type `tuple_type_to_name` and return a tuple
    with the same elements, but with names in `names_to_add` attached.

  Raises:
    ValueError: If `tuple_type_to_name` and `names_to_add` have different
    lengths.
  """
  py_typecheck.check_type(tuple_type_to_name, computation_types.StructType)
  if len(names_to_add) != len(tuple_type_to_name):  # pytype: disable=wrong-arg-types
    raise ValueError(
        'Number of elements in `names_to_add` must match number of element in '
        'the named tuple type `tuple_type_to_name`; here, `names_to_add` has '
        '{} elements and `tuple_type_to_name` has {}.'.format(
            len(names_to_add),  # pytype: disable=wrong-arg-types
            len(tuple_type_to_name),  # pytype: disable=wrong-arg-types
        )
    )
  naming_lambda_arg = building_blocks.Reference('x', tuple_type_to_name)

  def _create_struct_element(i):
    return (
        names_to_add[i],
        building_blocks.Selection(naming_lambda_arg, index=i),
    )

  named_result = building_blocks.Struct(
      [_create_struct_element(k) for k in range(len(names_to_add))],
      container_type,
  )
  return building_blocks.Lambda(
      'x', naming_lambda_arg.type_signature, named_result
  )


def create_named_tuple(
    comp: building_blocks.ComputationBuildingBlock,
    names: Sequence[str],
    container_type=None,
) -> building_blocks.ComputationBuildingBlock:
  """Creates a computation that applies `names` to `comp`.

  Args:
    comp: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType`.
    names: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to apply to `comp`.
    container_type: Optional Python container type to associated with the
      resulting tuple.

  Returns:
    A `building_blocks.ComputationBuildingBlock` representing a
    tuple with the elements from `comp` and the names from `names` attached to
    the `type_signature` of those elements.

  Raises:
    TypeError: If the types do not match.
  """
  py_typecheck.check_type(names, (list, tuple))
  if not all(isinstance(x, (str, type(None))) for x in names):
    raise TypeError(
        'Expected `names` containing only instances of `str` or '
        '`None`, found {}'.format(names)
    )
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.StructType)
  fn = _create_naming_function(comp.type_signature, names, container_type)
  return building_blocks.Call(fn, comp)


def zip_to_match_type(
    *,
    comp_to_zip: building_blocks.ComputationBuildingBlock,
    target_type: computation_types.Type,
) -> Optional[building_blocks.ComputationBuildingBlock]:
  """Zips computation argument to match target type.

  This function will apply the appropriate federated zips to match `comp_to_zip`
  to the requested type `target_type`, subject to a few caveats. We will
  traverse `computation_types.StructTypes` to match types, so for example we
  would zip `<<T@P, R@P>>` to match `<<T, R>@P>`, but we will not traverse
  `computation_types.FunctionTypes`. Therefore we would not apply a zip to the
  parameter of `(<<T@P, R@P>> -> Q)` to match (<<T, R>@P> -> Q).

  If zipping in this manner cannot match the type of `comp_to_zip` to
  `target_type`, `None` will be returned.

  Args:
    comp_to_zip: Instance of `building_blocks.ComputationBuildingBlock` to
      traverse and attempt to zip to match `target_type`.
    target_type: The type to target when traversing and zipping `comp_to_zip`.

  Returns:
    Either a potentially transformed version of `comp_to_zip` or `None`,
    depending on whether inserting a zip according to the semantics above
    can transformed `comp_to_zip` to the requested type.
  """
  py_typecheck.check_type(comp_to_zip, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(target_type, computation_types.Type)

  def _can_be_zipped_into(
      source_type: computation_types.Type, target_type: computation_types.Type
  ) -> bool:
    """Indicates possibility of the transformation `zip_to_match_type`."""

    def _struct_can_be_zipped_to_federated(
        struct_type: computation_types.StructType,
        federated_type: computation_types.FederatedType,
    ) -> bool:
      placements_encountered = set()

      def _remove_placement(
          subtype: computation_types.Type,
      ) -> tuple[computation_types.Type, bool]:
        if isinstance(subtype, computation_types.FederatedType):
          placements_encountered.add(subtype.placement)
          return subtype.member, True
        return subtype, False

      unplaced_struct, _ = type_transformations.transform_type_postorder(
          struct_type, _remove_placement
      )
      if not (
          all(x is federated_type.placement for x in placements_encountered)
      ):
        return False
      if (
          federated_type.placement is placements.CLIENTS
          and federated_type.all_equal
      ):
        # There is no all-equal clients zip; return false.
        return False
      return federated_type.member.is_assignable_from(unplaced_struct)

    def _struct_elem_zippable(
        source_name, source_element, target_name, target_element
    ):
      return _can_be_zipped_into(
          source_element, target_element
      ) and source_name in (target_name, None)

    if isinstance(source_type, computation_types.StructType):
      if isinstance(target_type, computation_types.FederatedType):
        return _struct_can_be_zipped_to_federated(source_type, target_type)
      elif isinstance(target_type, computation_types.StructType):
        elements_zippable = []
        for (s_name, s_el), (t_name, t_el) in zip(
            structure.iter_elements(source_type),
            structure.iter_elements(target_type),
        ):
          elements_zippable.append(
              _struct_elem_zippable(s_name, s_el, t_name, t_el)
          )
        return all(elements_zippable)
    else:
      return target_type.is_assignable_from(source_type)

  def _zip_to_match(
      *,
      source: building_blocks.ComputationBuildingBlock,
      target_type: computation_types.Type,
  ):
    if isinstance(target_type, computation_types.FederatedType) and isinstance(
        source.type_signature, computation_types.StructType
    ):
      return create_federated_zip(source)
    elif isinstance(target_type, computation_types.StructType) and isinstance(
        source.type_signature, computation_types.StructType
    ):
      zipped_elements = []
      # Bind a reference to the source to prevent duplication in the AST.
      ref_name = next(unique_name_generator(source))
      ref_to_source = building_blocks.Reference(ref_name, source.type_signature)
      for idx, ((_, t_el), (s_name, _)) in enumerate(
          zip(
              structure.iter_elements(target_type),  # pytype: disable=wrong-arg-types
              structure.iter_elements(source.type_signature),  # pytype: disable=wrong-arg-types
          )
      ):
        s_selection = building_blocks.Selection(ref_to_source, index=idx)
        zipped_elements.append(
            (s_name, _zip_to_match(source=s_selection, target_type=t_el))
        )
        # Insert binding above the constructed structure.
        return building_blocks.Block(
            [(ref_name, source)], building_blocks.Struct(zipped_elements)
        )
    else:
      # No zipping to be done here.
      return source

  if target_type.is_assignable_from(comp_to_zip.type_signature):
    # No zipping needs to be done; return directly.
    return comp_to_zip
  elif _can_be_zipped_into(comp_to_zip.type_signature, target_type):
    return _zip_to_match(source=comp_to_zip, target_type=target_type)
  else:
    # Zipping cannot be performed here.
    return None
