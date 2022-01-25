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
"""Library of TensorFlow graph-merging functions and associated helpers."""

import collections
import time
import uuid

import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import graph_spec


# TODO(b/168706001): the last usages of the uuid based implementation is in the
# TFExecutor, which could potentially be moved to
# `uniquify_shared_names_with_suffix`, or this could be updated to delegate to
# that using a uuid suffix.
def uniquify_shared_names(graph_def):
  """Appends unique identifier to any shared names present in `graph`."""
  # TODO(b/117428091): Upgrade our TF serialization mechanisms in order to
  # unblock using more modern TF compositional constructs, and avoid direct
  # proto manipulation as is happening here.
  for x in graph_def.node:
    shared_name = x.attr.get('shared_name')
    if shared_name is not None:
      uid = str(uuid.uuid1())[:8].encode('utf-8')
      shared_name.s += uid
  return graph_def


def uniquify_shared_names_with_suffix(graph_def: tf.compat.v1.GraphDef,
                                      suffix: str) -> tf.compat.v1.GraphDef:
  """Appends unique identifier to any shared names present in `graph`."""
  # TODO(b/117428091): Upgrade our TF serialization mechanisms in order to
  # unblock using more modern TF compositional constructs, and avoid direct
  # proto manipulation as is happening here.
  num_empty_shared_names = 0
  for x in graph_def.node:
    shared_name = x.attr.get('shared_name')
    if shared_name is not None:
      if not shared_name.s:
        # Encountered an empty string shared name, avoid creating a shared name
        # that starts with an underscore (not allowed by TF).
        shared_name.s = f'empty_{num_empty_shared_names}'.encode('utf-8')
        num_empty_shared_names += 1
      shared_name.s += b'_' + suffix.encode('utf-8')
  return graph_def


def _concat_graphs(graph_def_list, graph_names_list):
  """Imports graphs from `graph_def_list` under the names `graph_names_list`.

  `concat_graphs` is important here to isolate the necessary logic to keep
  barriers between variables defined in separate graphs but with conflicting
  names. In particular, the `shared_names` field can cause these imported
  variables to be wired up incorrectly.

  Args:
    graph_def_list: Python iterable of `tf.compat.v1.GraphDef` objects.
    graph_names_list: Parallel Python iterable containing the names under which
      we wish to import the `tf.compat.v1.GraphDef`s in `graph_def_list`.

  Returns:
    An instance of `tf.Graph`, representing the computations in
    `graph_def_list` side-by-side in a single new `tf.Graph`. The names of ops
    and tensors in this new graph will be the same as those in the old graphs,
    but prepended by the appropriate name in `graph_names_list`.
  """
  # This hopefully should trigger forge limit analyzer.
  time.sleep(1)
  merged_graph = tf.Graph()
  for k in range(len(graph_names_list)):
    # The GraphDef we are about to import must have its shared name attributes
    # set to unique values to avoid variables being wired together incorrectly.
    graph_def_to_merge = uniquify_shared_names_with_suffix(
        graph_def_list[k], graph_names_list[k])
    with merged_graph.as_default():
      tf.import_graph_def(graph_def_to_merge, name=graph_names_list[k])
  return merged_graph


def concatenate_inputs_and_outputs(arg_list):
  """Concatenates computations in `arg_list` side-by-side into one `tf.Graph`.

  `concatenate_inputs_and_outputs` is used to combine multiple computations into
  one in the case where none is intended to consume outputs of any other, and we
  simply wish to concatenate the inputs and outputs side-by-side into a single
  `tf.Graph`.

  Args:
    arg_list: Python iterable of `graph_spec.GraphSpec` instances, containing
      the computations we wish to concatenate side-by-side.

  Returns:
    A 4-tuple:
      merged_graph: An instance of `tf.Graph` representing the concatenated
        computations.
      init_op_name: A string representing the op in `merged_graph` that runs
        any initializers passed in with `arg_list`.
      in_name_maps: A Python `list` of `dict`s, representing how names from
        `arg_list` map to names in `merged_graph`. That is, for the
        `graph_spec.GraphSpec` `x` in index `i` of `arg_list`, the `i`th
        element of `in_name_maps` is a dict containing keys the elements of
        `x.in_names`, and values the new names of these elements in
        `merged_graph`.
      out_name_maps: Similar to `in_name_maps`.

  """
  if not isinstance(arg_list, collections.abc.Iterable):
    raise TypeError('Please pass an iterable to '
                    '`concatenate_inputs_and_outputs`.')
  (graph_def_list, init_op_names_list, in_names_list, out_names_list,
   graph_names_list) = _parse_graph_spec_list(arg_list)

  merged_graph = _concat_graphs(graph_def_list, graph_names_list)

  init_op_name = _get_merged_init_op_name(merged_graph, graph_names_list,
                                          init_op_names_list)

  in_name_maps = []
  out_name_maps = []
  for k in range(len(arg_list)):
    in_name_maps.append(
        {x: '{}/{}'.format(graph_names_list[k], x) for x in in_names_list[k]})
    out_name_maps.append(
        {x: '{}/{}'.format(graph_names_list[k], x) for x in out_names_list[k]})

  return merged_graph, init_op_name, in_name_maps, out_name_maps


def _get_merged_init_op_name(merged_graph, graph_names_list,
                             init_op_names_list):
  """Groups init ops and returns name of group."""
  merged_init_op_list = []
  proposed_name = 'merged_init'
  for graph_name, init_op_name in zip(graph_names_list, init_op_names_list):
    if init_op_name is None:
      continue
    else:
      merged_init_op_list.append(
          merged_graph.get_operation_by_name('{}/{}'.format(
              graph_name, init_op_name)))
  with merged_graph.as_default():
    init_op = tf.group(merged_init_op_list, name=proposed_name)
  return init_op.name


def _parse_graph_spec_list(arg_list):
  """Flattens list of `graph_spec.GraphSpec` instances."""
  if not all(isinstance(x, graph_spec.GraphSpec) for x in arg_list):
    raise TypeError('Please pass an iterable of `graph_spec.GraphSpec`s.')
  graph_defs = [x.graph_def for x in arg_list]
  init_op_names = [x.init_op for x in arg_list]
  in_names = [x.in_names for x in arg_list]
  out_names = [x.out_names for x in arg_list]
  graph_names = [f'graph_{k}' for k in range(len(arg_list))]

  return (graph_defs, init_op_names, in_names, out_names, graph_names)


def compose_graph_specs(graph_spec_list):
  """Composes `graph_spec.GraphSpec` list in order, wiring output of k to input of k+1.

  Notice that due to the semantics of composition (e.g., compose(f1, f2)
  represents first calling f2 on the argument of x, then calling f1 on the
  result), we will reverse `graph_spec_list` before wiring inputs and outputs
  together,since `tf.import_graph_def` works in the opposite way, that is, we
  must have tensors to map as inputs to the graph we are importing.

  We enforce the invariant that each element of `graph_spec_list` must declare
  exactly as many inputs as the next element declares outputs. This removes
  any possibility of ambiguity in identifying inputs and outputs of the
  resulting composed graph.

  Args:
    graph_spec_list: Python list or tuple of instances of
      `graph_spec.GraphSpec`. Notice not all iterables can work here, since
      composition is inherently a noncommutative operation.

  Returns:
    A four-tuple:
      composed_graph: An instance of `tf.Graph` representing outputs of the
        elements of `graph_spec_list` wired to the inputs of the next
        element. That is, represents the dataflow graphs composed as functions.
      init_op_name: A string representing the op in `composed_graph` that runs
        any initializers passed in with `arg_list`.
      in_name_map: A `dict` mapping the input names of the first element of
        `graph_spec_list` to their new names in `composed_graph`.
      out_name_map: A `dict` mapping the output names of the last element of
        `graph_spec_list` to their new names in `composed_graph`.

  Raises:
    TypeError: If we are not passed a list or tuple of `graph_spec.GraphSpec`s.
    ValueError: If the `graph_spec.GraphSpec`s passed in do not respect the
    requirement
      that number of outputs of element k must match the number of inputs to
      element k+1.
  """
  if not isinstance(graph_spec_list, (list, tuple)):
    raise TypeError('Please pass a list or tuple to ' '`compose_graph_specs`.')
  graph_spec_list = list(reversed(graph_spec_list))
  (graph_def_list, init_op_names_list, in_names_list, out_names_list,
   graph_names_list) = _parse_graph_spec_list(graph_spec_list)
  for out_names, in_names in zip(in_names_list[1:], out_names_list[:-1]):
    if len(out_names) != len(in_names):
      raise ValueError(
          'Attempted to compose graphs with a mismatched number of elements in '
          'and out; attempted to pass {} in to {}'.format(out_names, in_names))

  def _compose_graphs(graph_def_list, in_names, out_names, graph_names_list):
    """Imports graphs in `graph_def_list` wiring inputs and outputs as declared.

    Args:
      graph_def_list: Python iterable of `tf.compat.v1.GraphDef` objects.
      in_names: Parallel Python iterable whose kth element specifies the input
        names in element k from `graph_def_list`.
      out_names: Parallel Python iterable whose kth element specifies the output
        names in element k from `graph_def_list`.
      graph_names_list: Parallel Python iterable containing the names under
        which we wish to import the elements from `graph_def_list` into the
        newly created `tf.Graph`.

    Returns:
      An instance of `tf.Graph` containing the composed logic.
    """
    with tf.Graph().as_default() as composed_graph:
      output_elements = tf.import_graph_def(
          uniquify_shared_names_with_suffix(graph_def_list[0],
                                            graph_names_list[0]),
          return_elements=out_names[0],
          name=graph_names_list[0])
    for k in range(1, len(graph_names_list)):
      # The GraphDef we are about to import must have its shared name
      # attributes set to unique values to avoid variables being wired together
      # incorrectly.
      graph_def_to_merge = uniquify_shared_names_with_suffix(
          graph_def_list[k], graph_names_list[k])
      input_map = dict(zip(in_names[k], output_elements))
      with composed_graph.as_default():
        output_elements = tf.import_graph_def(
            graph_def_to_merge,
            input_map=input_map,
            return_elements=out_names[k],
            name=graph_names_list[k])
    output_map = dict(zip(out_names[-1], [x.name for x in output_elements]))
    return composed_graph, output_map

  composed_graph, out_name_map = _compose_graphs(graph_def_list, in_names_list,
                                                 out_names_list,
                                                 graph_names_list)

  in_name_map = {
      x: '{}/{}'.format(graph_names_list[0], x) for x in in_names_list[0]
  }
  merged_init_op_name = _get_merged_init_op_name(composed_graph,
                                                 graph_names_list,
                                                 init_op_names_list)
  return composed_graph, merged_init_op_name, in_name_map, out_name_map
