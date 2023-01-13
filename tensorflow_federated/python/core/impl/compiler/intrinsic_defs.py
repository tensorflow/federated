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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Definitions of all intrinsic for use within the system."""

import enum
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_factory

_intrinsic_registry = {}


@enum.unique
class BroadcastKind(enum.Enum):
  DEFAULT = 1
  SECURE = 2


@enum.unique
class AggregationKind(enum.Enum):
  DEFAULT = 1
  SECURE = 2


class IntrinsicDef:
  """Represents the definition of an intrinsic.

  This class represents the ultimate source of ground truth about what kinds of
  intrinsics exist and what their types are. To be consuled by all other code
  that deals with intrinsics.
  """

  def __init__(self,
               name: str,
               uri: str,
               type_signature: computation_types.Type,
               aggregation_kind: Optional[AggregationKind] = None,
               broadcast_kind: Optional[BroadcastKind] = None):
    """Constructs a definition of an intrinsic.

    Args:
      name: The short human-friendly name of this intrinsic.
      uri: The URI of this intrinsic.
      type_signature: The type of the intrinsic.
      aggregation_kind: Optional kind of aggregation performed by calls.
      broadcast_kind: Optional kind of broadcast performed by calls.
    """
    py_typecheck.check_type(name, str)
    py_typecheck.check_type(uri, str)
    py_typecheck.check_type(type_signature, computation_types.Type)
    self._name = str(name)
    self._uri = str(uri)
    self._type_signature = type_signature
    self._aggregation_kind = aggregation_kind
    self._broadcast_kind = broadcast_kind
    _intrinsic_registry[str(uri)] = self

  # TODO(b/113112885): Add support for an optional type checking function that
  # can verify whether this intrinsic is applicable to given kinds of arguments,
  # e.g., to allow sum-like functions to be applied only to arguments that are
  # composed of tensors as leaves of a possible nested structure.

  @property
  def name(self):
    return self._name

  @property
  def uri(self):
    return self._uri

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def aggregation_kind(self) -> AggregationKind:
    return self._aggregation_kind

  @property
  def broadcast_kind(self) -> BroadcastKind:
    return self._broadcast_kind

  def __str__(self):
    return self._name

  def __repr__(self):
    return 'IntrinsicDef(\'{}\')'.format(self._uri)


# TODO(b/113112885): Perhaps add a way for these to get auto-registered to
# enable things like lookup by URI, etc., similarly to how it's handled in the
# placements.py.

# TODO(b/113112885): Define the generic equivalents of all operators below,
# i.e., intrinsics that support arbitrary placements, to allow the federated
# operators to be decomposed into expressions that might involve one or more
# layers of intermediate aggregators. The type signatures of these generic
# intrinsics are tentatively defined as follows:
#
# - Place an unplaced value:
#
#     generic_place: <T,p> -> T@p
#
# - Compute an aggregate using the 4-part aggregation operator interface:
#
#     generic_aggregate: <{T}@p,U,(<U,T>->U),(<U,U>->U),(U->R),q> -> R@q
#
# - Compute an unweighted average:
#
#     generic_average: <{T}@p,q> -> T@q
#
# - Broadcast an item:
#
#     generic_broadcast: <T@p,q> -> T@q
#
# - Materialize a federated value as a set of sequences at another placement,
#   with the participants at 'q' collecting from disjoint subsets of 'p' that
#   jointly cover all of 'p'.
#
#     generic_partial_collect: <{T}@p,q> -> {T*}@q
#
# - Materialize a federated value as a single sequence:
#
#     generic_collect: <{T}@p,q> -> T*@q
#
# - Pointwise mapping of constituents of a federated value:
#
#     generic_map: <(T->U),{T}@p> -> {U}@p
#
# - Pointwise mapping of all-equal constituents of a federated value:
#
#     generic_apply: <(T->U),T@p> -> U@p
#
# - Perform one-stage set reduction of a federated value with a given operator,
#   with the participants at 'q' reducing over disjoint subsets of 'p' that
#   jointly cover all of 'p'.
#
#     generic_partial_reduce: <{T}@p,U,(<U,T>->U),q> -> {U}@q
#
# - Perform complete set reduction of a federated value with a given operator:
#
#     generic_reduce: <{T}@p,U,(<U,T>->U),q> -> U@q
#
# - Select and agree on a single member consistuent of a federated value (this
#   is technically need to project {T}@SERVER to T@SERVER in a manner that is
#   formally consistent; a technicality that we do not expect to surface in the
#   user API).
#
#     generic_only: {T}@p -> T@p
#
# - Compute a partial sum of a value (for values with numeric constituents):
#
#     generic_partial_sum: <{T}@p,q> -> {T}@q
#
# - Compute a simple sum of a value (for values with numeric constituents):
#
#     generic_sum: <{T}@p,q> -> T@q
#
# - Compute an average weighted by a numeric non-complex scalar:
#
#     generic_weighted_average: <{T}@p,{U}@p,q> -> T@q
#
# - Transform a pair of federated values into a federated pair (technicality we
#   expect to bury through implicit conversions, TBD).
#
#     generic_zip: <{T}@p,{U}@p> -> {<T,U>}@p

# Computes an aggregate of client items (the first, {T}@CLIENTS-typed parameter)
# using a multi-stage process in which client items are first partially
# aggregated at an intermediate layer, then the partial aggregates are further
# combined, and finally projected into the result. This multi-stage process is
# parameterized by a four-part aggregation interface that consists of the
# following:
# a) The 'zero' in the algebra used at the initial stage (partial aggregation),
#    This is the second, U-typed parameter.
# b) The operator that accumulates T-typed client items into the U-typed partial
#    aggregates. This is the third, (<U,T>->U)-typed parameter.
# c) The operator that combines pairs of U-typed partial aggregates. This is the
#    fourth, (<U,U>->U)-typed parameter.
# d) The operator that projects the top-level aggregate into the final result.
#    This is the fifth, (U->R)-typed parameter.
#
# Conceptually, given a new literal INTERMEDIATE_AGGREGATORS in a single-layer
# aggregation architecture, one could define this intrinsic in terms of generic
# intrinsics defined above, as follows.
#
# @federated_computation
# def federated_aggregate(x, zero, accumulate, merge, report):
#   a = generic_partial_reduce(x, zero, accumulate, INTERMEDIATE_AGGREGATORS)
#   b = generic_reduce(a, zero, merge, SERVER)
#   c = generic_map(report, b)
#   return c
#
# Actual implementations might vary.
#
# Type signature: <{T}@CLIENTS,U,(<U,T>->U),(<U,U>->U),(U->R)> -> R@SERVER
FEDERATED_AGGREGATE = IntrinsicDef(
    'FEDERATED_AGGREGATE',
    'federated_aggregate',
    computation_types.FunctionType(
        parameter=[
            computation_types.at_clients(computation_types.AbstractType('T')),
            computation_types.AbstractType('U'),
            type_factory.reduction_op(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('T')),
            type_factory.binary_op(computation_types.AbstractType('U')),
            computation_types.FunctionType(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('R'))
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('R'))),
    aggregation_kind=AggregationKind.DEFAULT)

# Applies a given function to a value on the server.
#
# Type signature: <(T->U),T@SERVER> -> U@SERVER
FEDERATED_APPLY = IntrinsicDef(
    'FEDERATED_APPLY', 'federated_apply',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U')),
            computation_types.at_server(computation_types.AbstractType('T')),
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('U'))))

# Broadcasts a server item to all clients.
#
# Type signature: T@SERVER -> T@CLIENTS
FEDERATED_BROADCAST = IntrinsicDef(
    'FEDERATED_BROADCAST',
    'federated_broadcast',
    computation_types.FunctionType(
        parameter=computation_types.at_server(
            computation_types.AbstractType('T')),
        result=computation_types.at_clients(
            computation_types.AbstractType('T'), all_equal=True)),
    broadcast_kind=BroadcastKind.DEFAULT)

# Evaluates a function at the clients.
#
# Type signature: (() -> T) -> {T}@CLIENTS
FEDERATED_EVAL_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_EVAL_AT_CLIENTS', 'federated_eval_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.FunctionType(
            None, computation_types.AbstractType('T')),
        result=computation_types.at_clients(
            computation_types.AbstractType('T'))))

# Evaluates a function at the server.
#
# Type signature: (() -> T) -> T@SERVER
FEDERATED_EVAL_AT_SERVER = IntrinsicDef(
    'FEDERATED_EVAL_AT_SERVER', 'federated_eval_at_server',
    computation_types.FunctionType(
        parameter=computation_types.FunctionType(
            None, computation_types.AbstractType('T')),
        result=computation_types.at_server(
            computation_types.AbstractType('T'))))

# Maps member constituents of a client value pointwise using a given mapping
# function that operates independently on each client.
#
# Type signature: <(T->U),{T}@CLIENTS> -> {U}@CLIENTS
FEDERATED_MAP = IntrinsicDef(
    'FEDERATED_MAP', 'federated_map',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U')),
            computation_types.at_clients(computation_types.AbstractType('T')),
        ],
        result=computation_types.at_clients(
            computation_types.AbstractType('U'))))

# Maps member constituents of a client all equal value pointwise using a given
# mapping function that operates independently on each client, as a result of
# this independence, the value is only garunteed to be all equal if the function
# is deterministic.
#
# Type signature: <(T->U),T@CLIENTS> -> U@CLIENTS
FEDERATED_MAP_ALL_EQUAL = IntrinsicDef(
    'FEDERATED_MAP_ALL_EQUAL', 'federated_map_all_equal',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U')),
            computation_types.at_clients(
                computation_types.AbstractType('T'), all_equal=True),
        ],
        result=computation_types.at_clients(
            computation_types.AbstractType('U'), all_equal=True)))

# Computes a simple (equally weighted) mean of client items. Only supported
# for numeric tensor types, or composite structures made up of numeric types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_MEAN = IntrinsicDef(
    'FEDERATED_MEAN',
    'federated_mean',
    computation_types.FunctionType(
        parameter=computation_types.at_clients(
            computation_types.AbstractType('T')),
        result=computation_types.at_server(
            computation_types.AbstractType('T'))),
    aggregation_kind=AggregationKind.DEFAULT)

# Computes the modular sum of client values on the server, securely. Only
# supported for integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,M> -> V@SERVER
FEDERATED_SECURE_MODULAR_SUM = IntrinsicDef(
    'FEDERATED_SECURE_MODULAR_SUM',
    'federated_secure_modular_sum',
    computation_types.FunctionType(
        parameter=[
            computation_types.at_clients(computation_types.AbstractType('V')),
            computation_types.AbstractType('M'),
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('V'))),
    aggregation_kind=AggregationKind.SECURE)

# Computes the sum of client values on the server, securely. Only supported for
# integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,M> -> V@SERVER
FEDERATED_SECURE_SUM = IntrinsicDef(
    'FEDERATED_SECURE_SUM',
    'federated_secure_sum',
    computation_types.FunctionType(
        parameter=[
            computation_types.at_clients(computation_types.AbstractType('V')),
            computation_types.AbstractType('M'),
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('V'))),
    aggregation_kind=AggregationKind.SECURE)

# Computes the sum of client values on the server, securely. Only supported for
# integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,B> -> V@SERVER
FEDERATED_SECURE_SUM_BITWIDTH = IntrinsicDef(
    'FEDERATED_SECURE_SUM_BITWIDTH',
    'federated_secure_sum_bitwidth',
    computation_types.FunctionType(
        parameter=[
            computation_types.at_clients(computation_types.AbstractType('V')),
            computation_types.AbstractType('B'),
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('V'))),
    aggregation_kind=AggregationKind.SECURE)

_SELECT_TYPE = computation_types.FunctionType(
    parameter=[
        computation_types.at_clients(
            computation_types.TensorType(dtype=tf.int32, shape=[None])
        ),  # client_keys
        computation_types.at_server(tf.int32),  # max_key
        computation_types.at_server(
            computation_types.AbstractType('T')
        ),  # server_state
        computation_types.FunctionType(
            [computation_types.AbstractType('T'), tf.int32],
            computation_types.AbstractType('U'),
        ),  # select_fn
    ],
    result=computation_types.at_clients(
        computation_types.SequenceType(computation_types.AbstractType('U'))
    ),
)

# Distributes server values to clients based on client keys.
#
# The signature spelled below is slightly more permissive than instantiation
# allows--the client key type must be a fixed-length vector.
#
# Type signature: <{int32[?]}@C,int32@S,T@S,(<T,int32> -> U)> -> {U}@C
FEDERATED_SELECT = IntrinsicDef(
    'FEDERATED_SELECT',
    'federated_select',
    _SELECT_TYPE,
    broadcast_kind=BroadcastKind.DEFAULT)

# Securely distributes server values to clients based on private client keys.
FEDERATED_SECURE_SELECT = IntrinsicDef(
    'FEDERATED_SECURE_SELECT',
    'federated_secure_select',
    _SELECT_TYPE,
    broadcast_kind=BroadcastKind.SECURE)

# Computes the sum of client values on the server. Only supported for numeric
# types, or nested structures made up of numeric computation_types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_SUM = IntrinsicDef(
    'FEDERATED_SUM',
    'federated_sum',
    computation_types.FunctionType(
        parameter=computation_types.at_clients(
            computation_types.AbstractType('T')),
        result=computation_types.at_server(
            computation_types.AbstractType('T'))),
    aggregation_kind=AggregationKind.DEFAULT)

# Places a value at the clients.
#
# Type signature: T -> T@CLIENTS
FEDERATED_VALUE_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_VALUE_AT_CLIENTS', 'federated_value_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.at_clients(
            computation_types.AbstractType('T'), True)))

# Places a value at the server.
#
# Type signature: T -> T@SERVER
FEDERATED_VALUE_AT_SERVER = IntrinsicDef(
    'FEDERATED_VALUE_AT_SERVER', 'federated_value_at_server',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.at_server(
            computation_types.AbstractType('T'))))

# Computes a weighted mean of client items. Only supported for numeric tensor
# types, or composite structures made up of numeric types. Weights must be
# simple scalar numeric (integer or floating point, not complex) tensor types.
# The types of weights and values must be compatible, i.e., multiplying and
# dividing member constituents of the value by weights should yield results of
# the same type as the type of these member constituents being weighted. Thus,
# for example, one may not supply values containing tf.int32 tensors, as the
# result of weighting such values is of a floating-point type. Casting must be
# injected, where appropriate, by the compiler before invoking this intrinsic.
#
# Type signature: <{T}@CLIENTS,{U}@CLIENTS> -> T@SERVER
FEDERATED_WEIGHTED_MEAN = IntrinsicDef(
    'FEDERATED_WEIGHTED_MEAN',
    'federated_weighted_mean',
    computation_types.FunctionType(
        parameter=[
            computation_types.at_clients(computation_types.AbstractType('T')),
            computation_types.at_clients(computation_types.AbstractType('U'))
        ],
        result=computation_types.at_server(
            computation_types.AbstractType('T'))),
    aggregation_kind=AggregationKind.DEFAULT)

# Zips a tuple of two federated types into a federated tuple.
#
# Type signature: T -> U@CLIENTS
# where `T` is a structure of client-placed values.
# `U` must be identical to `T` with federated placement removed.
FEDERATED_ZIP_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_ZIP_AT_CLIENTS', 'federated_zip_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.at_clients(
            computation_types.AbstractType('U'))))
# Type signature: T -> U@SERVER
# where `T` is a structure of server-placed values.
# `U` must be identical to `T` with federated placement removed.
FEDERATED_ZIP_AT_SERVER = IntrinsicDef(
    'FEDERATED_ZIP_AT_SERVER', 'federated_zip_at_server',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.at_server(
            computation_types.AbstractType('U'))))

# TODO(b/122728050): Define GENERIC_DIVIDE, GENERIC_MULTIPLY, and GENERIC_ONE
# to support intrinsic reductions (see the uses in intrinsic_bodies.py for
# the motivation and usage in support of which we need to define semantics).

# Generic plus operator that accepts a variety of types in federated computation
# context. The range of types 'T' supported to be defined. It should work in a
# natural manner for tensors, tuples, federated types, possibly sequences, and
# maybe even functions (although it's unclear if such generality is desirable).
#
# TODO(b/113123410): Define the range of supported computation_types.
#
# Type signature: <T,T> -> T
GENERIC_PLUS = IntrinsicDef(
    'GENERIC_PLUS', 'generic_plus',
    type_factory.binary_op(computation_types.AbstractType('T')))

# Performs pointwise TensorFlow division on its arguments.
# The type signature of generic divide is determined by TensorFlow's set of
# implicit type equations. For example, dividing `int32` by `int32` in TF
# generates a tensor of type `float64`. There is therefore more structure than
# is suggested by the type signature `<T,T> -> U`.
# Type signature: <T,T> -> U
GENERIC_DIVIDE = IntrinsicDef(
    'GENERIC_DIVIDE', 'generic_divide',
    computation_types.FunctionType([
        computation_types.AbstractType('T'),
        computation_types.AbstractType('T')
    ], computation_types.AbstractType('U')))

# Performs pointwise TensorFlow multiplication on its arguments.
# Type signature: <T,T> -> T
GENERIC_MULTIPLY = IntrinsicDef(
    'GENERIC_MULTIPLY', 'generic_multiply',
    computation_types.FunctionType([computation_types.AbstractType('T')] * 2,
                                   computation_types.AbstractType('T')))
# Generic zero operator that represents zero-filled values of diverse types (to
# be defined, but generally similar to that supported by GENERIC_ADD).
#
# TODO(b/113123410): Define the range of supported computation_types.
#
# Type signature: T
GENERIC_ZERO = IntrinsicDef('GENERIC_ZERO', 'generic_zero',
                            computation_types.AbstractType('T'))

# Maps elements of a sequence using a given mapping function that operates
# independently on each element.
#
# Type signature: <(T->U),T*> -> U*
SEQUENCE_MAP = IntrinsicDef(
    'SEQUENCE_MAP', 'sequence_map',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U')),
            computation_types.SequenceType(computation_types.AbstractType('T')),
        ],
        result=computation_types.SequenceType(
            computation_types.AbstractType('U'))))

# Reduces a sequence using a given 'zero' in the algebra (i.e., the result of
# reducing an empty sequence) and a given reduction operator with the signature
# U,T->U that incorporates a single T-typed element into a U-typed result of
# partial reduction. In the special case of T = U, this corresponds to the
# classical notion of reduction of a set using a commutative associative binary
# operator. The generalized reduction operator (with T != U) must yield the same
# results when repeatedly applied on sequences of elements in any order.
#
# Type signature: <T*,U,(<U,T>->U)> -> U
SEQUENCE_REDUCE = IntrinsicDef(
    'SEQUENCE_REDUCE', 'sequence_reduce',
    computation_types.FunctionType(
        parameter=[
            computation_types.SequenceType(computation_types.AbstractType('T')),
            computation_types.AbstractType('U'),
            type_factory.reduction_op(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('T'))
        ],
        result=computation_types.AbstractType('U')))

# Computes the sum of values in a sequence. Only supported for numeric types
# or nested structures made up of numeric types.
#
# Type signature: T* -> T
SEQUENCE_SUM = IntrinsicDef(
    'SEQUENCE_SUM', 'sequence_sum',
    computation_types.FunctionType(
        parameter=computation_types.SequenceType(
            computation_types.AbstractType('T')),
        result=computation_types.AbstractType('T')))


def uri_to_intrinsic_def(uri) -> Optional[IntrinsicDef]:
  return _intrinsic_registry.get(uri)
