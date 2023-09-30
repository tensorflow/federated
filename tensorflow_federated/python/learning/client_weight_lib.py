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
"""Enums for client weighting in learning methods."""

from collections.abc import Callable
import enum
from typing import Any, Union

import tensorflow as tf


class ClientWeighting(enum.Enum):
  """Enum for built-in methods for weighing clients."""

  UNIFORM = 1
  NUM_EXAMPLES = 2
  CUSTOMIZED = 3


ClientWeightFnType = Callable[[Any], tf.Tensor]
ClientWeightType = Union[ClientWeighting, ClientWeightFnType]


def check_is_client_weighting_or_callable(client_weighting):
  if not isinstance(client_weighting, ClientWeighting) and not callable(
      client_weighting
  ):
    raise TypeError(
        '`client_weighting` must be either an instance of '
        '`ClientWeighting` or it must be callable. '
        f'Found type {type(client_weighting)}.'
    )
