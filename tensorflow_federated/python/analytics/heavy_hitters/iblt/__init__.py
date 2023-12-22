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
"""Libraries for computing private heavy hitters with IBLT."""

from tensorflow_federated.python.analytics.heavy_hitters.iblt.chunkers import CharacterEncoding
from tensorflow_federated.python.analytics.heavy_hitters.iblt.chunkers import create_chunker
from tensorflow_federated.python.analytics.heavy_hitters.iblt.hyperedge_hashers import CoupledHyperEdgeHasher
from tensorflow_federated.python.analytics.heavy_hitters.iblt.hyperedge_hashers import RandomHyperEdgeHasher
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_clipping import ClippingIbltFactory
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_lib import decode_iblt_tf
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_lib import DEFAULT_REPETITIONS
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_lib import IbltDecoder
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_lib import IbltEncoder
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_subsampling import SubsampledIbltFactory
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_tensor import decode_iblt_tensor_tf
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_tensor import IbltTensorDecoder
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_tensor import IbltTensorEncoder
from tensorflow_federated.python.analytics.heavy_hitters.iblt.iblt_tff import build_iblt_computation
from tensorflow_federated.python.analytics.heavy_hitters.iblt.subsample_process import ThresholdSamplingProcess
