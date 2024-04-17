# Copyright 2024, The TensorFlow Federated Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.impl.types import dtype_utils


class ArrayShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none', None),
      ('object', object()),
  )
  def test_from_proto_raises_not_implemented_error(self, dtype_pb):
    with self.assertRaises(NotImplementedError):
      dtype_utils.from_proto(dtype_pb)

  @parameterized.named_parameters(
      ('none', None),
      ('object', object()),
  )
  def test_to_proto_raises_not_implemented_error(self, dtype):
    with self.assertRaises(NotImplementedError):
      dtype_utils.to_proto(dtype)

  @parameterized.named_parameters(
      ('bool', np.bool_),
      ('int8', np.int8),
      ('int16', np.int16),
      ('int32', np.int32),
      ('int64', np.int64),
      ('uint8', np.uint8),
      ('uint16', np.uint16),
      ('uint32', np.uint32),
      ('uint64', np.uint64),
      ('float16', np.float16),
      ('float32', np.float32),
      ('float64', np.float64),
      ('complex64', np.complex64),
      ('complex128', np.complex128),
      ('str', np.str_),
  )
  def test_is_valid_dtype_returns_true(self, dtype):
    self.assertTrue(dtype_utils.is_valid_dtype(dtype))

  @parameterized.named_parameters(
      ('bytes', np.bytes_),
      ('object', np.object_),
  )
  def test_is_valid_dtype_returns_false(self, dtype):
    self.assertFalse(dtype_utils.is_valid_dtype(dtype))

  @parameterized.named_parameters(
      ('bool', True, np.bool_),
      ('int32_min', int(np.iinfo(np.int32).min), np.int32),
      ('int32_max', int(np.iinfo(np.int32).max), np.int32),
      ('int64_min', int(np.iinfo(np.int64).min), np.int64),
      ('int64_max', int(np.iinfo(np.int64).max), np.int64),
      ('float', 1.0, np.float32),
      ('complex', complex(1.0, 1.0), np.complex128),
      ('str', 'a', np.str_),
      ('bytes', b'a', np.str_),
  )
  def test_infer_dtype(self, obj, expected_value):
    actual_value = dtype_utils.infer_dtype(obj)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('int_min', int(np.iinfo(np.int64).min) - 1),
      ('int_max', int(np.iinfo(np.int64).max) + 1),
  )
  def test_infer_dtype_raises_value_error(self, obj):
    with self.assertRaises(ValueError):
      dtype_utils.infer_dtype(obj)

  @parameterized.named_parameters(
      ('none', None),
      ('object', object()),
  )
  def test_infer_dtype_raises_not_implemented_error(self, obj):
    with self.assertRaises(NotImplementedError):
      dtype_utils.infer_dtype(obj)


if __name__ == '__main__':
  absltest.main()
