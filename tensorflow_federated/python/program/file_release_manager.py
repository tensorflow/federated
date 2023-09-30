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
"""Utilities for releasing values from a federated program to a file system.

Note: This library uses `tf.io.gfile` to perform file system operations, this
means that this library:

  * supports all the file systems supported by `tf.io.gfile`
  * encodes files in the same way as `tf.io.gfile`
"""

import asyncio
import collections
from collections.abc import Iterable, Mapping, Sequence
import csv
import enum
import os
import os.path
import random
from typing import Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


class CSVKeyFieldnameNotFoundError(Exception):
  """Raised when a file does not contain the key fieldname."""

  def __init__(
      self,
      file_path: Union[bytes, str, os.PathLike[Union[bytes, str]]],
      key_fieldname: str,
  ):
    super().__init__(
        f'Expected the file {file_path} to contain the key fieldname of '
        f'{key_fieldname}. It is possible that the file was not created by '
        'a `tff.program.CSVFileReleaseManager` or was created with a different '
        '`key_fieldname`.'
    )


@enum.unique
class CSVSaveMode(enum.Enum):
  APPEND = 'append'
  WRITE = 'write'
  READ = 'read'


class CSVFileReleaseManager(
    release_manager.ReleaseManager[release_manager.ReleasableStructure, int]
):
  """A `tff.program.ReleaseManager` that releases values to a CSV file.

  A `tff.program.CSVFileReleaseManager` is a utility for releasing values
  from a federated program to a CSV file and is used to release values from
  platform storage to customer storage in a federated program.

  Values are released to the file system as a CSV file and are quoted as
  strings. When the value is released, if the value is a value reference or a
  structure containing value references, each value reference is materialized.
  The value is then flattened, converted to a `numpy.ndarray`, and then
  converted to a nested list of Python scalars, and released as a CSV file.
  For example, `1` will be written as `'1'` and `tf.constant([[1, 1], [1, 1])`
  will be written as `'[[1, 1], [1, 1]'`.

  This manager can be configured to release values using a `save_mode` of either
  `CSVSaveMode.APPEND` or `CSVSaveMode.WRITE`.

  * In append mode, when a value is released, this manager will try and append
    the value to the CSV file instead of overwriting the existing file. While
    potentially more efficient, append mode is incompatible with compressed
    files (e.g. `.bz2` formats) and encoded directories. This mode is equivalent
    to write mode when releasing a value with a different structure than the
    currently released values, so it may not be useful when values with
    different structures are being released frequently.

  * In write mode (or in append mode when releasing new structures), when a
    value is realeased, this manager reads the entire CSV file and overwrites
    the existing file with the additional values. This can be slower than append
    mode, but is compatible with compressed files (e.g. `.bz2` formats) and
    encoded directories.
  """

  def __init__(
      self,
      file_path: Union[bytes, str, os.PathLike[Union[bytes, str]]],
      save_mode: CSVSaveMode = CSVSaveMode.APPEND,
      key_fieldname: str = 'key',
  ):
    """Returns an initialized `tff.program.CSVFileReleaseManager`.

    Args:
      file_path: A path on the file system to save releases values. If this file
        does not exist it will be created.
      save_mode: A `tff.program.CSVSaveMode` specifying how to save released
        values.
      key_fieldname: A `str` specifying the fieldname used for the key when
        saving released value.

    Raises:
      ValueError: If `file_path` or `key_fieldname` is an empty string.
      CSVKeyFieldnameNotFoundError: If the file exists but does not contain a
        fieldname of `key_fieldname`.
    """
    py_typecheck.check_type(file_path, (bytes, str, os.PathLike))
    if not file_path:
      raise ValueError('Expected `file_path` to not be an empty string.')
    py_typecheck.check_type(save_mode, CSVSaveMode)
    py_typecheck.check_type(key_fieldname, str)
    if not key_fieldname:
      raise ValueError('Expected `key_fieldname` to not be an empty string.')

    file_dir = os.path.dirname(file_path)
    if not tf.io.gfile.exists(file_dir):
      tf.io.gfile.makedirs(file_dir)
    self._file_path = file_path
    self._save_mode = save_mode
    self._key_fieldname = key_fieldname

    if tf.io.gfile.exists(self._file_path):
      fieldnames, values = self._read_values()
      if self._key_fieldname not in fieldnames:
        raise CSVKeyFieldnameNotFoundError(self._file_path, self._key_fieldname)
      if values:
        self._latest_key = int(values[-1][self._key_fieldname])
      else:
        self._latest_key = None
    else:
      self._write_values([self._key_fieldname], [])
      self._latest_key = None

  def _read_values(self) -> tuple[list[str], list[dict[str, str]]]:
    """Returns a tuple of fieldnames and values from the managed CSV."""
    with tf.io.gfile.GFile(self._file_path, 'r') as file:
      reader = csv.DictReader(file)
      if reader.fieldnames is not None:
        fieldnames = list(reader.fieldnames)
      else:
        fieldnames = []
      values = list(reader)
    return fieldnames, values

  def _write_values(
      self,
      fieldnames: Sequence[str],
      values: Iterable[Mapping[str, release_manager.ReleasableStructure]],
  ) -> None:
    """Writes `fieldnames` and `values` to the managed CSV."""
    py_typecheck.check_type(fieldnames, Sequence)
    if isinstance(fieldnames, str):
      raise TypeError(
          'Expected `fieldnames` to be a `Sequence` of `str`, found `str`.'
      )
    for fieldname in fieldnames:
      py_typecheck.check_type(fieldname, str)
    py_typecheck.check_type(values, Iterable)
    for value in values:
      py_typecheck.check_type(value, Mapping)
      for key in value.keys():
        py_typecheck.check_type(key, str)

    path = os.fspath(self._file_path)

    # Create a temporary file.
    temp_path = f'{path}_temp{random.randint(1000, 9999)}'
    if tf.io.gfile.exists(temp_path):
      tf.io.gfile.remove(temp_path)

    # Write to the temporary file.
    with tf.io.gfile.GFile(temp_path, 'w') as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(values)

    # Rename the temporary file to the final location atomically.
    tf.io.gfile.rename(temp_path, self._file_path, overwrite=True)

  async def _write_value(
      self, value: Mapping[str, release_manager.ReleasableStructure]
  ) -> None:
    """Writes `value` to the managed CSV."""
    py_typecheck.check_type(value, Mapping)
    for key in value.keys():
      py_typecheck.check_type(key, str)

    loop = asyncio.get_running_loop()
    fieldnames, values = await loop.run_in_executor(None, self._read_values)
    fieldnames.extend([x for x in value.keys() if x not in fieldnames])
    values.append(value)
    await loop.run_in_executor(None, self._write_values, fieldnames, values)

  async def _append_value(
      self, value: Mapping[str, release_manager.ReleasableStructure]
  ) -> None:
    """Appends `value` to the managed CSV."""
    py_typecheck.check_type(value, Mapping)
    for key in value.keys():
      py_typecheck.check_type(key, str)

    def _read_fieldnames_only() -> list[str]:
      with tf.io.gfile.GFile(self._file_path, 'r') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is not None:
          fieldnames = list(reader.fieldnames)
        else:
          fieldnames = []
      return fieldnames

    def _append_value(
        fieldnames: Sequence[str],
        value: Mapping[str, release_manager.ReleasableStructure],
    ) -> None:
      try:
        with tf.io.gfile.GFile(self._file_path, 'a') as file:
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(value)
      except (tf.errors.PermissionDeniedError, csv.Error) as e:
        raise PermissionError(
            f'Could not append a value to the file {self._file_path}. It is '
            'possible that this file is compressed or encoded. Please use '
            'write mode instead of append mode to release values to this '
            'file using a `tff.program.CSVFileReleaseManager`.'
        ) from e

    loop = asyncio.get_running_loop()
    fieldnames = await loop.run_in_executor(None, _read_fieldnames_only)
    if all(key in fieldnames for key in value.keys()):
      await loop.run_in_executor(None, _append_value, fieldnames, value)
    else:
      await self._write_value(value)

  async def _remove_values_greater_than(self, key: int) -> None:
    """Removes all values greater than `key` from the managed CSV."""
    py_typecheck.check_type(key, (int, np.integer))

    if self._latest_key is None or key > self._latest_key:
      return

    loop = asyncio.get_running_loop()
    if key < 0:
      await loop.run_in_executor(
          None, self._write_values, [self._key_fieldname], []
      )
      self._latest_key = None
    else:
      loop = asyncio.get_running_loop()
      filtered_fieldnames = [self._key_fieldname]
      filtered_values = []
      _, values = await loop.run_in_executor(None, self._read_values)
      for value in values:
        current_key = int(value[self._key_fieldname])
        if current_key <= key:
          fieldnames = [x for x in value.keys() if x not in filtered_fieldnames]
          filtered_fieldnames.extend(fieldnames)
          filtered_values.append(value)
      await loop.run_in_executor(
          None, self._write_values, filtered_fieldnames, filtered_values
      )
      self._latest_key = key

  async def release(
      self,
      value: release_manager.ReleasableStructure,
      type_signature: computation_types.Type,
      key: int,
  ) -> None:
    """Releases `value` from a federated program.

    This method will atomically update the managed CSV file by removing all
    values previously released with a key greater than or equal to `key` before
    writing `value`.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: An integer used to reference the released `value`; `key` represents a
        step in a federated program.
    """
    del type_signature  # Unused.
    py_typecheck.check_type(key, (int, np.integer))

    _, materialized_value = await asyncio.gather(
        self._remove_values_greater_than(key - 1),
        value_reference.materialize_value(value),
    )

    flattened_value = structure_utils.flatten_with_name(materialized_value)

    def _normalize(
        value: value_reference.MaterializedValue,
    ) -> value_reference.MaterializedValue:
      if isinstance(value, tf.data.Dataset):
        value = list(value)
      return np.array(value).tolist()

    normalized_value = [(k, _normalize(v)) for k, v in flattened_value]
    normalized_value.insert(0, (self._key_fieldname, key))
    normalized_value = collections.OrderedDict(normalized_value)
    if self._save_mode == CSVSaveMode.APPEND:
      await self._append_value(normalized_value)
    elif self._save_mode == CSVSaveMode.WRITE:
      await self._write_value(normalized_value)
    elif self._save_mode == CSVSaveMode.READ:
      await self._read_values(normalized_value)
    self._latest_key = key


class SavedModelFileReleaseManager(
    release_manager.ReleaseManager[
        release_manager.ReleasableStructure, release_manager.Key
    ]
):
  """A `tff.program.ReleaseManager` that releases values to a file system.

  A `tff.program.SavedModelFileReleaseManager` is a utility for releasing values
  from a federated program to a file system and is used to release values from
  platform storage to customer storage in a federated program.

  Values are released to the file system using the SavedModel (see
  `tf.saved_model`) format. When the value is released, if the value is a value
  reference or a structure containing value references, each value reference is
  materialized. The value is then flattened and released using the SavedModel
  format. The structure of the value is discarded.

  Note: The SavedModel format can only contain values that can be converted to a
  `tf.Tensor` (see `tf.convert_to_tensor`), releasing any other values will
  result in an error.

  See https://www.tensorflow.org/guide/saved_model for more information about
  the SavedModel format.
  """

  def __init__(
      self, root_dir: Union[str, os.PathLike[str]], prefix: str = 'release_'
  ):
    """Returns an initialized `tff.program.SavedModelFileReleaseManager`.

    Args:
      root_dir: A path on the file system to save program state. If this path
        does not exist it will be created.
      prefix: A string to use as the prefix for filenames.

    Raises:
      ValueError: If `root_dir` is an empty string.
    """
    py_typecheck.check_type(root_dir, (str, os.PathLike))
    if not root_dir:
      raise ValueError('Expected `root_dir` to not be an empty string.')
    py_typecheck.check_type(prefix, str)

    if not tf.io.gfile.exists(root_dir):
      tf.io.gfile.makedirs(root_dir)
    self._root_dir = root_dir
    self._prefix = prefix

  def _get_path_for_key(self, key: release_manager.Key) -> str:
    """Returns the path for the given `key`.

    This method does not assert that the given `key` or the returned path
    represent released values.

    Args:
      key: The key used to construct the path.
    """
    basename = f'{self._prefix}{str(key)}'
    return os.path.join(self._root_dir, basename)

  async def release(
      self,
      value: release_manager.ReleasableStructure,
      type_signature: computation_types.Type,
      key: release_manager.Key,
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: Used to reference (in the file system) the released `value`.
    """
    del type_signature  # Unused.
    path = self._get_path_for_key(key)
    materialized_value = await value_reference.materialize_value(value)
    flattened_value = structure_utils.flatten(materialized_value)
    await file_utils.write_saved_model(flattened_value, path, overwrite=True)

  async def get_value(
      self,
      key: release_manager.Key,
      structure: release_manager.ReleasableStructure,
  ) -> release_manager.ReleasableStructure:
    """Returns the value for the given `key`.

    Args:
      key: Used to reference the released `value`.
      structure: The structure of the saved program state for the given `key`
        used to support serialization and deserialization of user-defined
        classes in the structure.

    Returns:
      A retrieved value matching `structure`.

    Raises:
      ReleasedValueNotFoundError: If there is no released value for the given
        `key`.
    """
    path = self._get_path_for_key(key)
    if not await file_utils.exists(path):
      raise release_manager.ReleasedValueNotFoundError(key)
    flattened_value = await file_utils.read_saved_model(path)
    value = structure_utils.unflatten_as(structure, flattened_value)

    def _normalize(
        value: release_manager.ReleasableValue,
    ) -> release_manager.ReleasableValue:
      """Returns a normalized value.

      The `tff.program.SavedModelFileReleaseManager` releases and gets values
      to/from file system using the SavedModel format. When the values are first
      loaded, they will be TF-native types. This function normalizes those
      values as numpy values so that, when returned, they can be used more
      naturally.

      Args:
        value: The value to normalize.
      """
      if isinstance(value, tf.Tensor):
        return value.numpy()
      else:
        return value

    normalized_value = structure_utils.map_structure(_normalize, value)
    return normalized_value
