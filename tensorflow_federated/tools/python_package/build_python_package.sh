#!/usr/bin/env bash
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
# Tool to build the TensorFlow Federated pip package.
set -e

usage() {
  local script_name=$(basename "${0}")
  local options=(
      "--python=python3.11"
      "--output_dir=<path>"
  )
  echo "usage: ${script_name} ${options[@]}"
  echo "  --python=python3.11   The Python version used by the environment to"
  echo "                        build the Python package."
  echo "  --output_dir=<path>   An output directory."
  exit 1
}

main() {
  # Parse the arguments.
  local python="python3.11"
  local output_dir=""

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --python=*)
        python="${option#*=}"
        shift
        ;;
      --output_dir=*)
        output_dir="${option#*=}"
        shift
        ;;
      *)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        ;;
    esac
  done

  if [[ -z "${output_dir}" ]]; then
    echo "error: required option `--output_dir`" 1>&2
    usage
  elif [[ ! -d "${output_dir}" ]]; then
    echo "error: the directory '${output_dir}' does not exist" 1>&2
    usage
  fi

  # Create a working directory.
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  cp -LR "." "${temp_dir}"
  pushd "${temp_dir}"

  # Create a Python environment.
  "${python}" -m venv "venv"
  source "venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version
  # The `plat-name` manylinux tag should match GLIBC version returned by
  # `ldd --version`.
  ldd --version

  # Build the Python package.
  pip install --upgrade build
  python -m build --wheel
  cp "${temp_dir}/dist/"* "${output_dir}"

  # Check Python package sizes.
  local package="$(ls "${output_dir}/tensorflow_federated-"*".whl" | head -n1)"
  local actual_size="$(du -b "${package}" | cut -f1)"
  local maximum_size=80000000  # 80 MiB
  if [ "${actual_size}" -ge "${maximum_size}" ]; then
    echo "Error: expected $(basename ${package}) to be less than ${maximum_size} bytes; it was ${actual_size}." 1>&2
    exit 1
  fi

  # Cleanup.
  deactivate
  popd
}

main "$@"
