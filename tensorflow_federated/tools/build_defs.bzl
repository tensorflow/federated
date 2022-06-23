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
"""TensorFlow Federated build macros and rules."""

load("@rules_python//python:defs.bzl", "py_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

def if_static(extra, framework_shared_object = []):  # buildifier: disable=unused-variable
    return_value = {
        str(Label("@org_tensorflow//tensorflow:framework_shared_object")): framework_shared_object,
        "//conditions:default": extra,
    }
    return select(return_value)

def py_cpu_gpu_test(name, main = None, tags = [], **kwargs):
    """A version of `py_test` that tests both cpu and gpu.

    It accepts all `py_test` arguments.

    Args:
      name: A unique name for this target.
      main: The name of the source file that is the main entry point of
          the application
      tags: List of arbitrary text tags.
      **kwargs: `py_test` keyword arguments.
    """
    if main == None:
        main = name + ".py"
    py_test(
        name = name + "_cpu",
        main = main,
        tags = tags,
        **kwargs
    )
    py_test(
        name = name + "_gpu",
        main = main,
        tags = tags + ["requires-gpu-nvidia"],
        **kwargs
    )
    native.test_suite(
        name = name,
        tests = [
            name + "_cpu",
            name + "_gpu",
        ],
    )

def tff_cc_binary_with_tf_deps(name, tf_deps = [], **kwargs):
    """A version of `cc_binary` that links against TF statically or dynamically.

    Args:
      name: A unique name for this target.
      tf_deps: List of TensorFlow static dependencies.
      **kwargs: `cc_test` keyword arguments.
    """
    srcs = kwargs.pop("srcs", [])
    deps = kwargs.pop("deps", [])
    native.cc_binary(
        name = name,
        srcs = srcs + if_static(
            [],
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2",
            ],
        ),
        deps = deps + if_static(
            tf_deps,
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2.10.0",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2.10.0",
            ],
        ),
        **kwargs
    )

def tff_cc_test_with_tf_deps(name, tf_deps = [], **kwargs):
    """A version of `cc_test` that links against TF statically or dynamically.

    Args:
      name: A unique name for this target.
      tf_deps: List of TensorFlow static dependencies.
      **kwargs: `cc_test` keyword arguments.
    """
    srcs = kwargs.pop("srcs", [])
    deps = kwargs.pop("deps", [])
    native.cc_test(
        name = name,
        srcs = srcs + if_static(
            [],
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2",
            ],
        ),
        deps = deps + if_static(
            tf_deps,
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2.10.0",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2.10.0",
            ],
        ),
        **kwargs
    )

def tff_cc_library_with_tf_deps(name, tf_deps = [], **kwargs):
    """A version of `cc_library` that links against TF statically or dynamically.

    Args:
      name: A unique name for this target.
      tf_deps: List of TensorFlow static dependencies.
      **kwargs: `cc_test` keyword arguments.
    """
    deps = kwargs.pop("deps", [])
    native.cc_library(
        name = name,
        features = [
            "-use_header_modules",  # Required for pybind11.
            "-parse_headers",
        ],
        copts = ["-fexceptions"],
        deps = deps + if_static(
            tf_deps,
            framework_shared_object = ["@org_tensorflow//tensorflow:libtensorflow_framework.so.2.10.0"],
        ) + [
            "@pybind11",
            "@local_config_python//:python_headers",
        ],
        **kwargs
    )

def tff_cc_library_with_tf_runtime_deps(name, tf_deps = [], **kwargs):
    """A version of `cc_library` that links against TF statically or dynamically.

    Note that targets of this type will not work with pybind, due to conflicting
    symbols.

    Args:
      name: A unique name for this target.
      tf_deps: List of TensorFlow static dependencies.
      **kwargs: `cc_test` keyword arguments.
    """

    # TODO(b/209816646): This target will not work with pybind, but is
    # currently necessary to build dataset_conversions.cc in OSS.
    srcs = kwargs.pop("srcs", [])
    deps = kwargs.pop("deps", [])
    native.cc_library(
        name = name,
        srcs = srcs + if_static(
            [],
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2",
            ],
        ),
        deps = deps + if_static(
            tf_deps,
            framework_shared_object = [
                "@org_tensorflow//tensorflow:libtensorflow_framework.so.2.10.0",
                "@org_tensorflow//tensorflow:libtensorflow_cc.so.2.10.0",
            ],
        ),
        **kwargs
    )

def tff_pybind_extension_with_tf_deps(name, tf_deps = [], tf_python_dependency = False, **kwargs):
    """A version of `pybind_extension` that links against TF statically or dynamically.

    Args:
      name: A unique name for this target.
      tf_deps: List of TensorFlow static dependencies.
      tf_python_dependency: Wether this binding needs to depend on the TensorFlow Python bindings.
      **kwargs: `cc_test` keyword arguments.
    """
    deps = kwargs.pop("deps", [])
    srcs = kwargs.pop("srcs", [])
    if tf_python_dependency:
        extra_tf_dyn_srcs = select({
            "@org_tensorflow//tensorflow:framework_shared_object": ["@org_tensorflow//tensorflow/python:lib_pywrap_tensorflow_internal.so"],
            "//conditions:default": [],
        })
        extra_tf_dyn_deps = select({
            "@org_tensorflow//tensorflow:framework_shared_object": ["@org_tensorflow//tensorflow/python:_pywrap_tensorflow_internal.so"],
            "//conditions:default": [],
        })
    else:
        extra_tf_dyn_srcs = []
        extra_tf_dyn_deps = []
    pybind_extension(
        name = name,
        srcs = srcs + if_static(
            [],
            framework_shared_object = ["@org_tensorflow//tensorflow:libtensorflow_framework.so.2"],
        ) + extra_tf_dyn_srcs,
        deps = deps + if_static(
            tf_deps,
            framework_shared_object = ["@org_tensorflow//tensorflow:libtensorflow_framework.so.2.10.0"],
        ) + extra_tf_dyn_deps,
        **kwargs
    )

def tff_cc_cpu_gpu_test_with_tf_deps(name, tags = [], tf_deps = [], **kwargs):
    """A version of `cc_test` that tests both cpu and gpu.

    It accepts all `cc_test` arguments.

    Args:
      name: A unique name for this target.
      tags: List of arbitrary text tags.
      tf_deps: List of TensorFlow static dependencies.
      **kwargs: `cc_test` keyword arguments.
    """
    tff_cc_test_with_tf_deps(
        name = name + "_cpu",
        tags = tags,
        tf_deps = tf_deps,
        **kwargs
    )
    tff_cc_test_with_tf_deps(
        name = name + "_gpu",
        tags = tags + ["requires-gpu-nvidia"],
        tf_deps = tf_deps,
        **kwargs
    )
    native.test_suite(
        name = name,
        tests = [
            name + "_cpu",
            name + "_gpu",
        ],
    )
