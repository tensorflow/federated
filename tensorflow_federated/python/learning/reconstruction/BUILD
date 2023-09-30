load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
)

py_library(
    name = "reconstruction_utils",
    srcs = ["reconstruction_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "reconstruction_utils_test",
    srcs = ["reconstruction_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_utils",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:keras_utils",
    ],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model",
    ],
)
