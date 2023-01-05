load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":utils_packages",
    "//tensorflow_federated/python/core/impl:impl_users",
    "//tensorflow_federated/python/core/impl/compiler:compiler_packages",
    "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
    "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_context_packages",
])

package_group(
    name = "utils_packages",
    packages = ["//tensorflow_federated/python/core/impl/utils/..."],
)

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:named_containers",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/tensorflow_libs:graph_utils",
    ],
)

py_test(
    name = "tensorflow_utils_test",
    size = "small",
    srcs = ["tensorflow_utils_test.py"],
    args = [
        "--golden",
        "$(location tensorflow_utils_test_goldens/capture_result_with_sequence_of_dicts.expected)",
    ],
    data = [
        "tensorflow_utils_test_goldens/capture_result_with_sequence_of_dicts.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
    ],
)
