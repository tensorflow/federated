load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":wrappers_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/compiler:compiler_packages",
])

package_group(
    name = "wrappers_packages",
    packages = [
        "//tensorflow_federated/experimental/python/core/impl/wrappers/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",
    ],
)

licenses(["notice"])

py_library(
    name = "wrappers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:function_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/tensorflow_libs:function",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation_utils",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_serialization",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    args = [
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/tensorflow_wrapper_traceback.expected)",
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/non_callable_non_type_traceback.expected)",
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/federated_computation_wrapper_traceback.expected)",
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/returns_type_container_mismatch_traceback.expected)",
    ],
    data = [
        "computation_wrapper_instances_test_goldens/federated_computation_wrapper_traceback.expected",
        "computation_wrapper_instances_test_goldens/non_callable_non_type_traceback.expected",
        "computation_wrapper_instances_test_goldens/returns_type_container_mismatch_traceback.expected",
        "computation_wrapper_instances_test_goldens/tensorflow_wrapper_traceback.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        ":computation_wrapper_instances",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/context_stack:runtime_error_context",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
