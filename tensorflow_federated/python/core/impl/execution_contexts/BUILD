load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":execution_contexts_packages",
    "//tensorflow_federated/python/core/impl:impl_users",
])

package_group(
    name = "execution_contexts_packages",
    packages = ["//tensorflow_federated/python/core/impl/execution_contexts/..."],
)

licenses(["notice"])

py_library(
    name = "execution_contexts",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "async_execution_context",
    srcs = ["async_execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:retrying",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/executors:ingestable_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "async_execution_context_test",
    size = "small",
    srcs = ["async_execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":async_execution_context",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/executor_stacks:python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
    ],
)

py_test(
    name = "compiler_pipeline_test",
    size = "small",
    srcs = ["compiler_pipeline_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
    ],
)

py_library(
    name = "cpp_async_execution_context",
    srcs = ["cpp_async_execution_context.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/common_libs:retrying",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:value_serialization",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_library(
    name = "cpp_sync_execution_context",
    srcs = ["cpp_sync_execution_context.py"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_async_execution_context",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
    ],
)

py_library(
    name = "sync_execution_context",
    srcs = ["sync_execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":async_execution_context",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_test(
    name = "sync_execution_context_test",
    size = "small",
    srcs = ["sync_execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sync_execution_context",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/executor_stacks:python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "mergeable_comp_execution_context",
    srcs = ["mergeable_comp_execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "mergeable_comp_execution_context_test",
    size = "small",
    timeout = "moderate",
    srcs = ["mergeable_comp_execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":async_execution_context",
        ":mergeable_comp_execution_context",
        ":sync_execution_context",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/executor_stacks:python_executor_stacks",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)
