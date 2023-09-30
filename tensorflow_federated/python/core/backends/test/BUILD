load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":test_packages",
        "//tensorflow_federated/python/core:core_users",

        # TODO: b/233801300 - Remove dependency from `mapreduce` to `test`.
        "//tensorflow_federated/python/core/backends/mapreduce:mapreduce_packages",
        "//tensorflow_federated/python/core/backends/native:native_packages",
    ],
)

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/core/backends/test/..."],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [
        ":compiler",
        ":execution_contexts",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
    ],
)

py_library(
    name = "cpp_execution_contexts",
    srcs = ["cpp_execution_contexts.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/backends/native:compiler",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
    ],
)

py_test(
    name = "cpp_execution_contexts_test",
    srcs = ["cpp_execution_contexts_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_execution_contexts",
        ":execution_contexts",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "execution_contexts",
    srcs = ["execution_contexts.py"],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/backends/native:compiler",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_test(
    name = "execution_contexts_test",
    srcs = ["execution_contexts_test.py"],
    deps = [
        ":execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
