load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":executors_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    ],
)

package_group(
    name = "executors_packages",
    packages = ["//tensorflow_federated/python/core/impl/executors/..."],
)

licenses(["notice"])

py_library(
    name = "executors",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "cardinalities_utils",
    srcs = ["cardinalities_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":cardinality_carrying_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "cardinalities_utils_test",
    srcs = ["cardinalities_utils_test.py"],
    deps = [
        ":cardinalities_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "cardinality_carrying_base",
    srcs = ["cardinality_carrying_base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "cpp_to_python_executor",
    srcs = ["cpp_to_python_executor.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_base",
        ":executor_bindings",
        ":executor_value_base",
        ":executors_errors",
        ":value_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "cpp_to_python_executor_test",
    srcs = ["cpp_to_python_executor_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_to_python_executor",
        ":executor_bindings",
        ":value_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "data_conversions",
    srcs = ["data_conversions.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:placements"],
)

py_test(
    name = "data_conversions_test",
    srcs = ["data_conversions_test.py"],
    deps = [
        ":data_conversions",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "data_descriptor",
    srcs = ["data_descriptor.py"],
    srcs_version = "PY3",
    deps = [
        ":cardinality_carrying_base",
        ":ingestable_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "data_descriptor_test",
    srcs = ["data_descriptor_test.py"],
    deps = [
        ":data_descriptor",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    srcs_version = "PY3",
    deps = [":executor_value_base"],
)

py_library(
    name = "executor_bindings",
    srcs = ["executor_bindings.py"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":data_conversions",
        "//tensorflow_federated/cc/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "executor_bindings_test",
    timeout = "moderate",
    srcs = ["executor_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_bindings",
        ":executor_test_utils_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_test_utils_bindings",
    testonly = True,
    srcs = ["executor_test_utils_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = ["//tensorflow_federated/cc/core/impl/executors:executor_test_utils_bindings"],
)

py_library(
    name = "executor_utils",
    srcs = ["executor_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "executor_utils_test",
    srcs = ["executor_utils_test.py"],
    deps = [
        ":executor_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_library(
    name = "executors_errors",
    srcs = ["executors_errors.py"],
    srcs_version = "PY3",
)

py_library(
    name = "ingestable_base",
    srcs = ["ingestable_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        ":executors_errors",
        ":remote_executor_stub",
        ":value_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "remote_executor_test",
    size = "small",
    srcs = ["remote_executor_test.py"],
    deps = [
        ":remote_executor",
        ":remote_executor_stub",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:computation_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "remote_executor_stub",
    srcs = ["remote_executor_stub.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/proto/v0:executor_py_pb2"],
)

py_library(
    name = "remote_executor_grpc_stub",
    srcs = ["remote_executor_grpc_stub.py"],
    srcs_version = "PY3",
    deps = [
        ":executors_errors",
        ":remote_executor_stub",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2_grpc",
        "//tensorflow_federated/python/common_libs:tracing",
    ],
)

py_test(
    name = "remote_executor_grpc_stub_test",
    srcs = ["remote_executor_grpc_stub_test.py"],
    deps = [
        ":executors_errors",
        ":remote_executor_grpc_stub",
        ":value_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2_grpc",
        "//tensorflow_federated/python/core/impl/compiler:computation_factory",
    ],
)

py_library(
    name = "value_serialization",
    srcs = ["value_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:array",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:array_shape",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "value_serialization_test",
    size = "small",
    srcs = ["value_serialization_test.py"],
    deps = [
        ":value_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:computation_factory",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)
