load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":mapreduce_packages",
        "//tensorflow_federated/python/core:core_users",

        # TODO: b/233800075 - Remove dependency from `native` to `mapreduce`.
        "//tensorflow_federated/python/core/backends/native:native_packages",
    ],
)

package_group(
    name = "mapreduce_packages",
    packages = ["//tensorflow_federated/python/core/backends/mapreduce/..."],
)

licenses(["notice"])

py_library(
    name = "mapreduce",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [
        ":compiler",
        ":form_utils",
        ":forms",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:compiled_computation_transformations",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    deps = [
        ":compiler",
        ":form_utils",
        ":mapreduce_test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_block_test_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "forms",
    srcs = ["forms.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "forms_test",
    srcs = ["forms_test.py"],
    deps = [
        ":distribute_aggregate_test_utils",
        ":forms",
        ":mapreduce_test_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "form_utils",
    srcs = ["form_utils.py"],
    deps = [
        ":compiler",
        ":forms",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_building_block_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "form_utils_test",
    size = "large",
    srcs = ["form_utils_test.py"],
    deps = [
        ":distribute_aggregate_test_utils",
        ":form_utils",
        ":forms",
        ":mapreduce_test_utils",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
    ],
)

py_library(
    name = "distribute_aggregate_test_utils",
    testonly = True,
    srcs = ["distribute_aggregate_test_utils.py"],
    deps = [
        ":forms",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "mapreduce_test_utils",
    testonly = True,
    srcs = ["mapreduce_test_utils.py"],
    deps = [
        ":forms",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)
