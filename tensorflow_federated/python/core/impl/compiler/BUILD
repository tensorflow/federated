load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":compiler_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/computation:computation_packages",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
        "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    ],
)

package_group(
    name = "compiler_packages",
    packages = ["//tensorflow_federated/python/core/impl/compiler/..."],
)

licenses(["notice"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "building_block_analysis",
    srcs = ["building_block_analysis.py"],
    deps = [":building_blocks"],
)

py_library(
    name = "building_block_factory",
    srcs = ["building_block_factory.py"],
    deps = [
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
    ],
)

py_test(
    name = "building_block_factory_test",
    size = "large",
    srcs = ["building_block_factory_test.py"],
    args = [
        "--golden",
        "$(location building_block_factory_test_goldens/constructs_correct_computation_clients.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/constructs_correct_computation_server.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/federated_secure_modular_sum.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/replaces_single_element.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/skips_unnamed_element.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_different_typed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_different_typed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_tuple_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_tuple_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_reference.expected)",
    ],
    data = [
        "building_block_factory_test_goldens/constructs_correct_computation_clients.expected",
        "building_block_factory_test_goldens/constructs_correct_computation_server.expected",
        "building_block_factory_test_goldens/federated_secure_modular_sum.expected",
        "building_block_factory_test_goldens/replaces_single_element.expected",
        "building_block_factory_test_goldens/skips_unnamed_element.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_different_typed.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_named.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_unnamed.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_different_typed.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_named.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_unnamed.expected",
        "building_block_factory_test_goldens/zips_reference.expected",
        "building_block_factory_test_goldens/zips_tuple_named.expected",
        "building_block_factory_test_goldens/zips_tuple_unnamed.expected",
    ],
    deps = [
        ":building_block_factory",
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "building_block_test_utils",
    testonly = True,
    srcs = ["building_block_test_utils.py"],
    deps = [
        ":building_block_factory",
        ":building_blocks",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "building_blocks",
    srcs = ["building_blocks.py"],
    deps = [
        ":intrinsic_defs",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "building_blocks_test",
    size = "small",
    srcs = ["building_blocks_test.py"],
    deps = [
        ":building_block_test_utils",
        ":building_blocks",
        ":computation_factory",
        ":intrinsic_defs",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "computation_factory",
    srcs = ["computation_factory.py"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "computation_factory_test",
    srcs = ["computation_factory_test.py"],
    deps = [
        ":computation_factory",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_factory",
    ],
)

py_test(
    name = "intrinsic_defs_test",
    size = "small",
    srcs = ["intrinsic_defs_test.py"],
    deps = [":intrinsic_defs"],
)

py_library(
    name = "local_computation_factory_base",
    srcs = ["local_computation_factory_base.py"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "tensorflow_computation_factory",
    srcs = ["tensorflow_computation_factory.py"],
    deps = [
        ":local_computation_factory_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:array_shape",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
    ],
)

py_cpu_gpu_test(
    name = "tensorflow_computation_factory_test",
    srcs = ["tensorflow_computation_factory_test.py"],
    deps = [
        ":tensorflow_computation_factory",
        ":tensorflow_computation_test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "tensorflow_computation_test_utils",
    testonly = True,
    srcs = ["tensorflow_computation_test_utils.py"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "transformations_test",
    srcs = ["transformations_test.py"],
    deps = [
        ":building_block_factory",
        ":building_block_test_utils",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        ":transformations",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "transformation_utils",
    srcs = ["transformation_utils.py"],
    deps = [
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "transformation_utils_test",
    size = "small",
    srcs = ["transformation_utils_test.py"],
    deps = [
        ":building_block_test_utils",
        ":building_blocks",
        ":computation_factory",
        ":transformation_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "tree_analysis",
    srcs = ["tree_analysis.py"],
    deps = [
        ":building_block_analysis",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "tree_analysis_test",
    srcs = ["tree_analysis_test.py"],
    deps = [
        ":building_block_factory",
        ":building_block_test_utils",
        ":building_blocks",
        ":intrinsic_defs",
        ":tree_analysis",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "tree_transformations",
    srcs = ["tree_transformations.py"],
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        ":tree_analysis",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
    ],
)

py_test(
    name = "tree_transformations_test",
    size = "small",
    srcs = ["tree_transformations_test.py"],
    args = [
        "--golden",
        "$(location tree_transformations_test_goldens/uniquify_names_blocks_nested_inside_of_locals.expected)",
    ],
    data = [
        "tree_transformations_test_goldens/uniquify_names_blocks_nested_inside_of_locals.expected",
    ],
    deps = [
        ":building_block_factory",
        ":building_block_test_utils",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "compiler_test_utils",
    testonly = True,
    srcs = ["compiler_test_utils.py"],
    deps = [
        ":building_blocks",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "compiler_test_utils_test",
    size = "small",
    srcs = ["compiler_test_utils_test.py"],
    args = [
        "--golden",
        "$(location compiler_test_utils_test_goldens/test_check_computations.expected)",
    ],
    data = [
        "compiler_test_utils_test_goldens/test_check_computations.expected",
    ],
    deps = [
        ":building_block_test_utils",
        ":compiler_test_utils",
        "//tensorflow_federated/python/common_libs:golden",
    ],
)
