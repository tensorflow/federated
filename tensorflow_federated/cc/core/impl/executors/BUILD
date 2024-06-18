load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary")
load("//tools:build_defs.bzl", "tff_cc_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":executors_packages",
        "//tensorflow_federated/cc/core/impl:impl_users",
        "//tensorflow_federated/cc/core/impl/executor_stacks:executor_stacks_packages",
    ],
)

package_group(
    name = "executors_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/executors/...",
    ],
)

licenses(["notice"])

cc_library(
    name = "array_shape_test_utils",
    testonly = True,
    hdrs = ["array_shape_test_utils.h"],
    deps = ["//tensorflow_federated/proto/v0:array_cc_proto"],
)

cc_library(
    name = "array_test_utils",
    testonly = True,
    hdrs = ["array_test_utils.h"],
    deps = [
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "cardinalities",
    srcs = ["cardinalities.cc"],
    hdrs = ["cardinalities.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "composing_executor",
    srcs = ["composing_executor.cc"],
    hdrs = ["composing_executor.h"],
    deps = [
        ":cardinalities",
        ":computations",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core/platform:macros",
    ],
)

cc_test(
    name = "composing_executor_test",
    srcs = ["composing_executor_test.cc"],
    deps = [
        ":cardinalities",
        ":composing_executor",
        ":computations",
        ":executor",
        ":executor_test_base",
        ":mock_executor",
        ":type_utils",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "computations",
    hdrs = ["computations.h"],
    deps = ["//tensorflow_federated/proto/v0:computation_cc_proto"],
)

cc_library(
    name = "data_backend",
    hdrs = ["data_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "data_executor",
    srcs = ["data_executor.cc"],
    hdrs = ["data_executor.h"],
    deps = [
        ":data_backend",
        ":executor",
        ":status_macros",
        ":threading",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "data_executor_test",
    srcs = ["data_executor_test.cc"],
    deps = [
        ":data_executor",
        ":executor",
        ":executor_test_base",
        ":mock_data_backend",
        ":mock_executor",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_library(
    name = "dataset_conversions",
    srcs = ["dataset_conversions.cc"],
    hdrs = ["dataset_conversions.h"],
    deps = [
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core/data:standalone",
    ],
)

cc_test(
    name = "dataset_conversions_test",
    srcs = ["dataset_conversions_test.cc"],
    deps = [
        ":dataset_conversions",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/data:standalone",
    ],
)

cc_library(
    name = "dataset_from_tensor_structures",
    srcs = ["dataset_from_tensor_structures.cc"],
    hdrs = ["dataset_from_tensor_structures.h"],
    deps = [
        ":session_provider",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:graph",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "dataset_from_tensor_structures_test",
    srcs = ["dataset_from_tensor_structures_test.cc"],
    args = [
        "--reduce_graph_path",
        "$(location structural_reduce_test.pbtxt)",
    ],
    data = ["structural_reduce_test.pbtxt"],
    deps = [
        ":dataset_from_tensor_structures",
        ":session_provider",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "eager_computation",
    srcs = ["eager_computation.cc"],
    hdrs = ["eager_computation.h"],
    deps = [
        ":status_macros",
        ":tensorflow_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/c:tf_datatype",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/c/eager:tfe_tensorhandle_internal",
        "@org_tensorflow//tensorflow/compiler/mlir/python:mlir",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/common_runtime:core",
    ],
)

tff_cc_cpu_gpu_test(
    name = "eager_computation_test",
    srcs = ["eager_computation_test.cc"],
    deps = [
        ":eager_computation",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/c:tf_datatype",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c:tf_tensor",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/c/eager:c_api_internal",
        "@org_tensorflow//tensorflow/c/eager:immediate_execution_tensor_handle",
        "@org_tensorflow//tensorflow/c/eager:tfe_tensorhandle_internal",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:functional_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

pybind_extension(
    name = "executor_bindings",
    srcs = ["executor_bindings.cc"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cardinalities",
        ":composing_executor",
        ":executor",
        ":federating_executor",
        ":reference_resolving_executor",
        ":remote_executor",
        ":sequence_executor",
        ":status_macros",
        ":streaming_remote_executor",
        ":tensor_serialization",
        ":tensorflow_executor",
        ":xla_executor",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@org_tensorflow//tensorflow/core/platform:strcat",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "executor_service",
    srcs = ["executor_service.cc"],
    hdrs = ["executor_service.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_conversion",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "executor_service_test",
    srcs = ["executor_service_test.cc"],
    deps = [
        ":cardinalities",
        ":executor",
        ":executor_service",
        ":mock_executor",
        ":status_conversion",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executor_test_base",
    testonly = True,
    srcs = [],
    hdrs = ["executor_test_base.h"],
    deps = [
        ":executor",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

pybind_extension(
    name = "executor_test_utils_bindings",
    testonly = True,
    srcs = ["executor_test_utils_bindings.cc"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [":mock_executor"],
)

cc_library(
    name = "federated_intrinsics",
    srcs = ["federated_intrinsics.cc"],
    hdrs = ["federated_intrinsics.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "federating_executor",
    srcs = ["federating_executor.cc"],
    hdrs = ["federating_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":tensor_serialization",
        ":threading",
        ":value_validation",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:macros",
    ],
)

cc_test(
    name = "federating_executor_test",
    srcs = ["federating_executor_test.cc"],
    deps = [
        ":executor",
        ":executor_test_base",
        ":federating_executor",
        ":mock_executor",
        ":status_macros",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

genrule(
    name = "reduce_lambda_test_graph",
    testonly = True,
    srcs = [],
    outs = ["reduce_lambda_test.pbtxt"],
    cmd = "$(location :make_reduce_lambda_test_graph) --output \"$@\"",
    tools = [":make_reduce_lambda_test_graph"],
)

# Python target which generates an example GraphDef for testing `reduce`
# in `tensorflow_executor_test`.
py_binary(
    name = "make_reduce_lambda_test_graph",
    testonly = True,
    srcs = ["make_reduce_lambda_test_graph.py"],
)

genrule(
    name = "structural_reduce_test_graph",
    testonly = True,
    srcs = [],
    outs = ["structural_reduce_test.pbtxt"],
    cmd = "$(location :make_structural_reduce_test_graph) --output \"$@\"",
    tools = [":make_structural_reduce_test_graph"],
)

# Python target which generates an example GraphDef for testing structural
# `reduce` in `dataset_from_tensor_structures_test`.
py_binary(
    name = "make_structural_reduce_test_graph",
    testonly = True,
    srcs = ["make_structural_reduce_test_graph.py"],
)

cc_library(
    name = "mock_data_backend",
    testonly = True,
    hdrs = ["mock_data_backend.h"],
    deps = [
        ":data_backend",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_executor",
    testonly = True,
    srcs = ["mock_executor.cc"],
    hdrs = ["mock_executor.h"],
    deps = [
        ":executor",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_grpc",
    testonly = True,
    hdrs = ["mock_grpc.h"],
    deps = [
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.cc"],
    hdrs = ["reference_resolving_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":executor",
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "reference_resolving_executor_test",
    timeout = "short",
    srcs = ["reference_resolving_executor_test.cc"],
    deps = [
        ":executor",
        ":executor_test_base",
        ":mock_executor",
        ":reference_resolving_executor",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:math_ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "remote_executor",
    srcs = ["remote_executor.cc"],
    hdrs = ["remote_executor.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_conversion",
        ":status_macros",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "remote_executor_test",
    timeout = "short",
    srcs = ["remote_executor_test.cc"],
    deps = [
        ":cardinalities",
        ":executor",
        ":mock_grpc",
        ":remote_executor",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:math_ops",
    ],
)

cc_library(
    name = "sequence_executor",
    srcs = ["sequence_executor.cc"],
    hdrs = ["sequence_executor.h"],
    deps = [
        ":dataset_conversions",
        ":executor",
        ":sequence_intrinsics",
        ":status_macros",
        ":struct_traversal_order",
        ":tensor_serialization",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/data:standalone",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_test(
    name = "sequence_executor_test",
    timeout = "short",
    srcs = ["sequence_executor_test.cc"],
    flaky = True,
    tags = [
        "nokokoro",  # TODO: b/340292788 - Temporarily disable flaky test.
    ],
    deps = [
        ":executor",
        ":executor_test_base",
        ":mock_executor",
        ":sequence_executor",
        ":sequence_intrinsics",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "sequence_intrinsics",
    srcs = ["sequence_intrinsics.cc"],
    hdrs = ["sequence_intrinsics.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "session_provider",
    srcs = ["session_provider.cc"],
    hdrs = ["session_provider.h"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/common_runtime:device_mgr",
        "@org_tensorflow//tensorflow/core/common_runtime:session",
        "@org_tensorflow//tensorflow/core/common_runtime:session_options",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_test(
    name = "session_provider_test",
    srcs = ["session_provider_test.cc"],
    deps = [
        ":session_provider",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "status_conversion",
    srcs = ["status_conversion.cc"],
    hdrs = ["status_conversion.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "streaming_remote_executor",
    srcs = ["streaming_remote_executor.cc"],
    hdrs = ["streaming_remote_executor.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":status_conversion",
        ":status_macros",
        ":threading",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "streaming_remote_executor_test",
    timeout = "short",
    srcs = ["streaming_remote_executor_test.cc"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":mock_grpc",
        ":streaming_remote_executor",
        ":type_utils",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:math_ops",
    ],
)

cc_library(
    name = "struct_traversal_order",
    hdrs = ["struct_traversal_order.h"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "struct_traversal_order_test",
    srcs = ["struct_traversal_order_test.cc"],
    deps = [
        ":struct_traversal_order",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tensor_serialization",
    srcs = ["tensor_serialization.cc"],
    hdrs = ["tensor_serialization.h"],
    deps = [
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tensorflow_executor",
    srcs = ["tensorflow_executor.cc"],
    hdrs = ["tensorflow_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_from_tensor_structures",
        ":executor",
        ":session_provider",
        ":status_macros",
        ":tensor_serialization",
        ":tensorflow_utils",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/common_runtime:core",
        "@org_tensorflow//tensorflow/core/common_runtime:session",
        "@org_tensorflow//tensorflow/core/platform:macros",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

tff_cc_cpu_gpu_test(
    name = "tensorflow_executor_parameterized_test",
    srcs = ["tensorflow_executor_parameterized_test.cc"],
    args = [
        "--reduce_graph_path",
        "$(location reduce_lambda_test.pbtxt)",
    ],
    data = ["reduce_lambda_test.pbtxt"],
    env = {"TF_XLA_FLAGS": "--tf_mlir_enable_mlir_bridge=true"},
    deps = [
        ":array_shape_test_utils",
        ":array_test_utils",
        ":executor",
        ":status_macros",
        ":tensorflow_executor",
        ":value_test_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/c/eager:tfe_context_internal",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.cc"],
    hdrs = ["tensorflow_utils.h"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensorflow_utils_test",
    srcs = ["tensorflow_utils_test.cc"],
    deps = [
        ":array_shape_test_utils",
        ":array_test_utils",
        ":tensorflow_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen3",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "threading",
    srcs = ["threading.cc"],
    hdrs = ["threading.h"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
    ],
)

cc_test(
    name = "threading_test",
    timeout = "short",
    srcs = ["threading_test.cc"],
    deps = [
        ":threading",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "type_test_utils",
    testonly = True,
    hdrs = ["type_test_utils.h"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "type_utils",
    srcs = ["type_utils.cc"],
    hdrs = ["type_utils.h"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "value_test_utils",
    testonly = True,
    hdrs = ["value_test_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cardinalities",
        ":dataset_conversions",
        ":status_macros",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:const_op",
        "@org_tensorflow//tensorflow/cc:dataset_ops_internal",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/data:standalone",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "value_validation",
    srcs = ["value_validation.cc"],
    hdrs = ["value_validation.h"],
    deps = [
        ":cardinalities",
        ":federated_intrinsics",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "value_validation_test",
    timeout = "short",
    srcs = ["value_validation_test.cc"],
    deps = [
        ":value_test_utils",
        ":value_validation",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_executor",
    srcs = ["xla_executor.cc"],
    hdrs = ["xla_executor.h"],
    deps = [
        ":executor",
        ":status_macros",
        ":tensor_serialization",
        ":threading",
        ":xla_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/client",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:global_data",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:multi_platform_manager",
        "@local_xla//xla/stream_executor/host:host_platform",  # buildcleaner: keep # Linking in the host platform here ensures that the stream executor can execute on CPU.
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",  # buildcleaner: keep # Linking in this dependency ensures that XLA can compile its code for the CPU host.
        "@org_tensorflow//tensorflow/compiler/tf2xla:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tff_cc_cpu_gpu_test(
    name = "xla_executor_test",
    srcs = ["xla_executor_test.cc"],
    args = ["--tff_xla_executor_test_platform=Host"],
    deps = [
        ":array_shape_test_utils",
        ":array_test_utils",
        ":executor",
        ":status_macros",
        ":type_test_utils",
        ":value_test_utils",
        ":xla_executor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/stream_executor",
        "@org_tensorflow//tensorflow/compiler/tf2xla:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:statusor",
    ],
)

cc_test(
    name = "xla_executor_gpu_test",
    srcs = ["xla_executor_test.cc"],
    args = ["--tff_xla_executor_test_platform=CUDA"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":array_shape_test_utils",
        ":array_test_utils",
        ":executor",
        ":status_macros",
        ":type_test_utils",
        ":value_test_utils",
        ":xla_executor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/cuda:cuda_platform",  # buildcleaner: keep # Linking in the host platform here ensures that the stream executor can execute on GPU.
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",  # buildcleaner: keep # Linking in this dependency ensures that XLA can compile its code for the GPU.
        "@org_tensorflow//tensorflow/compiler/tf2xla:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:statusor",
    ],
)

cc_library(
    name = "xla_utils",
    srcs = ["xla_utils.cc"],
    hdrs = ["xla_utils.h"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:types",
    ],
)

cc_test(
    name = "xla_utils_test",
    srcs = ["xla_utils_test.cc"],
    deps = [
        ":array_shape_test_utils",
        ":array_test_utils",
        ":xla_utils",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "//tensorflow_federated/proto/v0:array_cc_proto",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:data_type_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen3",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:types",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)
