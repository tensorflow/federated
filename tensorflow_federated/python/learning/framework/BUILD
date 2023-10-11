load("@rules_python//python:defs.bzl", "py_library")
load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":framework_packages",
        "//tensorflow_federated/python/learning:learning_users",
    ],
)

package_group(
    name = "framework_packages",
    packages = ["//tensorflow_federated/python/learning/framework/..."],
)

licenses(["notice"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [":optimizer_utils"],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
    ],
)

py_cpu_gpu_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    shard_count = 10,
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
    ],
)
