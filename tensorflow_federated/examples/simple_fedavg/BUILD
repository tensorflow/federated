load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    deps = [
        ":simple_fedavg_tff",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    deps = [
        ":simple_fedavg_tf",
        "//tensorflow_federated",
    ],
)

py_cpu_gpu_test(
    name = "simple_fedavg_test",
    size = "medium",
    srcs = ["simple_fedavg_test.py"],
    shard_count = 3,
    tags = ["nokokoro"],
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//tensorflow_federated",
    ],
)
