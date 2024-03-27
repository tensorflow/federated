load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":tests_packages"],
)

package_group(
    name = "tests_packages",
    packages = ["//tensorflow_federated/python/tests/..."],
)

licenses(["notice"])

py_test(
    name = "ast_generation_test",
    size = "small",
    srcs = ["ast_generation_test.py"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "async_execution_context_integration_test",
    srcs = ["async_execution_context_integration_test.py"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "backend_test",
    timeout = "long",
    srcs = ["backend_test.py"],
    deps = [
        ":temperature_sensor_example",
        ":test_contexts",
        "//tensorflow_federated",
    ],
)

py_cpu_gpu_test(
    name = "backend_accelerators_test",
    srcs = ["backend_accelerators_test.py"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "map_reduce_form_test",
    size = "medium",
    srcs = ["map_reduce_form_test.py"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "mergeable_comp_execution_context_integration_test",
    size = "small",
    timeout = "long",
    srcs = ["mergeable_comp_execution_context_integration_test.py"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "simulation_test",
    size = "medium",
    timeout = "long",
    srcs = ["simulation_test.py"],
    shard_count = 4,
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "remote_runtime_stream_structs_test",
    size = "large",
    timeout = "moderate",
    srcs = ["remote_runtime_stream_structs_test.py"],
    tags = ["requires-mem:20g"],
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "sync_local_cpp_execution_context_test",
    size = "small",
    srcs = ["sync_local_cpp_execution_context_test.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "temperature_sensor_example",
    srcs = ["temperature_sensor_example.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "test_contexts",
    testonly = True,
    srcs = ["test_contexts.py"],
    data = ["//tensorflow_federated/cc/simulation:worker_binary"],
    deps = ["//tensorflow_federated"],
)
