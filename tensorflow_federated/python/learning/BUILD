load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":learning_packages",
    ":learning_visibility",
])

package_group(
    name = "learning_packages",
    packages = [
        "//tensorflow_federated/experimental/python/learning/...",
        "//tensorflow_federated/python/learning/...",
    ],
)

package_group(
    name = "learning_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_packages"],
)

licenses(["notice"])

py_library(
    name = "learning",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":client_weight_lib",
        ":debug_measurements",
        ":federated_averaging",
        ":federated_evaluation",
        ":keras_utils",
        ":model",
        ":model_update_aggregator",
        ":model_utils",
        ":personalization_eval",
        "//tensorflow_federated/python/learning/algorithms",
        "//tensorflow_federated/python/learning/framework",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/learning/metrics",
        "//tensorflow_federated/python/learning/models",
        "//tensorflow_federated/python/learning/optimizers",
        "//tensorflow_federated/python/learning/reconstruction",
        "//tensorflow_federated/python/learning/templates",
    ],
)

py_library(
    name = "client_weight_lib",
    srcs = ["client_weight_lib.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_library(
    name = "debug_measurements",
    srcs = ["debug_measurements.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:measurements",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "debug_measurements_test",
    srcs = ["debug_measurements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debug_measurements",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "federated_averaging",
    srcs = ["federated_averaging.py"],
    srcs_version = "PY3",
    deps = [
        ":client_weight_lib",
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/learning/optimizers:keras_optimizer",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "federated_averaging_test",
    timeout = "long",
    srcs = ["federated_averaging_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":client_weight_lib",
        ":federated_averaging",
        ":model_examples",
        ":model_update_aggregator",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/test:static_assert",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "federated_evaluation",
    srcs = ["federated_evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "federated_evaluation_test",
    srcs = ["federated_evaluation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_evaluation",
        ":keras_utils",
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/core/test:static_assert",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/framework:encoding_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/learning/metrics:counters",
        "//tensorflow_federated/python/learning/metrics:finalizer",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "medium",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:counters",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_examples",
    srcs = ["model_examples.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_examples_test",
    size = "small",
    srcs = ["model_examples_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_examples",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_update_aggregator",
    srcs = ["model_update_aggregator.py"],
    srcs_version = "PY3",
    deps = [
        ":debug_measurements",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:encoded",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/aggregators:robust",
        "//tensorflow_federated/python/aggregators:secure",
    ],
)

py_test(
    name = "model_update_aggregator_test",
    timeout = "long",
    srcs = ["model_update_aggregator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_update_aggregator",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/mapreduce:form_utils",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/test:static_assert",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "personalization_eval",
    srcs = ["personalization_eval.py"],
    srcs_version = "PY3",
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/aggregators:sampling",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "personalization_eval_test",
    srcs = ["personalization_eval_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        ":personalization_eval",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
