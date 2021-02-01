load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Packages that have visibility into the `learning` package.
package_group(name = "learning_visibility")

package_group(
    name = "default_visibility",
    includes = [":learning_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/learning/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "learning",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":federated_averaging",
        ":federated_evaluation",
        ":federated_sgd",
        ":keras_utils",
        ":model",
        ":model_update_aggregator",
        ":model_utils",
        ":personalization_eval",
        "//tensorflow_federated/python/learning/framework",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
    ],
)

py_library(
    name = "federated_averaging",
    srcs = ["federated_averaging.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_cpu_gpu_test(
    name = "federated_averaging_test",
    size = "large",
    srcs = ["federated_averaging_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_averaging",
        ":keras_utils",
        ":model_examples",
        ":model_update_aggregator",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
    ],
)

py_library(
    name = "federated_evaluation",
    srcs = ["federated_evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/utils:tf_computation_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
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
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
    ],
)

py_library(
    name = "federated_sgd",
    srcs = ["federated_sgd.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_cpu_gpu_test(
    name = "federated_sgd_test",
    size = "large",
    srcs = ["federated_sgd_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_sgd",
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
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
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
)

py_library(
    name = "model_examples",
    srcs = ["model_examples.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
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
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_library(
    name = "model_update_aggregator",
    srcs = ["model_update_aggregator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:dp_factory",
        "//tensorflow_federated/python/aggregators:encoded_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean_factory",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/aggregators:robust_factory",
        "//tensorflow_federated/python/aggregators:secure_factory",
    ],
)

py_test(
    name = "model_update_aggregator_test",
    srcs = ["model_update_aggregator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_update_aggregator",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

py_library(
    name = "personalization_eval",
    srcs = ["personalization_eval.py"],
    srcs_version = "PY3",
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/utils:federated_aggregations",
        "//tensorflow_federated/python/core/utils:tf_computation_utils",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_cpu_gpu_test(
    name = "personalization_eval_test",
    srcs = ["personalization_eval_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    # TODO(b/160896627): Kokoro GPU tests provide multi-GPU environment by default,
    # bring back the Kokoro tests after we fix the dataset.reduce issue
    tags = ["nokokoro"],
    deps = [
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        ":personalization_eval",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
    ],
)
