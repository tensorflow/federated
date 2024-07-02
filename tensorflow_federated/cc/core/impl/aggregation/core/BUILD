# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

# TODO: b/254719929 - Create a "core" library that bundles all core libraries together.

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

java_proto_library(
    name = "tensor_java_proto",
    visibility = ["//visibility:public"],
    deps = [":tensor_proto"],
)

py_proto_library(
    name = "tensor_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":tensor_proto"],
)

proto_library(
    name = "agg_core_proto",
    srcs = ["agg_core.proto"],
    deps = [":tensor_proto"],
)

cc_proto_library(
    name = "agg_core_cc_proto",
    deps = [":agg_core_proto"],
)

cc_library(
    name = "tensor",
    srcs = [
        "datatype.cc",
        "input_tensor_list.cc",
        "tensor.cc",
        "tensor_data.cc",
        "tensor_shape.cc",
        "tensor_spec.cc",
    ],
    hdrs = [
        "agg_vector.h",
        "agg_vector_iterator.h",
        "datatype.h",
        "input_tensor_list.h",
        "mutable_vector_data.h",
        "tensor.h",
        "tensor_data.h",
        "tensor_shape.h",
        "tensor_spec.h",
        "vector_string_data.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "aggregator",
    srcs = [
        "tensor_aggregator.cc",
        "tensor_aggregator_registry.cc",
    ],
    hdrs = [
        "agg_vector_aggregator.h",
        "aggregator.h",
        "one_dim_grouping_aggregator.h",
        "tensor_aggregator.h",
        "tensor_aggregator_factory.h",
        "tensor_aggregator_registry.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "intrinsic",
    hdrs = ["intrinsic.h"],
    visibility = ["//visibility:public"],
    deps = [":tensor"],
)

cc_library(
    name = "aggregation_cores",
    srcs = [
        "composite_key_combiner.cc",
        "dp_closed_domain_histogram.cc",
        "dp_composite_key_combiner.cc",
        "dp_group_by_factory.cc",
        "dp_grouping_federated_sum.cc",
        "dp_open_domain_histogram.cc",
        "federated_mean.cc",
        "federated_sum.cc",
        "group_by_aggregator.cc",
        "grouping_federated_sum.cc",
        "one_dim_grouping_aggregator.cc",
    ],
    hdrs = [
        "composite_key_combiner.h",
        "dp_closed_domain_histogram.h",
        "dp_composite_key_combiner.h",
        "dp_group_by_factory.h",
        "dp_open_domain_histogram.h",
        "group_by_aggregator.h",
        "one_dim_grouping_aggregator.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":dp_fedsql_constants",
        ":fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_cc_differential_privacy//algorithms:partition-selection",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_data_test.cc",
        "tensor_shape_test.cc",
        "tensor_spec_test.cc",
        "tensor_test.cc",
    ],
    deps = [
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "agg_vector_test",
    srcs = ["agg_vector_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_test(
    name = "aggregator_test",
    srcs = ["agg_vector_aggregator_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "tensor_aggregator_registry_test",
    srcs = ["tensor_aggregator_registry_test.cc"],
    deps = [
        ":aggregator",
        ":intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_library(
    name = "mutable_string_data",
    hdrs = ["mutable_string_data.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

cc_library(
    name = "fedsql_constants",
    hdrs = ["fedsql_constants.h"],
)

cc_library(
    name = "dp_fedsql_constants",
    hdrs = ["dp_fedsql_constants.h"],
)

cc_test(
    name = "federated_mean_test",
    srcs = ["federated_mean_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "federated_sum_test",
    srcs = ["federated_sum_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "input_tensor_list_test",
    srcs = ["input_tensor_list_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_test(
    name = "composite_key_combiner_test",
    srcs = ["composite_key_combiner_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_composite_key_combiner_test",
    srcs = ["dp_composite_key_combiner_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mutable_vector_data_test",
    srcs = ["mutable_vector_data_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "vector_string_data_test",
    srcs = [
        "vector_string_data.h",
        "vector_string_data_test.cc",
    ],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "mutable_string_data_test",
    srcs = ["mutable_string_data_test.cc"],
    deps = [
        ":mutable_string_data",
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "one_dim_grouping_aggregator_test",
    srcs = ["one_dim_grouping_aggregator_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "group_by_aggregator_test",
    srcs = ["group_by_aggregator_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_open_domain_histogram_test",
    srcs = ["dp_open_domain_histogram_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_closed_domain_histogram_test",
    srcs = ["dp_closed_domain_histogram_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "grouping_federated_sum_test",
    srcs = ["grouping_federated_sum_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_grouping_federated_sum_test",
    srcs = ["dp_grouping_federated_sum_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "federated_sum_bench",
    testonly = 1,
    srcs = ["federated_sum_bench.cc"],
    linkstatic = 1,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "group_by_bench",
    testonly = 1,
    srcs = ["group_by_bench.cc"],
    linkstatic = 1,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":mutable_string_data",
        ":tensor",
        ":tensor_cc_proto",
        "@com_google_benchmark//:benchmark",
    ],
)
