load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "v0",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

proto_library(
    name = "array_proto",
    srcs = ["array.proto"],
    deps = [":data_type_proto"],
)

py_proto_library(
    name = "array_py_pb2",
    deps = [":array_proto"],
)

cc_proto_library(
    name = "array_cc_proto",
    deps = [":array_proto"],
)

proto_library(
    name = "computation_proto",
    srcs = ["computation.proto"],
    deps = [
        ":array_proto",
        ":data_type_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

py_proto_library(
    name = "computation_py_pb2",
    deps = [":computation_proto"],
)

cc_proto_library(
    name = "computation_cc_proto",
    deps = [":computation_proto"],
)

proto_library(
    name = "data_type_proto",
    srcs = ["data_type.proto"],
)

py_proto_library(
    name = "data_type_py_pb2",
    deps = [":data_type_proto"],
)

cc_proto_library(
    name = "data_type_cc_proto",
    deps = [":data_type_proto"],
)

proto_library(
    name = "executor_proto",
    srcs = ["executor.proto"],
    deps = [
        ":array_proto",
        ":computation_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

py_proto_library(
    name = "executor_py_pb2",
    deps = [":executor_proto"],
)

py_grpc_library(
    name = "executor_py_pb2_grpc",
    srcs = [":executor_proto"],
    deps = [":executor_py_pb2"],
)

cc_proto_library(
    name = "executor_cc_proto",
    deps = [":executor_proto"],
)

cc_grpc_library(
    name = "executor_cc_grpc_proto",
    srcs = [":executor_proto"],
    grpc_only = True,
    deps = [":executor_cc_proto"],
)
