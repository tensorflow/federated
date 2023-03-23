workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

#
# Direct dependencies
#

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.3.0",
)

git_repository(
    name = "com_google_absl",
    commit = "66665d8d2e3fedff340b83f9841ca427145a7b26",
    remote = "https://github.com/abseil/abseil-cpp.git",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.19.0",
)

git_repository(
    name = "io_bazel_rules_go",
    remote = "https://github.com/bazelbuild/rules_go.git",
    tag = "v0.29.0",
)

git_repository(
    name = "org_tensorflow",
    # TODO(b/256948367): Temporarily updating the version of TF past the version
    # in https://github.com/tensorflow/federated/blob/main/requirements.txt.
    #
    # The version of this dependency should match the version in
    # https://github.com/tensorflow/federated/blob/main/requirements.txt.
    commit = "5e905e50fb9ee78dd9a79ccd7ca7436b1c75c299",
    patches = [
        # Depending on restricted visibility BUILD target om external git
        # repository does not seem to be supported.
        # E.g. issue: https://github.com/bazelbuild/bazel/issues/3744
        # TODO(b/263201501): Make DTensor C++ API public and remove this patch.
        "//third_party/tensorflow:dtensor_internal_visibility.patch",
        "//third_party/tensorflow:internal_visibility.patch",
        "//third_party/tensorflow:tf2xla_visibility.patch",
    ],
    remote = "https://github.com/tensorflow/tensorflow.git",
)

git_repository(
    name = "pybind11_abseil",
    commit = "38111ef06d426f75bb335a3b58aa0342f6ce0ce3",
    remote = "https://github.com/pybind/pybind11_abseil.git",
)

git_repository(
    name = "pybind11_protobuf",
    commit = "a3d93a93387af7fa57d72d56cfc0a4ba7f4a60e4",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
)

git_repository(
    name = "rules_license",
    remote = "https://github.com/bazelbuild/rules_license.git",
    tag = "0.0.4",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.5.0",
)

git_repository(
    name = "tensorflow_compression",
    remote = "https://github.com/tensorflow/compression.git",
    # The version of this dependency should match the version in
    # https://github.com/tensorflow/federated/blob/main/requirements.txt.
    tag = "v2.11.0",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_abseil and pybind11_protobuf
new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v2.9.2",
)

#
# Transitive dependencies, grouped by direct dependency.
#

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies")

go_rules_dependencies()

# TODO(b/260612484): Temporarily disable the direct dependency on
# `go_register_toolchains`, for now we pick this dependency up via TensorFlows
# workspace.
# go_register_toolchains(version = "1.17.1")

git_repository(
    name = "bazel_gazelle",
    remote = "https://github.com/bazelbuild/bazel-gazelle.git",
    tag = "v0.24.0",
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

git_repository(
    name = "com_github_grpc_grpc",
    remote = "https://github.com/grpc/grpc.git",
    tag = "v1.38.1",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# TODO(b/260598663): Temporarily disable the direct dependency on
# `grpc_extra_deps`, for now we pick this dependency up via TensorFlows
# workspace.
# load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
#
# grpc_extra_deps()
