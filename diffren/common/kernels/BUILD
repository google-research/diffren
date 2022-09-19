load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
)

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "rasterize_triangles_types",
    srcs = [],
    hdrs = ["rasterize_triangles_types.h"],
    deps = [],
)

cc_library(
    name = "rasterize_utils",
    hdrs = ["rasterize_utils.h"],
    deps = [
        ":rasterize_triangles_types",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "rasterize_triangles_impl",
    srcs = ["rasterize_triangles_impl.cc"],
    hdrs = ["rasterize_triangles_impl.h"],
    deps = [
        ":rasterize_triangles_types",
        ":rasterize_utils",
        "@com_google_absl//absl/base",
    ],
)

cuda_library(
    name = "rasterize_triangles_impl_cuda",
    srcs = ["rasterize_triangles_impl_cuda.cu.cc"],
    hdrs = ["rasterize_triangles_impl_cuda.cu.h"],
    deps = [
        ":rasterize_triangles_types",
        ":rasterize_utils",
        "@com_google_absl//absl/status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "rasterize_test_utils",
    srcs = ["rasterize_test_utils.cc"],
    hdrs = ["rasterize_test_utils.h"],
    deps = [
        "//third_party/lodepng",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rasterize_triangles_impl_test",
    srcs = ["rasterize_triangles_impl_test.cc"],
    data = [
        "//diffren/common/test_data:images",
    ],
    linkopts = [
        # TODO(fcole): figure out why this linker option is necessary, or figure
        # out how to use CUDA_HOME to find cuda.
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lm",
    ],
    linkstatic = 1,
    deps = [
        ":rasterize_test_utils",
        ":rasterize_triangles_impl",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rasterize_triangles_impl_cuda_test",
    srcs = ["rasterize_triangles_impl_cuda_test.cc"],
    data = [
        "//diffren/common/test_data:images",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lm",
    ],
    linkstatic = 1,
    deps = [
        ":rasterize_test_utils",
        ":rasterize_triangles_impl_cuda",
        "@com_google_googletest//:gtest_main",
    ],
)
