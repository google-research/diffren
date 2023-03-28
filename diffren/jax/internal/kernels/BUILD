load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "pybind_extension",
)
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

proto_library(
    name = "descriptors_proto",
    srcs = ["descriptors.proto"],
)

py_proto_library(
    name = "descriptors_py_pb2",
    srcs = ["descriptors.proto"],
)

cc_proto_library(  # buildifier: disable=native-cc-proto
    name = "descriptors_cc_proto",
    deps = [":descriptors_proto"],
)

pybind_extension(
    name = "rasterize_triangles_gpu",
    srcs = [
        "rasterize_triangles_gpu.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rasterize_triangles_gpu",
    deps = [
        ":descriptors_cc_proto",
        "//diffren/common/kernels:rasterize_triangles_impl_cuda",
        "@com_google_absl//absl/status",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor/cuda:cudart_stub",
        "@pybind11",
    ],
)

pybind_extension(
    name = "rasterize_triangles_cpu",
    srcs = ["rasterize_triangles_cpu.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rasterize_triangles_cpu",
    deps = [
        "//diffren/common/kernels:rasterize_triangles_impl",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@pybind11",
    ],
)

py_test(
    name = "rasterize_triangles_xla_test",
    srcs = ["rasterize_triangles_xla_test.py"],
    data = [
        "//diffren/common/test_data:images",
    ],
    deps = [
        ":rasterize_triangles_xla",
        "//diffren/common:test_utils",
        "@diffren_deps//pypi__absl_py",
        "@diffren_deps//pypi__chex",
        "@diffren_deps//pypi__jax",
    ],
)

py_library(
    name = "rasterize_triangles_xla",
    srcs = ["rasterize_triangles_xla.py"],
    deps = [
        ":descriptors_py_pb2",
        ":rasterize_triangles_cpu",
        ":rasterize_triangles_gpu",
    ],
)
