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
    #deps = [":descriptors_proto"],
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
        #"//third_party/tensorflow/compiler/xla/service:custom_call_status_public_headers",
        #"//third_party/tensorflow/compiler/xla/service:custom_call_target_registry",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        #"@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@com_google_absl//absl/status",
        #"@com_google_absl//absl/cast",
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
        "@com_google_absl//absl/types:span",
        "//diffren/common/kernels:rasterize_triangles_impl",
        #"//third_party/tensorflow/compiler/xla/service:custom_call_status_public_headers",
        #"//third_party/tensorflow/compiler/xla/service:custom_call_target_registry",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        #"@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
        #"@com_google_absl//absl/cast",
        "@pybind11",
    ],
)

# cc_binary(
#    name = "rasterize_triangles_cpu_bin",
#    deps = [
#         ":rasterize_triangles_cpu",
#    ],
#    linkshared = 1,
#    linkstatic = 0,
# )

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
