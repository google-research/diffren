load(
    "@jax//jaxlib:jax.bzl",
    "pybind_extension",
)
load("@pip//:requirements.bzl", "requirement")

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

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
        "//diffren/common/kernels:rasterize_triangles_impl_cuda",
        "//diffren/common/kernels:rasterize_triangles_types",
        "@com_google_absl//absl/base",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor/cuda:cudart_stub",
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
        "//diffren/common/kernels:rasterize_triangles_types",
        "@com_google_absl//absl/base",
        "@pybind11",
        "@xla//xla/ffi/api:ffi",
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
        requirement("absl_py"),
        requirement("chex"),
        requirement("jax"),
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
