# Internal modules for JAX Diffren.

load("@pip//:requirements.bzl", "requirement")

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
)

py_test(
    name = "framebuffer_test",
    srcs = ["framebuffer_test.py"],
    deps = [
        ":framebuffer",
        requirement("absl_py"),
        requirement("chex"),
        requirement("jax"),
    ],
)

py_library(
    name = "framebuffer",
    srcs = ["framebuffer.py"],
    deps = [
        requirement("flax"),
        requirement("jax"),
    ],
)

py_library(
    name = "interpolate",
    srcs = ["interpolate.py"],
    deps = [
        "//diffren/jax/internal:framebuffer",
        requirement("jax"),
    ],
)

py_test(
    name = "interpolate_test",
    srcs = ["interpolate_test.py"],
    data = [
        "//diffren/common/test_data:images",
    ],
    deps = [
        "//diffren/common:test_utils",
        "//diffren/jax:constants",
        "//diffren/jax:render",
        requirement("absl_py"),
        requirement("chex"),
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
    ],
)

py_library(
    name = "rasterize",
    srcs = ["rasterize.py"],
    deps = [
        "//diffren/jax:constants",
        "//diffren/jax/internal:framebuffer",
        "//diffren/jax/internal/kernels:rasterize_triangles_xla",
        requirement("jax"),
    ],
)

py_test(
    name = "rasterize_test",
    srcs = ["rasterize_test.py"],
    data = [
        "//diffren/common/test_data:images",
    ],
    deps = [
        ":rasterize",
        "//diffren/common:test_utils",
        requirement("absl_py"),
        requirement("chex"),
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
    ],
)

py_library(
    name = "splat",
    srcs = ["splat.py"],
    deps = [
        ":framebuffer",
        ":interpolate",
        "//diffren/jax:composite",
        "//diffren/jax/utils:transforms",
        requirement("jax"),
    ],
)

py_test(
    name = "splat_test",
    srcs = ["splat_test.py"],
    data = [
        "//diffren/common/test_data:images",
        "//diffren/common/test_data:meshes",
    ],
    deps = [
        "//diffren/common:compare_images",
        "//diffren/common:obj_loader",
        "//diffren/common:test_utils",
        "//diffren/jax:camera",
        "//diffren/jax:constants",
        "//diffren/jax:render",
        "//diffren/jax/utils:mesh",
        "//diffren/jax/utils:shaders",
        "//diffren/jax/utils:transforms",
        requirement("absl_py"),
        requirement("chex"),
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
        requirement("pillow"),
        requirement("scikit_image"),
    ],
)
