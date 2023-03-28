# JAX Diffren main interface modules.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load( "@pip//:requirements.bzl", "requirement",)

py_library(
    name = "camera",
    srcs = ["camera.py"],
    deps = [
        "//diffren/jax/utils:transforms",
        requirement("jax"),
    ],
)

py_library(
    name = "render",
    srcs = ["render.py"],
    deps = [
        ":composite",
        ":constants",
        "//diffren/jax/internal:interpolate",
        "//diffren/jax/internal:rasterize",
        "//diffren/jax/internal:splat",
        requirement("jax"),
    ],
)

py_library(
    name = "composite",
    srcs = ["composite.py"],
    deps = [requirement("jax")],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_test(
    name = "camera_test",
    srcs = ["camera_test.py"],
    deps = [
        ":camera",
        "//diffren/common:test_utils",
        requirement("absl_py"),
        requirement("chex"),
        requirement("jax"),
        requirement("numpy"),
        requirement("transformations"),
    ],
)

py_test(
    name = "render_test",
    srcs = ["render_test.py"],
    data = [
        "//diffren/common/test_data:images",
        "//diffren/common/test_data:meshes",
    ],
    deps = [
        ":constants",
        ":render",
        "//diffren/common:compare_images",
        "//diffren/common:obj_loader",
        "//diffren/common:test_utils",
        "//diffren/jax/internal:rasterize",
        "//diffren/jax/utils:transforms",
        requirement("absl_py"),
        requirement("chex"),
        requirement("jax"),
        requirement("numpy"),
    ],
)
