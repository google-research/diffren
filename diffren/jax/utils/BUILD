# Utility functions for JAX Diffren.

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
)

load( "@pip//:requirements.bzl", "requirement",)

py_library(
    name = "transforms",
    srcs = ["transforms.py"],
    deps = [requirement("jax")],
)

py_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    deps = [
        ":transforms",
        requirement("absl_py"),
        requirement("chex"),
        requirement("jax"),
        requirement("numpy"),
        requirement("transformations"),
    ],
)

py_library(
    name = "mesh",
    srcs = ["mesh.py"],
    deps = [
        ":transforms",
        requirement("jax"),
    ],
)

py_library(
    name = "shaders",
    srcs = ["shaders.py"],
    deps = [
        ":transforms",
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
        requirement("pillow"),
    ],
)

py_test(
    name = "shaders_test",
    srcs = ["shaders_test.py"],
    data = [
        "//diffren/common/test_data:images",
        "//diffren/common/test_data:meshes",
    ],
    deps = [
        ":mesh",
        ":shaders",
        ":transforms",
        "//diffren/common:compare_images",
        "//diffren/common:obj_loader",
        "//diffren/common:test_utils",
        "//diffren/jax:camera",
        "//diffren/jax:constants",
        "//diffren/jax:render",
        requirement("absl_py"),
        requirement("chex"),
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
    ],
)

py_test(
    name = "mesh_test",
    srcs = ["mesh_test.py"],
    data = ["//diffren/common/test_data:meshes"],
    deps = [
        ":mesh",
        "//diffren/common:obj_loader",
        "//diffren/common:test_utils",
        requirement("absl_py"),
        requirement("chex"),
        requirement("etils"),
        requirement("jax"),
        requirement("numpy"),
    ],
)

py_library(
    name = "depthmap",
    srcs = ["depthmap.py"],
    deps = [
        ":mesh",
        requirement("einops"),
        requirement("jax"),
    ],
)

py_test(
    name = "depthmap_test",
    srcs = ["depthmap_test.py"],
    data = [
        "//diffren/common/test_data:images",
        "//diffren/common/test_data:meshes",
    ],
    deps = [
        ":depthmap",
        ":shaders",
        ":transforms",
        "//diffren/common:compare_images",
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
