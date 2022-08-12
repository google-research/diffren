# Utility functions for JAX Diffren.

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "transforms",
    srcs = ["transforms.py"],
    deps = ["@diffren_deps//pypi__jax"],
)

py_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    deps = [
        ":transforms",
        "@diffren_deps//pypi__absl_py",
        "@diffren_deps//pypi__chex",
        "@diffren_deps//pypi__jax",
        "@diffren_deps//pypi__numpy",
        "@diffren_deps//pypi__transformations",
    ],
)

py_library(
    name = "mesh",
    srcs = ["mesh.py"],
    deps = [
        ":transforms",
        "@diffren_deps//pypi__jax",
    ],
)

py_library(
    name = "shaders",
    srcs = ["shaders.py"],
    deps = [
        ":transforms",
        "@diffren_deps//pypi__etils",
        "@diffren_deps//pypi__jax",
        "@diffren_deps//pypi__numpy",
        "@diffren_deps//pypi__pillow",
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
        "@diffren_deps//pypi__absl_py",
        "@diffren_deps//pypi__chex",
        "@diffren_deps//pypi__jax",
        "@diffren_deps//pypi__numpy",
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
        "@diffren_deps//pypi__absl_py",
        "@diffren_deps//pypi__chex",
        "@diffren_deps//pypi__jax",
        "@diffren_deps//pypi__numpy",
    ],
)
