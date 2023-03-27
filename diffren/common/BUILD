package(default_visibility = ["//:__subpackages__"])

load( "@pip//:requirements.bzl", "requirement",)

licenses(["notice"])

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":compare_images",
        ":obj_loader",
    ],
)

py_library(
    name = "obj_loader",
    srcs = ["obj_loader.py"],
    deps = [
        requirement("etils"),
        requirement("importlib-resources"),
        requirement("typing-extensions"),
    ],
)

py_test(
    name = "obj_loader_test",
    srcs = ["obj_loader_test.py"],
    data = ["//diffren/common/test_data:meshes"],
    deps = [
        ":obj_loader",
        ":test_utils",
        requirement("absl_py"),
        requirement("numpy"),
    ],
)

py_library(
    name = "compare_images",
    testonly = 1,
    srcs = ["compare_images.py"],
    deps = [
        requirement("pillow"),
    ],
)
