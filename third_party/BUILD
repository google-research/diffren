load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_visibility = ["//:__subpackages__"],
)

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
)
