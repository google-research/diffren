# Initialize rules_python for special Python Bazel rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    sha256 = "a3a6e99f497be089f81ec082882e40246bfd435f52f4e82f37e89449b04573f6",
    strip_prefix = "rules_python-0.10.2",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.10.2.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_10",
    # Available versions are listed in @rules_python//python:versions.bzl.
    # We recommend using the same version your team is already standardized on.
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")

# Create a pip repo @diffren_deps that contains Bazel targets for
# all the third-party packages specified in the requirements.txt file.
load("@rules_python//python:pip.bzl", "pip_install")
pip_install(
    python_interpreter_target = interpreter,
    requirements = "//third_party:requirements.txt",
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
)

# import xla in the same way that jax does.

http_archive(
    name = "xla",
    sha256 = "9f39af4d81d2c8bd52b47f4ef37dfd6642c6950112e4d9d3d95ae19982c46eba",
    strip_prefix = "xla-0f31407ee498e6dba242d03f8d382ebcfcc61790",
    urls = [
        "https://github.com/openxla/xla/archive/0f31407ee498e6dba242d03f8d382ebcfcc61790.tar.gz",
    ],
)

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

local_repository(
    name = "jax",
    path = "../jax"
)

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

