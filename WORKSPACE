# Initialize rules_python for special Python Bazel rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    sha256 = "a3a6e99f497be089f81ec082882e40246bfd435f52f4e82f37e89449b04573f6",
    strip_prefix = "rules_python-0.10.2",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.10.2.tar.gz",
)

# Create a pip repo @diffren_deps that contains Bazel targets for
# all the third-party packages specified in the requirements.txt file.
load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")
pip_install_dependencies()

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
   name = "diffren_deps",
   requirements = "//third_party:requirements.txt",
)

# Import Tensorflow. This version should track the version imported by JAX.
http_archive(
    name = "org_tensorflow",
    sha256 = "9073ab3cbf3a89baee459f6e953cee240864393774f568fdba200a6ff5512c9f",
    strip_prefix = "tensorflow-a4905aa04186bcaf89b06032baa450cc1ce103ad",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/a4905aa04186bcaf89b06032baa450cc1ce103ad.tar.gz",
    ],
)


# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

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
