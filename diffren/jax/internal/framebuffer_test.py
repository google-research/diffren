# Copyright 2022 The diffren Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Diffren framebuffer class."""

from absl.testing import absltest
import chex
from diffren.jax.internal import framebuffer as fb
import jax.numpy as jnp


class FramebufferTest(chex.TestCase):

  def test_initialize_framebuffer_with_wrong_rank(self):
    with self.assertRaisesRegex(ValueError,
                                "Expected all input shapes to match"):
      fb.Framebuffer(
          jnp.ones([1, 1, 4, 4, 1]), jnp.ones([4, 3]), jnp.ones([3, 4, 4, 5,
                                                                 5]),
          jnp.ones([3, 4, 4, 5, 5]))

  def test_initialize_framebuffer_with_wrong_shapes(self):
    with self.assertRaisesRegex(ValueError,
                                "Expected all input shapes to match"):
      fb.Framebuffer(
          jnp.ones([1, 1, 4, 4, 3]), jnp.ones([1, 1, 4, 4, 1]),
          jnp.ones([1, 1, 4, 4, 3]), jnp.ones([1, 1, 4, 4, 1]),
          {"an_attr": jnp.ones([1, 1, 4, 3, 4])})


if __name__ == "__main__":
  absltest.main()
