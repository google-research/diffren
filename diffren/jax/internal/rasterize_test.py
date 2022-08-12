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

"""Tests for rasterization functions."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import test_utils
from diffren.common.test_utils import compare_images
from diffren.jax.internal import rasterize
import jax.numpy as jnp
import numpy as np


class RasterizeTest(chex.TestCase, parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ('w constant', [1.0, 1.0, 1.0], 'Simple_Triangle.png', False),
      ('w constant None camera', [1.0, 1.0, 1.0], 'Simple_Triangle.png', True),
      ('w varying', [0.2, 0.5, 2.0
                    ], 'Perspective_Corrected_Triangle.png', False))
  def test_render_simple_triangle(self, w_vector, target_image_name,
                                  use_none_camera):
    """Directly renders a rasterized triangle's barycentric coordinates.

    Tests the wrapping code as well as the kernel.

    Args:
      w_vector: 3 element vector of w components to scale triangle vertices.
      target_image_name: image file name to compare result against.
      use_none_camera: pass in None as the camera transform, or the identity
        matrix
    """
    clip_coordinates = jnp.array(
        [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
        dtype=jnp.float32)
    clip_coordinates = clip_coordinates * jnp.reshape(
        jnp.array(w_vector, dtype=jnp.float32), [3, 1])
    camera = None if use_none_camera else jnp.eye(4)

    @self.variant
    def call_rasterize(vertices, triangles):
      return rasterize.rasterize_triangles(vertices, triangles, camera,
                                           test_utils.IMAGE_WIDTH,
                                           test_utils.IMAGE_HEIGHT)

    framebuffer = call_rasterize(clip_coordinates, ((0, 1, 2),))

    rendered_coordinates = jnp.pad(
        framebuffer.barycentrics[0, ...], ((0, 0), (0, 0), (0, 1)),
        constant_values=(1,))
    image = compare_images.get_pil_formatted_image(
        np.array(rendered_coordinates))
    baseline_image_path = test_utils.make_resource_path(target_image_name)
    compare_images.expect_image_file_and_image_are_near(
        self, baseline_image_path, image, target_image_name,
        '%s does not match.' % target_image_name)


if __name__ == '__main__':
  absltest.main()
