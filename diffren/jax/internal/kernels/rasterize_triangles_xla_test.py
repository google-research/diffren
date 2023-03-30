# Copyright 2023 The diffren Authors.
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

"""Tests for rasterize_triangles_xla."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import test_utils
from diffren.jax.internal.kernels import rasterize_triangles_xla
import jax
import jax.numpy as jnp
import numpy as np


class RasterizeTrianglesTest(chex.TestCase):

  def setUp(self):
    super(RasterizeTrianglesTest, self).setUp()
    self.clip_coordinates = jnp.array(
        [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
        dtype=jnp.float32)
    self.triangles = jnp.array([[0, 1, 2]], dtype=jnp.int32)

  def check_simple_triangle(self, rendered_coordinates):
    width, height = (test_utils.IMAGE_WIDTH, test_utils.IMAGE_HEIGHT)
    rgba_image = jnp.concatenate([
        jnp.reshape(rendered_coordinates, [height, width, 3]),
        jnp.ones([height, width, 1])
    ],
                                 axis=2)
    rgba_image = np.array(rgba_image)
    test_utils.check_image(self, rgba_image, 'Simple_Triangle.png')

  @chex.variants(with_jit=True, without_jit=True)
  def test_renders_simple_triangle_no_batch(self):

    @self.variant
    def call_rasterize(vertices, triangles):
      _, _, rendered_coordinates = rasterize_triangles_xla.rasterize_triangles(
          vertices, triangles, test_utils.IMAGE_WIDTH, test_utils.IMAGE_HEIGHT,
          1, 0)
      return rendered_coordinates

    self.check_simple_triangle(
        call_rasterize(self.clip_coordinates, self.triangles))

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters((None, 0), (0, None), (0, 0))
  def test_renders_simple_triangle_with_batch(self, vertices_batch_axis,
                                              triangles_batch_axis):

    @self.variant
    def call_rasterize(vertices, triangles):
      _, _, rendered_coordinates = rasterize_triangles_xla.rasterize_triangles(
          vertices, triangles, test_utils.IMAGE_WIDTH, test_utils.IMAGE_HEIGHT,
          1, 0)
      return rendered_coordinates

    clip_coordinates = self.clip_coordinates
    if vertices_batch_axis is not None:
      clip_coordinates = jnp.expand_dims(
          clip_coordinates, axis=vertices_batch_axis)

    triangles = self.triangles
    if triangles_batch_axis is not None:
      triangles = jnp.expand_dims(triangles, axis=triangles_batch_axis)

    rendered_coordinates = jax.vmap(
        call_rasterize,
        in_axes=(vertices_batch_axis, triangles_batch_axis))(clip_coordinates,
                                                             triangles)
    rendered_coordinates = rendered_coordinates[0, ...]
    self.check_simple_triangle(rendered_coordinates)


if __name__ == '__main__':
  absltest.main()
