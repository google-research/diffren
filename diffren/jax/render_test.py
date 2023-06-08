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

"""Tests for render, the main Diffren entry point."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import compare_images
from diffren.common import obj_loader
from diffren.common import test_utils
from diffren.jax import constants
from diffren.jax import render
from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp
import numpy as np

WORLD_POSITION = 'world_position'


class RenderTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RenderTest, self).setUp()

    self.cube_vertex_positions = jnp.array(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=jnp.float32)
    self.cube_triangles = jnp.array(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=jnp.int32)

    perspective = test_utils.make_perspective_matrix()
    projection_1 = transforms.hi_prec_matmul(
        perspective, test_utils.make_look_at_matrix('view_1')
    )
    projection_2 = transforms.hi_prec_matmul(
        perspective, test_utils.make_look_at_matrix('view_2')
    )
    self.projection = jnp.stack([projection_1, projection_2], axis=0)

  @parameterized.parameters(['no batching', 'batched vertices', 'all batched'])
  def testRendersTwoCubesBatchModes(self, batch_mode):
    """Renders a simple cube in two viewpoints with various batching."""
    vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
    vertex_rgba = jnp.concatenate([vertex_rgb, jnp.ones([8, 1])], axis=1)

    tile_dims = jnp.array([2, 1, 1])
    if batch_mode == 'no batching':
      vertices = self.cube_vertex_positions
      attributes = vertex_rgba
      triangles = self.cube_triangles
      in_axes = (None, None, None, 0)
    elif batch_mode == 'batched vertices':
      vertices = jnp.tile(self.cube_vertex_positions[jnp.newaxis, ...],
                          tile_dims)
      attributes = vertex_rgba
      triangles = self.cube_triangles
      in_axes = (0, None, None, 0)
    elif batch_mode == 'all batched':
      vertices = jnp.tile(self.cube_vertex_positions[jnp.newaxis, ...],
                          tile_dims)
      attributes = jnp.tile(vertex_rgba[jnp.newaxis, ...], tile_dims)
      triangles = jnp.tile(self.cube_triangles[jnp.newaxis, ...], tile_dims)
      in_axes = (0, 0, 0, 0)

    def render_for_vmap(v, a, t, p):
      """Wraps render_triangles to expose only the parameters we want to vmap."""
      return render.render_triangles(
          v, {'rgba': a},
          t,
          p,
          test_utils.IMAGE_WIDTH,
          test_utils.IMAGE_HEIGHT,
          shading_function=lambda x: x['rgba'],
          face_culling_mode=constants.FaceCullingMode.NONE,
          compositing_mode=constants.CompositingMode.OVER)

    rendered = jax.vmap(render_for_vmap, in_axes)(vertices, attributes,
                                                  triangles, self.projection)

    for i in (0, 1):
      test_utils.check_image(self, np.array(rendered[i, :, :, :]),
                             'Unlit_Cube_{}_0.png'.format(i))

  def testRendersSphereDepthMap(self):
    """Renders a disparity (1/depth) map of a sphere."""

    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('sphere.obj'))

    def shader(attributes):
      """Computes a normalized disparity value from the rasterized positions."""
      positions = attributes[WORLD_POSITION]
      look_at = test_utils.make_look_at_matrix('view_1')
      flat_positions = jnp.reshape(positions, (-1, 3))
      camera_positions = transforms.transform_homogeneous(
          look_at, flat_positions)
      disparity = -1 / camera_positions[:, 2]
      # TODO(b/238067814): update test baseline after rasterize_triangles_test
      # is removed. This test uses per-pixel interpolation vs. the per-vertex
      # intepolation in rasterize_triangles_test, so the min and max disparity
      # values are slightly different.
      test_image_min = 0.12523349
      test_image_max = 0.16625337
      disparity = (disparity - test_image_min) / (
          test_image_max - test_image_min)
      disparity = jnp.reshape(disparity, positions.shape[:2])
      return jnp.stack([disparity, jnp.ones_like(disparity)], axis=-1)

    rendered = render.render_triangles(  # pytype: disable=wrong-arg-types  # numpy-scalars
        vertices, {WORLD_POSITION: vertices},
        triangles,
        self.projection[0, ...],
        test_utils.IMAGE_WIDTH,
        test_utils.IMAGE_HEIGHT,
        shader,
        compositing_mode=constants.CompositingMode.OVER)

    rendered_disparity = rendered[:, :, 0:1]
    image = compare_images.get_pil_formatted_image(np.array(rendered_disparity))
    target_image_name = 'Sphere_Disparity.png'
    baseline_image_path = test_utils.make_resource_path(target_image_name)
    compare_images.expect_image_file_and_image_are_near(
        self, baseline_image_path, image, target_image_name,
        '%s does not match.' % target_image_name)


if __name__ == '__main__':
  absltest.main()
