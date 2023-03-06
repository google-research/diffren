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
import jax
import jax.numpy as jnp
import numpy as np


class RasterizeTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RasterizeTest, self).setUp()

    self.cube_vertex_positions = jnp.array(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=jnp.float32)
    self.cube_triangles = jnp.array(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=jnp.int32)

    perspective = test_utils.make_perspective_matrix()
    self.projection = jnp.matmul(
        perspective, test_utils.make_look_at_matrix('view_1')
    )

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ('w constant', [1.0, 1.0, 1.0], 'Simple_Triangle.png', False, False),
      ('w constant diff barys', [1.0, 1.0, 1.0], 'Simple_Triangle.png', False,
       True), ('w constant None camera', [1.0, 1.0, 1.0
                                         ], 'Simple_Triangle.png', True, False),
      ('w varying', [0.2, 0.5, 2.0], 'Perspective_Corrected_Triangle.png',
       False, False), ('w varying diff barys', [0.2, 0.5, 2.0],
                       'Perspective_Corrected_Triangle.png', False, True))
  def test_render_simple_triangle(self, w_vector, target_image_name,
                                  use_none_camera, use_diff_barys):
    """Directly renders a rasterized triangle's barycentric coordinates.

    Tests the wrapping code as well as the kernel.

    Args:
      w_vector: 3 element vector of w components to scale triangle vertices.
      target_image_name: image file name to compare result against.
      use_none_camera: pass in None as the camera transform, or the identity
        matrix
      use_diff_barys: compute and test differentiable barycentric coordinates.
    """
    clip_coordinates = jnp.array(
        [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
        dtype=jnp.float32)
    clip_coordinates = clip_coordinates * jnp.reshape(
        jnp.array(w_vector, dtype=jnp.float32), [3, 1])
    camera = None if use_none_camera else jnp.eye(4)

    @self.variant
    def call_rasterize(vertices, triangles):
      return rasterize.rasterize_triangles(
          vertices,
          triangles,
          camera,
          test_utils.IMAGE_WIDTH,
          test_utils.IMAGE_HEIGHT,
          compute_diff_barys=use_diff_barys)

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

  @parameterized.named_parameters(('diff_barys', True),
                                  ('no_diff_barys', False))
  def test_renders_cube_barycentrics(self, use_diff_barys):
    """Renders barycentric coordinates of a cube."""
    rendered = rasterize.rasterize_triangles(
        self.cube_vertex_positions,
        self.cube_triangles,
        self.projection,
        test_utils.IMAGE_WIDTH,
        test_utils.IMAGE_HEIGHT,
        compute_diff_barys=use_diff_barys)

    test_utils.check_image(
        self,
        np.array(rendered.barycentrics[0]),
        'Barycentrics_Cube.png',
        add_transparency=False)

  def test_render_simple_triangle_gradient(self):
    """Verifies the Jacobian for a single pixel.

    The pixel is in the center of a triangle facing the camera. This makes it
    easy to check which entries of the Jacobian might not make sense without
    worrying about corner cases.
    """
    image_height = 48
    image_width = 64
    test_pixel_x = 32
    test_pixel_y = 24

    triangles = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    def non_diff_compute(ndc_coordinates):
      framebuffer = rasterize.rasterize_triangles(ndc_coordinates, triangles,
                                                  jnp.eye(4), image_width,
                                                  image_height)
      pixels_to_compare = (
          framebuffer.barycentrics[0, test_pixel_y, test_pixel_x, :])
      return pixels_to_compare

    def diff_compute(ndc_coordinates):
      framebuffer = rasterize.rasterize_triangles(
          ndc_coordinates,
          triangles,
          jnp.eye(4),
          image_width,
          image_height,
          compute_diff_barys=True)
      pixels_to_compare = (
          framebuffer.barycentrics[0, test_pixel_y, test_pixel_x, :])
      return pixels_to_compare

    ndc_init = np.array([[-0.5, -0.5, 0.8], [0.0, 0.5, 0.3], [0.5, -0.5, 0.3]],
                        dtype=np.float32)

    # Check that the non-differentiable version has a zero Jacobian.
    non_diff_jac = jax.jacfwd(non_diff_compute)(ndc_init)
    np.testing.assert_array_equal(non_diff_jac, jnp.zeros_like(non_diff_jac))

    # Check that the differentiable version has the expected Jacobian.
    # Note that middle 3x3 of the Jacobian, which corresponds to the
    # derivative of the barycentric coordinate of the bottom triangle vertex
    # w.r.t. the ndc coordinates, has zero X coordinate since sliding the
    # vertices left and right does not change the barycentric coordinate of the
    # bottom vertex.
    expected_jacobian = np.array([[[0.22395831, 0.11197916, -0.],
                                   [0.5208334, 0.2604167, -0.],
                                   [0.2552083, 0.12760416, -0.]],
                                  [[0., -0.22395831, -0.],
                                   [-0., -0.5208334, -0.],
                                   [0., -0.2552083, -0.]],
                                  [[-0.22395831, 0.11197916, -0.],
                                   [-0.5208334, 0.2604167, -0.],
                                   [-0.2552083, 0.12760416, -0.]]])

    diff_jac = jax.jacfwd(diff_compute)(ndc_init)
    np.testing.assert_array_almost_equal(diff_jac, expected_jacobian)


if __name__ == '__main__':
  absltest.main()
