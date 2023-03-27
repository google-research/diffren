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

"""Tests for Diffren rasterize-then-splat mode."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import compare_images
from diffren.common import obj_loader
from diffren.common import test_utils
from diffren.jax import constants
from diffren.jax import render
from diffren.jax.utils import mesh
from diffren.jax.utils import shaders
from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image as PilImage
from skimage import filters


def load_and_blur_image(image_name, blur_sigma):
  image_path = test_utils.make_resource_path(image_name)
  image = np.array(PilImage.open(image_path)).astype(np.float32) / 255.0
  image = image[..., :3]
  blurred_image = filters.gaussian(image, blur_sigma, channel_axis=-1)
  return compare_images.get_pil_formatted_image(blurred_image)


class SplatTest(chex.TestCase, parameterized.TestCase):

  def test_two_triangle_layers(self):
    """Checks that two overlapping triangles are accumulated correctly."""
    image_width = 32
    image_height = 32

    vertices = jnp.array(
        [[-0.2, -0.2, 0, 1], [0.5, -0.2, 0, 1], [0.5, 0.5, 0, 1],
         [0.2, -0.2, 0.5, 1], [-0.5, -0.2, 0.5, 1], [-0.5, 0.5, 0.5, 1]],
        dtype=jnp.float32)
    triangles = [[0, 1, 2], [3, 5, 4]]
    colors = jnp.array([[0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0],
                        [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]],
                       dtype=jnp.float32)

    composite, _, normalized_layers = render.render_triangles(  # pytype: disable=wrong-arg-types  # jax-ndarray
        vertices, {'colors': colors},
        triangles,
        None,
        image_width,
        image_height,
        lambda x: x['colors'],
        num_layers=2,
        return_accum_buffers=True)
    with self.subTest(name='full composite'):
      test_utils.check_image(
          self,
          np.array(composite)[:, :, :],
          'Two_Triangles_Splat_Composite.png',
          resize_image_to=(512, 512))
    for i in range(3):
      with self.subTest(name=f'layer {i}'):
        test_utils.check_image(
            self,
            np.array(normalized_layers)[i, :, :, :],
            f'Two_Triangles_Splat_Layer_{i}.png',
            resize_image_to=(512, 512))

  @parameterized.named_parameters(
      ('_rgba', 4),
      ('_ba', 2),
  )
  def test_optimize_single_triangle(self, num_components):
    """Checks that the position of a triangle can be optimized correctly.

    The optimization target is a translated version of the same triangle.
    Naive rasterization produces zero gradient in this case, but
    rasterize-then-splat produces a useful gradient.

    Args:
      num_components: The number of components of the shading function outputs.
    """
    image_width = 32
    image_height = 32

    initial_vertices = jnp.array(
        [[[0, 0, 0, 1], [0.5, 0, 0, 1], [0.5, 0.5, 0, 1]]], dtype=jnp.float32)
    target_vertices = jnp.array(
        [[[-0.25, 0, 0, 1], [0.25, 0, 0, 1], [0.25, 0.5, 0, 1]]],
        dtype=jnp.float32)
    triangles = jnp.array([[0, 1, 2]], jnp.int32)
    colors = jnp.array(
        [[[0, 0., 0.8, 1.0], [0, 0.8, 0, 1.0], [0, 0., 0.8, 1.0]]],
        dtype=jnp.float32)[..., -num_components:]

    @functools.partial(jax.jit, static_argnames=['do_splat'])
    def render_splat(verts, do_splat):

      def render_inner(
          vertices_slice: jnp.ndarray,
          attributes_slice: jnp.ndarray,
      ):
        return render.render_triangles(  # pytype: disable=wrong-arg-types  # jax-ndarray
            vertices_slice, {'colors': attributes_slice},
            triangles,
            None,
            image_width,
            image_height,
            lambda x: x['colors'],
            compositing_mode=constants.CompositingMode.SPLAT_OVER
            if do_splat else constants.CompositingMode.OVER)

      return jax.vmap(render_inner)(verts, colors)

    # Perform a few iterations of gradient descent.
    num_iters = 15
    var_verts = initial_vertices
    splat_loss_initial = 0.0
    target_image = render_splat(target_vertices, do_splat=False)

    @jax.jit
    def loss_fn(verts):
      rasterized_only_image = render_splat(verts, do_splat=False)
      splat_image = render_splat(verts, do_splat=True)

      rasterized_loss = jnp.mean((rasterized_only_image - target_image)**2)
      splat_loss = jnp.mean((splat_image - target_image)**2)

      return rasterized_loss, splat_loss

    rasterized_grad_fn = jax.jit(jax.grad(lambda vs: loss_fn(vs)[0]))
    splat_grad_fn = jax.jit(jax.grad(lambda vs: loss_fn(vs)[1]))

    for i in range(num_iters):
      _, splat_loss = loss_fn(var_verts)
      rasterized_grad = rasterized_grad_fn(var_verts)
      splat_grad = splat_grad_fn(var_verts)

      if i == 0:
        # Check that the rasterized-only gradient is zero, while the
        # rasterize-then-splat gradient is non-zero.
        self.assertAlmostEqual(
            float(np.array(jnp.linalg.norm(rasterized_grad))), 0.0)
        self.assertGreater(np.array(jnp.linalg.norm(splat_grad)), 0.01)
        splat_loss_initial = splat_loss

      # Apply the gradient.
      var_verts = var_verts - splat_grad

    # Check that gradient descent reduces the loss by at least 50%.
    opt_image = render_splat(var_verts, do_splat=True)
    opt_loss = jnp.mean((opt_image - target_image)**2)
    self.assertLess(np.array(opt_loss), np.array(splat_loss_initial) * 0.5)

  def test_spiky_geometry_derivative(self):
    """Renders the derivative of single vertex spike extending from a sphere.

    The output of this test is an image showing the (normalized) dImage / dx
    where x is the offset of a single vertex on the sphere. A faint overlay of
    the original rendered image is included for visualization.

    The ideal derivative should be bright all along the spike. However, since
    we do not normalize splats along occlusion edges with the background, there
    are slight errors (dark pixels). These dark pixels are  approximation error
    of the rasterize-then-splat algorithm and can be controlled with the
    extra_accumulation_epsilon parameter.
    """
    perspective = test_utils.make_perspective_matrix()
    look_at = test_utils.make_look_at_matrix('view_1')
    projection = transforms.hi_prec_matmul(perspective, look_at)

    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('sphere.obj'))

    # Make a spike from a single vertex
    offset_amount = jnp.array([2.0])

    @jax.jit
    def loss_fn(vertices, triangles, projection, offset_amount):
      vertices = jnp.concatenate(
          (vertices[:32, :], vertices[32:33, :] * offset_amount,
           vertices[33:, :]),
          axis=0)

      rendered = render.render_triangles(
          vertices, {'colors': jnp.ones((vertices.shape[0], 4))}, triangles,
          projection, test_utils.IMAGE_WIDTH, test_utils.IMAGE_HEIGHT,
          lambda x: x['colors'])
      return rendered

    # evaluate jacobian vector product and get tangents
    rendered, jacobian = jax.jvp(
        functools.partial(loss_fn, vertices, triangles, projection),
        (offset_amount,), (jnp.array([1.0]),))
    normalized_jacobian = (jacobian / jnp.amax(jnp.abs(jacobian))) * 0.5 + 0.5

    visualized = normalized_jacobian * 0.95 + rendered * 0.05

    image = compare_images.get_pil_formatted_image(
        np.array(visualized[:, :, :3]))
    baseline_image_path = test_utils.make_resource_path('Spike_Derivative.png')
    compare_images.expect_image_file_and_image_are_near(
        self,
        baseline_image_path,
        image,
        'Spike_Derivative.png',
        'Spike derivative does not match.',
        pixel_error_threshold=0.0)

  @parameterized.named_parameters(
      (
          'ycb_toy_airplane',
          'ycb_toy_airplane.obj',
          'ycb_toy_airplane_texture.png',
          'Toy_Airplane_Textured.png',
      ),
      (
          'spot',
          'spot_triangulated.obj',
          'spot_texture.png',
          'Spot_Textured.png',
      ),
  )
  def test_renders_textured_object(self, obj_name, png_name, target_image_name):
    """Renders objects with texture mapping."""
    look_at_matrix = test_utils.make_look_at_matrix(obj_name)
    perspective_matrix = test_utils.make_perspective_matrix()
    projection = transforms.hi_prec_matmul(perspective_matrix, look_at_matrix)

    vertices, triangles = test_utils.load_test_obj(obj_name)
    positions = vertices[:, :3]

    uvs = vertices[:, 3:5]
    if vertices.shape[1] > 5:
      normals = vertices[:, 5:]
    else:
      normals = mesh.compute_vertex_normals(positions, jnp.array(triangles))
    attributes = {'uvs': jnp.array(uvs), 'normals': jnp.array(normals)}

    def shade(rasterized):
      texture_path = test_utils.make_resource_path(png_name)
      textured = shaders.texture_map(rasterized['uvs'], texture_path)
      lit = shaders.diffuse_light(textured, rasterized['normals'])
      alpha = (rasterized['uvs'][..., 0:1] > -1.0).astype(jnp.float32)
      lit_rgba = jnp.concatenate((lit, alpha), axis=-1)
      return lit_rgba

    rendered = render.render_triangles(positions, attributes, triangles,  # pytype: disable=wrong-arg-types  # numpy-scalars
                                       projection, test_utils.IMAGE_WIDTH,
                                       test_utils.IMAGE_HEIGHT, shade)

    rgb_image = compare_images.get_pil_formatted_image(
        np.array(rendered)[:, :, :3])
    blurred_target_image = load_and_blur_image(target_image_name, 0.5)
    compare_images.expect_images_are_near_and_save_comparison(
        self, blurred_target_image, rgb_image, target_image_name,
        'rendered result and target differ')


if __name__ == '__main__':
  absltest.main()
