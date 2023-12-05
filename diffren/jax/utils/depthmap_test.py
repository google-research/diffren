# Copyright 2024 The diffren Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import compare_images
from diffren.common import test_utils
from diffren.jax import constants
from diffren.jax import render
from diffren.jax.utils import depthmap
from diffren.jax.utils import shaders
from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp
import numpy as np


def reproject_depthmap(map_verts, map_uv, map_faces, projection, texture):
  return render.render_triangles(
      map_verts,
      {'uv': map_uv},
      map_faces,
      projection,
      texture.shape[1],
      texture.shape[0],
      shading_function=lambda x: shaders.texture_map(x['uv'], texture),
      compositing_mode=constants.CompositingMode.OVER,
  )


class DepthmapTest(chex.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('lower_left', 'full_res'),
      ('upper_left', 'full_res'),
      ('upper_left', 'one_tenth_res'),
  )
  def test_full_screen_xyz_map_matches_image(self, origin, resolution):
    clip_space_quad_verts = jnp.array((
        (-1.0, -1.0, 0.0, 1.0),
        (1.0, -1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (-1.0, 1.0, 0.0, 1.0),
    ))
    quad_faces = jnp.array(((0, 1, 2), (0, 2, 3)))

    spot_image = shaders.load_texture(
        test_utils.make_resource_path('spot_global_illumination.png'),
        lower_left_origin=False,
    )

    spot_texture = (
        spot_image if origin == 'upper_left' else jnp.flipud(spot_image)
    )

    xyz = render.render_triangles(
        clip_space_quad_verts,
        {'xyzw': clip_space_quad_verts},
        quad_faces,
        None,
        spot_texture.shape[1],
        spot_texture.shape[0],
        shading_function=lambda x: x['xyzw'],
        compositing_mode=constants.CompositingMode.OVER,
    )

    if resolution == 'one_tenth_res':
      xyz = jax.image.resize(
          xyz, (int(xyz.shape[0] * 0.1), int(xyz.shape[1] * 0.1), 4), 'bilinear'
      )

    map_verts, map_uv, map_faces = depthmap.create_mesh_from_positions(
        xyz, uv_origin=origin
    )

    reprojected = reproject_depthmap(
        map_verts, map_uv, map_faces, None, spot_texture
    )

    output_rgb = compare_images.get_pil_formatted_image(np.asarray(reprojected))
    baseline_rgb = compare_images.get_pil_formatted_image(
        np.asarray(spot_image)
    )

    if resolution == 'one_tenth_res':
      # Due to the tessellation of the depth map, at one-tength resolution
      # there is a 10/2 + 1 = 6 pixel black border around the result image.
      baseline_rgb[:6, :, :3] = 0
      baseline_rgb[:, :6, :3] = 0
      baseline_rgb[-6:, :, :3] = 0
      baseline_rgb[:, -6:, :3] = 0

    compare_images.expect_images_are_near_and_save_comparison(
        self,
        baseline_rgb,
        output_rgb,
        f'spot_full_screen_quad_{origin}_{resolution}',
        'spot full screen quad reprojection does not match.',
    )

  def test_xyz_map_looks_like_mesh_from_same_camera(self):
    look_at_matrix = test_utils.make_look_at_matrix('spot')
    perspective_matrix = test_utils.make_perspective_matrix()
    projection = transforms.hi_prec_matmul(perspective_matrix, look_at_matrix)

    vertices, triangles = test_utils.load_test_obj('spot_triangulated.obj')
    positions = vertices[:, :3]

    xyza = render.render_triangles(
        positions,
        {
            'xyza': jnp.concatenate(
                (positions, jnp.ones_like(positions[..., 0:1])), axis=-1
            )
        },
        triangles,
        projection,
        test_utils.IMAGE_WIDTH,
        test_utils.IMAGE_HEIGHT,
        shading_function=lambda x: x['xyza'],
        compositing_mode=constants.CompositingMode.OVER,
    )

    xyz = xyza[..., :3]
    quad_mask = xyza[:-1, :-1, 3]

    map_verts, map_uv, map_faces = depthmap.create_mesh_from_positions(
        xyz, quad_mask
    )

    spot_texture = shaders.load_texture(
        test_utils.make_resource_path('Spot_Textured.png')
    )
    reprojected = reproject_depthmap(
        map_verts, map_uv, map_faces, projection, spot_texture
    )

    output_rgb = compare_images.get_pil_formatted_image(
        np.asarray(reprojected[..., :3])
    )
    compare_images.expect_image_file_and_image_are_near(
        self,
        test_utils.make_resource_path('Spot_Textured.png'),
        output_rgb,
        'spot_triangulated_reprojection',
        'spot reprojection does not match.',
    )


if __name__ == '__main__':
  absltest.main()
