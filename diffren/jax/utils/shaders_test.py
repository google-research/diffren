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

"""Tests for Diffren shader utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import compare_images
from diffren.common import obj_loader
from diffren.common import test_utils
from diffren.jax import camera
from diffren.jax import constants
from diffren.jax import render
from diffren.jax.utils import mesh
from diffren.jax.utils import shaders
from diffren.jax.utils import transforms
import jax.numpy as jnp
import numpy as np


class ShadersTest(chex.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ycb_toy_airplane', 'ycb_toy_airplane.obj', 1.0,
       'ycb_toy_airplane_texture.png', (0.2, 0.3, 0.5),
       (0.0, 1.0, 0.0), 'Toy_Airplane_Textured.png', 'diffuse'),
      ('spot', 'spot_triangulated.obj', 0.2, 'spot_texture.png',
       (0.2, 0.3, -0.5), (0.0, -1.0, 0.0), 'Spot_Textured.png', 'diffuse'),
      ('spot_point_light', 'spot_triangulated.obj', 0.2, 'spot_texture.png',
       (0.2, 0.3, -0.5),
       (0.0, -1.0, 0.0), 'Spot_Textured_Pt_Light.png', 'point'),
      ('spot_phong_point', 'spot_triangulated.obj', 0.2, 'spot_texture.png',
       (0.2, 0.3, -0.5),
       (0.0, -1.0, 0.0), 'Spot_Textured_Phong_Point.png', 'phong_point'),
  )
  def test_renders_textured_object(self, obj_name, obj_scale, png_name,
                                   eye_position, eye_up, target_image_name,
                                   lighting_type):
    """Renders an objects with texture mapping."""
    look_at_matrix = camera.look_at(eye_position, (0.0, 0.0, 0.0), eye_up)
    perspective_matrix = test_utils.make_perspective_matrix()
    projection = transforms.hi_prec_matmul(perspective_matrix, look_at_matrix)

    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path(obj_name))

    positions = vertices[:, :3] * obj_scale
    uvs = vertices[:, 3:5]
    if vertices.shape[1] > 5:
      normals = vertices[:, 5:]
    else:
      normals = mesh.compute_vertex_normals(positions, jnp.array(triangles))

    texture_path = test_utils.make_resource_path(png_name)

    def shader(attrs):
      textured = shaders.texture_map(attrs['uvs'], texture_path)

      if lighting_type == 'diffuse':
        return shaders.diffuse_light(textured, attrs['normals'])
      elif lighting_type == 'point':
        return shaders.point_light(textured, attrs['normals'],
                                   attrs['positions'],
                                   jnp.array(eye_position, dtype=jnp.float32))
      elif lighting_type == 'phong_point':
        return shaders.point_light(
            textured,
            attrs['normals'],
            attrs['positions'],
            jnp.array(eye_position, dtype=jnp.float32),
            eye_position=eye_position,
            ks=1.0,
            ns=50.0)

    rendered = render.render_triangles(
        positions, {
            'uvs': uvs,
            'positions': positions,
            'normals': normals
        },
        triangles,
        projection,
        test_utils.IMAGE_WIDTH,
        test_utils.IMAGE_HEIGHT,
        shading_function=shader,
        compositing_mode=constants.CompositingMode.OVER)

    image = compare_images.get_pil_formatted_image(np.asarray(rendered))
    baseline_image_path = test_utils.make_resource_path(target_image_name)
    compare_images.expect_image_file_and_image_are_near(
        self, baseline_image_path, image, target_image_name,
        '%s does not match.' % target_image_name)


if __name__ == '__main__':
  absltest.main()
