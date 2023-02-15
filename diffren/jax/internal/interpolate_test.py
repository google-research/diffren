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

"""Tests for Diffren interpolation routines."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import test_utils
from diffren.jax import constants
from diffren.jax import render
import jax
import jax.numpy as jnp
import numpy as np


class InterpolateTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super(InterpolateTest, self).setUp()

    self.cube_vertex_positions = jnp.array(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=jnp.float32)
    self.cube_triangles = jnp.array(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=jnp.int32)

    perspective = test_utils.make_perspective_matrix()
    projection_1 = jnp.matmul(
        perspective, test_utils.make_look_at_matrix('view_1')
    )
    projection_2 = jnp.matmul(
        perspective, test_utils.make_look_at_matrix('view_2')
    )
    self.projection = jnp.stack([projection_1, projection_2], axis=0)

  @parameterized.parameters([1, 2, 3])
  def test_renders_colored_cube(self, num_layers):
    """Renders a simple colored cube in two viewpoints with multiple layers."""

    vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
    vertex_rgba = jnp.pad(vertex_rgb, (0, 1), constant_values=1.0)

    def render_cube(projection):
      return render.render_triangles(
          self.cube_vertex_positions, {'rgba': vertex_rgba},
          self.cube_triangles,
          projection,
          test_utils.IMAGE_WIDTH,
          test_utils.IMAGE_HEIGHT,
          num_layers=num_layers,
          shading_function=lambda x: x['rgba'],
          face_culling_mode=constants.FaceCullingMode.NONE,
          compositing_mode=constants.CompositingMode.NONE)

    rendered = jax.vmap(render_cube)(self.projection)

    for i in (0, 1):
      for l in range(num_layers):
        test_utils.check_image(self, np.array(rendered[i, l, :, :, :]),
                               'Unlit_Cube_{}_{}.png'.format(i, l))


if __name__ == '__main__':
  absltest.main()
