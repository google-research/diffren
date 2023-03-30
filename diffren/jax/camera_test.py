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

"""Tests for Diffren camera routines."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import test_utils
from diffren.jax import camera
import jax
import jax.numpy as jnp
import numpy as np


class CameraTest(chex.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('with_vmap', True), ('without_vmap', False))
  def test_perspective(self, use_vmap):
    fov_y = jnp.array([40.0, 60.0])
    aspect = 640 / 480
    near = jnp.array([0.01, 0.1])
    far = jnp.array([10.0, 20.0])
    if use_vmap:
      result = jax.vmap(
          camera.perspective, in_axes=(None, 0, 0, 0))(aspect, fov_y, near, far)
    else:
      result = []
      for i in range(2):
        result.append(camera.perspective(aspect, fov_y[i], near[i], far[i]))
      result = jnp.stack(result, axis=0)

    expected = np.stack(
        [test_utils.make_perspective_matrix(i) for i in range(len(fov_y))],
        axis=0)

    np.testing.assert_allclose(expected, result, rtol=1e-06)

  def test_perspective_from_intrinsics_center_of_projection(self):
    matrix = camera.perspective_from_intrinsics(
        focal_x=100.0,
        focal_y=100.0,
        center_offset_x=50.0,
        center_offset_y=25.0,
        near_clip=10.0,
        far_clip=100.0,
        image_width=100,
        image_height=50)
    # pyformat: disable
    vertices = jnp.array([[0.0, 0.0, -100.0, 1.0],
                          [-50.0, -25.0, -100.0, 1.0],
                          [-50.0, -50.0, -100.0, 1.0],
                          [-50.0, -50.0, -100.0, 1.0],
                          [-5.0, -5.0, -10.0, 1.0]])
    # pyformat: enable
    projected = jnp.matmul(vertices, jnp.transpose(matrix))
    result = projected[:, :3] / projected[:, 3:4]

    # pyformat: disable
    np.testing.assert_allclose(
        result,
        np.array([[1.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 1.0],
                  [0.0, -1.0, 1.0],
                  [0.0, -1.0, -1.0]]),
        rtol=1e-06)
    # pyformat: enable

  @chex.variants(with_jit=True, without_jit=True)
  def test_ortho(self):

    @self.variant
    def ortho_transform():
      matrix = camera.ortho(
          left=-50.0, right=50.0, bottom=-25.0, top=25.0, near=10.0, far=100.0)
      vertex = jnp.array([50.0, 25.0, -100.0, 1.0])
      return jnp.matmul(matrix, vertex)

    np.testing.assert_allclose(
        self.variant(ortho_transform)(),
        np.array([1.0, 1.0, 1.0, 1.0]),
        rtol=1e-06)

  @parameterized.parameters((
      'up and gaze are close',
      (0.0, 0.0, 1.0),
      (0.0, 0.0, 0.0),
      (0.0, 0.0, 1.0),
  ), (
      'eye and center are close',
      (0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
  ))
  def test_degenerate_look_at(self, msg, eye, center, up):
    with self.assertRaisesRegex(ValueError, msg):
      _ = camera.look_at(eye, center, up)

    _ = jax.jit(camera.look_at)(eye, center, up)


if __name__ == '__main__':
  absltest.main()
