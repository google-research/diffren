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

"""Tests for Diffren transformation utilities."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp
import numpy as np
import transformations as xf


class MathTest(chex.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-batched xyz', False, False), ('batched xyz', True, False),
      ('non-batched xyzw', False, True), ('batched xyzw', True, True))
  def test_transform_homogeneous_shapes(self, do_batched, do_xyzw):
    num_vertices = 10
    batch_size = 3
    num_channels = 4 if do_xyzw else 3
    vertices_shape = ([batch_size, num_vertices, num_channels]
                      if do_batched else [num_vertices, num_channels])
    vertices = jnp.ones(vertices_shape, dtype=jnp.float32)
    matrices = jnp.eye(4, dtype=jnp.float32)
    if do_batched:
      matrices = jnp.tile(matrices[jnp.newaxis, ...], [batch_size, 1, 1])

    transformed = transforms.transform_homogeneous(matrices, vertices)

    expected_shape = ((batch_size, num_vertices, 4) if do_batched else
                      (num_vertices, 4))
    self.assertEqual(transformed.shape, expected_shape)

  # TODO(fcole): change to an OSS compatible baseline implementation.
  # The external version of transformations.py has a different convention.

  @parameterized.named_parameters(
      ('identity', [0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]]),
      ('roll', [1.0, 0.0, 0.0
               ], [[1.0, 0.0, 0.0], [0.0, np.cos(1.0), -np.sin(1.0)],
                   [0.0, np.sin(1.0), np.cos(1.0)]]),
      ('pitch', [0.0, 1.0, 0.0
                ], [[np.cos(1.0), 0.0, np.sin(1.0)], [0.0, 1.0, 0.0],
                    [-np.sin(1.0), 0.0, np.cos(1.0)]]),
      ('yaw', [0.0, 0.0, 1.0], [[np.cos(1.0), -np.sin(1.0), 0.0],
                                [np.sin(1.0), np.cos(1.0), 0.0],
                                [0.0, 0.0, 1.0]]),
      ('all', [0.1, 0.2, 0.3
              ], [[0.936293363584199, -0.275095847318244, 0.218350663146334],
                  [0.289629477625516, 0.956425085849232, -0.036957013524625],
                  [-0.198669330795061, 0.097843395007256, 0.975170327201816]]),
  )
  def test_roll_pitch_yaw(self, angles, target_rotation):
    rotation = transforms.roll_pitch_yaw_to_rotation_matrices(angles)
    np.testing.assert_allclose(rotation, target_rotation)

  def test_make_affine(self):
    bs = 2
    linear = jnp.array(np.random.normal(size=[bs, 3, 3]))
    translation = jnp.array(np.random.normal(size=[bs, 3]))
    affine = np.array(transforms.make_affine(linear, translation))
    np.testing.assert_equal(affine[:, :, 0:3], linear)
    np.testing.assert_equal(affine[:, :, 3], translation)

  def test_expand_transform_3x4_to_4x4(self):
    bs = 2
    transforms_3x4 = jnp.array(np.random.normal(size=[bs, 3, 4]))
    transforms_4x4 = np.array(
        transforms.expand_transform_3x4_to_4x4(transforms_3x4))
    np.testing.assert_equal(transforms_4x4[:, 0:3, :], transforms_3x4)
    dtype = transforms_4x4.dtype
    np.testing.assert_equal(transforms_4x4[:, 3, 0:3],
                            np.zeros([bs, 3], dtype=dtype))
    np.testing.assert_equal(transforms_4x4[:, 3, 3], np.ones([bs], dtype=dtype))

  def test_expand_transform_3to4(self):
    bs = 2
    transforms_3x3 = jnp.array(np.random.normal(size=[bs, 3, 3]))
    transforms_4x4 = np.array(transforms.expand_transforms_3to4(transforms_3x3))
    np.testing.assert_equal(transforms_4x4[:, 0:3, 0:3], transforms_3x3)
    dtype = transforms_4x4.dtype
    np.testing.assert_equal(transforms_4x4[:, 3, 0:3],
                            np.zeros([bs, 3], dtype=dtype))
    np.testing.assert_equal(transforms_4x4[:, 0:3, 3],
                            np.zeros([bs, 3], dtype=dtype))
    np.testing.assert_equal(transforms_4x4[:, 3, 3], np.ones([bs], dtype=dtype))

  @parameterized.named_parameters(('singleton batch', 1), ('5 batch', 5))
  def test_euler_matrices(self, batch_size):
    random_angles = jnp.array(np.random.rand(batch_size, 3)) * math.pi * 2

    matrices = transforms.euler_matrices(random_angles)

    baseline = np.zeros([batch_size, 4, 4])
    for i in range(batch_size):
      baseline[i, :, :] = xf.euler_matrix(random_angles[i, 0],
                                          random_angles[i, 1], random_angles[i,
                                                                             2])

    np.testing.assert_allclose(baseline, matrices, atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
