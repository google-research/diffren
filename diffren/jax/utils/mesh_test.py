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

"""Tests for Diffren mesh utilities."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import obj_loader
from diffren.common import test_utils
from diffren.jax.utils import mesh
import jax
import jax.numpy as jnp
import numpy as np


class MeshTest(chex.TestCase, parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test_compute_vertex_normals_unbatched(self):
    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('sphere.obj'))

    # The normals for a unit sphere are the same as the vertex positions.
    expected_normals = vertices

    @self.variant
    def compute_normals():
      return mesh.compute_vertex_normals(
          jnp.array(vertices), jnp.array(triangles))

    np.testing.assert_array_almost_equal(
        expected_normals, self.variant(compute_normals)(), decimal=2)

  def test_compute_vertex_normals(self):
    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('sphere.obj'))
    stretched_vertices = np.copy(vertices)
    stretched_vertices[:, 2] *= 2.0

    # The normals for a unit sphere are the same as the vertex positions.
    expected_normals = vertices
    # Stretching the sphere by 2x in Z should halve the Z component of
    # the normals:
    expected_stretched_normals = np.copy(vertices)
    expected_stretched_normals[:, 2] *= 0.5
    expected_stretched_normals /= np.linalg.norm(
        expected_stretched_normals, axis=1, keepdims=True)

    batched_vertices = np.stack([vertices, stretched_vertices])

    batched_normals = jax.vmap(
        functools.partial(mesh.compute_vertex_normals, triangles=triangles))(
            batched_vertices)

    np.testing.assert_array_almost_equal(
        expected_normals, batched_normals[0, ...], decimal=2)
    np.testing.assert_array_almost_equal(
        expected_stretched_normals, batched_normals[1, ...], decimal=2)


if __name__ == '__main__':
  absltest.main()
