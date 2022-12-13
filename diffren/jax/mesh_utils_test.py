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

"""Tests for diffren.jax.mesh_utils."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.common import obj_loader
from diffren.common import test_utils
from diffren.jax import mesh_utils
import jax
import jax.numpy as jnp
import numpy as np


class MeshUtilsTest(chex.TestCase, parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test_compute_vertex_normals_unbatched(self):
    path = test_utils.make_resource_path('sphere.obj')
    vertices, triangles = obj_loader.load_and_flatten_obj(path)

    # The normals for a unit sphere are the same as the vertex positions.
    expected_normals = vertices

    @self.variant
    def compute_normals():
      return mesh_utils.compute_vertex_normals(
          jnp.array(vertices), jnp.array(triangles))

    np.testing.assert_array_almost_equal(
        expected_normals, self.variant(compute_normals)(), decimal=2)

  def test_compute_vertex_normals(self):
    path = test_utils.make_resource_path('sphere.obj')
    vertices, triangles = obj_loader.load_and_flatten_obj(path)
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
        functools.partial(
            mesh_utils.compute_vertex_normals, triangles=triangles))(
                batched_vertices)

    np.testing.assert_array_almost_equal(
        expected_normals, batched_normals[0, ...], decimal=2)
    np.testing.assert_array_almost_equal(
        expected_stretched_normals, batched_normals[1, ...], decimal=2)


def test_scatter_points(_):
  path = test_utils.make_resource_path('sphere.obj')
  vertices, triangles = obj_loader.load_and_flatten_obj(path)

  # Create a dummy vertex color attribute.
  vertex_rgb = (vertices - jnp.min(vertices, axis=0)) / (
      jnp.max(vertices, axis=0) - jnp.min(vertices, axis=0))

  xyz, attr = mesh_utils.scatter_points(
      jnp.array(vertices), {'rgb': vertex_rgb}, jnp.array(triangles), 1000)
  rgb = attr['rgb']

  # Test that the points are reasonably spaced over the sphere.
  sample_center = jnp.mean(xyz, axis=0)
  sample_std = jnp.linalg.norm(xyz - sample_center, axis=1)
  rgb_center = jnp.mean(rgb, axis=0)
  np.testing.assert_almost_equal(jnp.linalg.norm(sample_center), 0.0, decimal=1)
  np.testing.assert_almost_equal(jnp.mean(sample_std), 1.0, decimal=1)
  np.testing.assert_array_almost_equal(
      rgb_center, jnp.array((0.5, 0.5, 0.5)), decimal=1)

if __name__ == '__main__':
  absltest.main()
