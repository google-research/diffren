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

"""Utilities for computing quantities on meshes."""

from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp


def compute_vertex_normals(vertices, triangles):
  """Computes area-weighted vertex normals for a triangle mesh.

  This method computes vertex normals as the weighted sum of the adjacent
  face normals, weighted by the area of the face.

  Args:
    vertices: [vertex_count, 3] Array of vertex positions.
    triangles: [face_count, 3] integer Array of triangle indices.

  Returns:
    [vertex_count, 3] Array of normalized vertex normals.
  """
  corner_verts = vertices[triangles, :]
  edge_u = corner_verts[:, 1] - corner_verts[:, 0]
  edge_v = corner_verts[:, 2] - corner_verts[:, 0]
  triangle_normals = jax.vmap(jnp.cross)(edge_u, edge_v)

  # Now construct the vertex normals by scattering the face normals into the
  # vertex indices. index_add/update will sum scattered points where they
  # collide, which is what we want.
  corners_scattered = jnp.zeros_like(vertices)
  corners_scattered = [
      corners_scattered.at[triangles[:, i]].add(triangle_normals)
      for i in range(3)
  ]
  vertex_normals = (
      corners_scattered[0] + corners_scattered[1] + corners_scattered[2])
  normalized_normals = transforms.l2_normalize(vertex_normals, axis=1)
  return normalized_normals


def scatter_points(vertices, attributes, triangles, num_points, rng=None):
  """Uniformly scatters points on a mesh and interpolates attributes.

  Args:
    vertices: [vertex_count, 3] Array of vertex positions.
    attributes: dictionary of [vertex_count, k] Arrays of vertex attributes.
      Attributes may have any number of dimensions k.
    triangles: [face_count, 3] integer Array of triangle indices.
    num_points: integer specifying number of points to scatter.
    rng: JAX random number key. May be None.

  Returns:
    [num_points, 3] Array of xyz positions and a dictionary of [num_points, 3]
      Arrays of interpolated attributes.
  """
  if rng is None:
    rng = jax.random.PRNGKey(20201212)

  verts_a = vertices[triangles[:, 0], :]
  verts_b = vertices[triangles[:, 1], :]
  verts_c = vertices[triangles[:, 2], :]

  edge_ab = verts_b - verts_a
  edge_ac = verts_c - verts_a

  # face_areas is actually double the face area, but it doesn't matter because
  # we normalize to get a PDF.
  face_areas = jnp.linalg.norm(jnp.cross(edge_ab, edge_ac), axis=1)
  face_pdf = face_areas / jnp.sum(face_areas)

  # Sample num_points face indices proportionally to the face areas.
  random_faces = jax.random.choice(
      rng, triangles.shape[0], shape=(num_points,), p=face_pdf)

  # Pick two random barycentric coordinates in [0.0, 1.0).
  random_barys = jax.random.uniform(rng, (num_points, 2))

  # Uniformly sampling two barycentric coordinates puts half the samples outside
  # the triangle. To put those samples back in the triangle, we flip both
  # barycentrics when their sum is > 1.0.
  random_barys = jnp.where(
      jnp.sum(random_barys, axis=1, keepdims=True) > 1.0, 1.0 - random_barys,
      random_barys)

  def construct_samples(a, e_ab, e_ac):
    """Interpolates samples using random_faces and random_barys."""
    return (a[random_faces, :] + e_ab[random_faces, :] * random_barys[:, 0:1] +
            e_ac[random_faces, :] * random_barys[:, 1:2])

  xyz_samples = construct_samples(verts_a, edge_ab, edge_ac)

  attr_samples = {}
  for key, attr in attributes.items():
    attr_a = attr[triangles[:, 0], :]
    attr_b = attr[triangles[:, 1], :]
    attr_c = attr[triangles[:, 2], :]
    attr_samples[key] = construct_samples(attr_a, attr_b - attr_a,
                                          attr_c - attr_a)

  return xyz_samples, attr_samples
