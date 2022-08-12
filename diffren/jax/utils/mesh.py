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
