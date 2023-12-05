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

"""Utility functions for creating an manipulating depth maps."""

from diffren.jax.utils import mesh
import einops
import jax.numpy as jnp


def create_triangles(image_width, image_height, quad_mask=None):
  """Creates triangle topology for an image-sized quad mesh, optionally masked.

  The vertices of the mesh are assumed to lie at the pixel centers of the image.
  The triangles form quads that connect four neighboring pixel centers. With no
  masking, there are 2*(image_width-1)*(image_height-1) triangles.

  If quad_mask is specified, it must be an (image_height-1, image_width-1)
  boolean mask image. Only triangles that lie inside this mask will be returned.

  Args:
    image_width: image width (number of columns)
    image_height: image height (number of rows)
    quad_mask: an (image_height-1, image_width-1) boolean array defining which
      quads to keep. May be None.

  Returns:
    a (num_triangles, 3) array of triangle indices.
  """
  w, h = image_width, image_height
  x, y = jnp.meshgrid(jnp.arange(w - 1), jnp.arange(h - 1))
  bl = y * w + x
  br = y * w + x + 1
  tl = (y + 1) * w + x
  tr = (y + 1) * w + x + 1

  triangles = jnp.array([bl, br, tl, br, tr, tl])
  # Move triangle indices to the last channel.
  triangles = einops.rearrange(triangles, 'i h w -> h w i')
  # Separate the two triangles of each quad and leave one triangle per row.
  triangles = einops.rearrange(triangles, 'h w (x y) -> (h w x) y', x=2, y=3)

  if quad_mask is None:
    return triangles

  # Remove triangles that lie outside of the quad mask.
  triangle_ids = jnp.arange(triangles.shape[0])
  quad_indices = triangle_ids // 2

  keep_triangle = quad_mask.flatten()[quad_indices]
  triangles = triangles[keep_triangle, :]

  return triangles


def create_mesh_from_positions(
    xyz_map,
    quad_mask=None,
    uv_origin='lower_left',
):
  """Creates a triangle mesh from an image of vertex positions, possibly masked.

  Given an image where pixels define positions in space (in any dimension),
  constructs a triangle mesh where the vertices lie at the positions defined
  by the pixel centers, and the triangles define quads that connect four
  neighboring pixel centers.

  If quad_mask is specified, it must be an (image_height-1, image_width-1)
  boolean mask image. The mesh will only contain vertices and triangles that
  lie in this mask.

  Args:
    xyz_map: a (image_height, image_width, c) array of positions.
    quad_mask: an (image_height-1, image_width-1) boolean array defining which
      quads to keep. May be None.
    uv_origin: a string defining the UV origin. If 'lower_left', the output
      texture coordinates assume a texture specified with lower-left origin (the
      OpenGL convention). If 'upper_left', the origin is upper-left (the OpenCV
      convention).

  Returns:
    a (num_vertices, c) array of vertex positions, where c is the dimension of
    the input xyz_map, a (num_vertices, 2) array of texture coordinates, and a
    (num_triangles, 3) array of triangle indices.
  """
  w, h = (xyz_map.shape[1], xyz_map.shape[0])
  xyz_map = jnp.array(xyz_map)
  quad_mask = jnp.array(quad_mask).astype(bool)

  verts = einops.rearrange(xyz_map, 'h w c -> (h w) c')

  uv = jnp.stack(jnp.meshgrid(jnp.arange(w), jnp.arange(h)), axis=-1)
  uv = (uv + 0.5) / jnp.array((w, h))
  uv = einops.rearrange(uv, 'h w c -> (h w) c')
  if uv_origin == 'lower_left':
    uv = uv.at[:, 1].set(1.0 - uv[:, 1])
  elif uv_origin != 'upper_left':
    raise ValueError(
        f"uv_origin must be 'lower_left' or 'upper_left', but got {uv_origin}"
    )

  faces = create_triangles(w, h, quad_mask)

  if quad_mask is not None:
    verts, uv_dict, faces = mesh.remove_unused_vertices(
        verts, {'uv': uv}, faces
    )
    uv = uv_dict['uv']

  return verts, uv, faces
