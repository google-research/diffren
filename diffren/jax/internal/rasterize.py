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

"""Functions to rasterize geometry into framebuffers."""

from typing import Optional

from diffren.jax import constants
from diffren.jax.internal import framebuffer as fb
from diffren.jax.internal.kernels import rasterize_triangles_xla
import jax
import jax.numpy as jnp


def rasterize_triangles(
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
    camera: Optional[jnp.ndarray],
    image_width: int,
    image_height: int,
    num_layers=1,
    face_culling_mode=constants.FaceCullingMode.NONE) -> fb.Framebuffer:
  """Rasterizes triangles and computes per-pixel ids and barycentric coords.

  If num_layers > 1, the layers are returned in front-to-back order (layer 0 is
  closest to the camera).

  None of the outputs of this function are differentiable.

  Args:
    vertices: float32 array of xyz positions with shape [vertex_count, d]. If
      projection_matrix is specified, d may be 3 or 4. If camera is None, d must
      be 4 and the values are assumed to be xyzw homogenous coordinates.
    triangles: int32 array with shape [triangle_count, 3].
    camera: float array with shape [4, 4] containing a model-view-perspective
      projection matrix (later, this will optionally be a camera with distortion
      coefficients). May be None. If None, vertices are assumed to be 4D
      homogenous clip-space coordinates.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    num_layers: Number of depth layers to render.
    face_culling_mode: one of FaceCullingMode. Defaults to NONE.

  Returns:
    A Framebuffer containing the rasterized values: barycentrics, triangle_id,
    foreground_mask, vertex_ids with shape [num_layers, height, width, channels]
    See framebuffer.py for a description of the Framebuffer fields.

  Raises:
    ValueError: if an argument has an invalid value or shape.
  """

  rank = lambda t: len(t.shape)

  vertices = jnp.array(vertices)
  if rank(vertices) != 2 or vertices.shape[-1] not in (3, 4):
    raise ValueError(
        f'vertices must have shape [n, 3] or [n, 4], but found {vertices.shape}'
    )

  triangles = jnp.array(triangles)
  if rank(triangles) != 2 or triangles.shape[-1] != 3:
    raise ValueError(
        f'triangles must have shape [n, 3], but found {triangles.shape}.')

  if camera is not None and camera.shape != (4, 4):
    raise ValueError(f'camera must be a (4, 4) matrix but found {camera.shape}')

  if not image_width > 0:
    raise ValueError(f'image_width must be > 0 but is {image_width}.')
  if not image_height > 0:
    raise ValueError(f'image_height must be > 0 but is {image_height}.')
  if not num_layers > 0:
    raise ValueError(f'num_layers must be > 0 but is {num_layers}.')

  if camera is not None:
    if vertices.shape[-1] == 3:
      vertices = jnp.pad(vertices, ((0, 0), (0, 1)), constant_values=1)
    # Apply the camera transformation using HIGHEST precision to get float32
    # accuracy on TPU.
    vertices = jnp.matmul(
        vertices, jnp.transpose(camera), precision=jax.lax.Precision.HIGHEST)

  triangle_id, z_buffer, barycentrics = rasterize_triangles_xla.rasterize_triangles(
      vertices, triangles, image_width, image_height, num_layers,
      face_culling_mode)

  mask = (z_buffer != 1.0).astype(jnp.float32)

  vertex_ids = triangles[triangle_id[:], :]
  vertex_ids = jnp.reshape(vertex_ids,
                           (num_layers, image_height, image_width, 3))

  return fb.Framebuffer(
      foreground_mask=jnp.expand_dims(mask, axis=-1),
      triangle_id=jnp.expand_dims(triangle_id, axis=-1),
      vertex_ids=vertex_ids,
      barycentrics=barycentrics)
