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


def rasterize_triangles(vertices: jnp.ndarray,
                        triangles: jnp.ndarray,
                        camera: Optional[jnp.ndarray],
                        image_width: int,
                        image_height: int,
                        num_layers=1,
                        face_culling_mode=constants.FaceCullingMode.NONE,
                        compute_diff_barys=False) -> fb.Framebuffer:
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
    compute_diff_barys: bool specifying whether to compute differentiable
      barycentric coordinates. Defaults to False.

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

  output = fb.Framebuffer(
      foreground_mask=jnp.expand_dims(mask, axis=-1),
      triangle_id=jnp.expand_dims(triangle_id, axis=-1),
      vertex_ids=vertex_ids,
      barycentrics=barycentrics)

  if compute_diff_barys:
    output = differentiable_barycentrics(output, vertices, triangles)

  return output


def differentiable_barycentrics(framebuffer: fb.Framebuffer,
                                clip_space_vertices: jnp.ndarray,
                                triangles: jnp.ndarray) -> fb.Framebuffer:
  """Computes differentiable barycentric coordinates from a Framebuffer.

  The barycentric coordinates will be differentiable w.r.t. the input vertices.
  Later, we may support derivatives w.r.t. pixel position for mip-mapping.

  Args:
    framebuffer: a multi-layer Framebuffer containing triangle ids and a
      foreground mask.
    clip_space_vertices: a float32 array with shape [vertex_count, 4].
    triangles: a int32 array with shape [triangle_count, 3].

  Returns:
    a Framebuffer with the same values as the input, but with the barycentric
    coordinates replaced with differentiable values.
  """
  # Compute image pixel coordinates.
  px, py = normalized_pixel_coordinates(framebuffer.width, framebuffer.height)

  # Compute per-triangle inverse matrices.
  triangle_matrices = jax.vmap(
      compute_triangle_matrix, in_axes=(None, 0))(clip_space_vertices,
                                                  triangles)

  def compute_single_pixel_coordinates(triangle_id, px, py):
    triangle_matrix = triangle_matrices[triangle_id]
    return compute_barycentric_coordinates(triangle_matrix, px, py)

  full_image_compute = jax.vmap(jax.vmap(compute_single_pixel_coordinates))
  all_layer_compute = jax.vmap(full_image_compute, in_axes=(0, None, None))
  barycentric_coords = all_layer_compute(framebuffer.triangle_id[..., 0], px,
                                         py)

  # Mask out barycentrics for background pixels.
  barycentric_coords = barycentric_coords * framebuffer.foreground_mask

  return framebuffer.replace(barycentrics=barycentric_coords)


def normalized_pixel_coordinates(image_width, image_height):
  """Computes the normalized pixel coordinates for the specified image size.

  The x-coordinates will range from -1 to 1 left to right.
  The y-coordinates will range from -1 to 1 top to bottom.
  The extrema +-1 will fall onto the exterior pixel boundaries, while the
  coordinates will be evaluated at pixel centers. So, image of width 4 will have
  normalized pixel x-coordinates at [-0.75 -0.25 0.25 0.75], while image of
  width 3 will have them at [-0.667 0 0.667].

  Args:
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

  Returns:
    Two float32 arrays with shape [image_height, image_width] containing x- and
    y- coordinates, respectively, for each image pixel.
  """
  x_range = (2 * jnp.arange(image_width) + 1) / image_width - 1
  y_range = (2 * jnp.arange(image_height) + 1) / image_height - 1
  x_coords, y_coords = jnp.meshgrid(x_range, y_range)
  return x_coords, y_coords


def compute_triangle_matrix(clip_space_vertices, triangle):
  """Computes the triangle matrix used in barycentric coordinate calculation.

  The result corresponds to the inverse matrix from equation (4) in the paper
  "Triangle Scan Conversion using 2D Homogeneous Coordinates". Our matrix
  inverses are not divided by the determinant, only multiplied by its sign. The
  division happens in compute_barycentric_coordinates anyways.

  Args:
    clip_space_vertices: float32 tensor with shape [vertex_count, 4] containing
      vertex positions in clip space (x, y, z, w).
    triangle: 1-D int32 array with shape [3] containing the triangle's vertex
      indices in counter-clockwise order.

  Returns:
    2-D float32 array with shape [3, 3] containing the triangle matrix.
  """
  xyzw = clip_space_vertices[triangle, :]
  xyw = jnp.concatenate((xyzw[:, :2], xyzw[:, 3:4]), axis=-1)
  xyw = jnp.transpose(xyw)

  # Compute the sub-determinants.
  d11 = xyw[1, 1] * xyw[2, 2] - xyw[1, 2] * xyw[2, 1]
  d21 = xyw[1, 2] * xyw[2, 0] - xyw[1, 0] * xyw[2, 2]
  d31 = xyw[1, 0] * xyw[2, 1] - xyw[1, 1] * xyw[2, 0]
  d12 = xyw[2, 1] * xyw[0, 2] - xyw[2, 2] * xyw[0, 1]
  d22 = xyw[2, 2] * xyw[0, 0] - xyw[2, 0] * xyw[0, 2]
  d32 = xyw[2, 0] * xyw[0, 1] - xyw[2, 1] * xyw[0, 0]
  d13 = xyw[0, 1] * xyw[1, 2] - xyw[0, 2] * xyw[1, 1]
  d23 = xyw[0, 2] * xyw[1, 0] - xyw[0, 0] * xyw[1, 2]
  d33 = xyw[0, 0] * xyw[1, 1] - xyw[0, 1] * xyw[1, 0]
  matrix = jnp.array([[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]])
  # Multiply by the sign of the determinant while avoiding multiplying by zero.
  determinant = xyw[0, 0] * d11 + xyw[1, 0] * d12 + xyw[2, 0] * d13
  sign = jnp.sign(determinant) + (determinant == 0).astype(jnp.float32)
  matrix = sign * matrix
  return matrix


def compute_barycentric_coordinates(triangle_matrix, px, py):
  """Computes barycentric coordinates at a pixel.

  Args:
    triangle_matrix: 2-D float32 array with shape [3, 3] containing the triangle
      matrix computed by compute_triangle_matrix.
    px: 0-D float32 pixel x-coordinate, as computed by
      normalized_pixel_coordinates.
    py: 0-D float32 pixel y-coordinates, as computed by
      normalized_pixel_coordinates.

  Returns:
    1-D float32 array with shape [3] containing the barycentric
    coordinates of the point within the triangle.
  """
  barycentric_coords = (
      triangle_matrix[:, 0] * px + triangle_matrix[:, 1] * py +
      triangle_matrix[:, 2])
  # Normalize so the barycentric coordinates sum to 1. Guard against division
  # by zero in the case that the barycentrics sum to zero, which can happen for
  # background pixels when the 0th triangle in the list is degenerate, due to
  # the way we use triangle id 0 for both background and the first triangle.
  # TODO(fcole): fix so that id 0 is not both background and first triangle.
  barycentric_coords = jnp.nan_to_num(barycentric_coords /
                                      jnp.sum(barycentric_coords))
  return barycentric_coords
