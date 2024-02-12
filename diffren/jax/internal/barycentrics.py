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

"""Functions to compute barycentric coordinates."""

from diffren.jax.internal import framebuffer as fb
import jax
import jax.numpy as jnp


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
