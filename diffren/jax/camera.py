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

"""Collection of JAX functions for managing 3D camera matrices."""

import math
from diffren.jax.utils import transforms
import jax
import jax.numpy as jnp


def perspective(aspect_ratio: float, fov_y: float, near_clip: float,
                far_clip: float) -> jnp.ndarray:
  """Computes a perspective transformation matrix.

  Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

  Args:
    aspect_ratio: float value specifying the image aspect ratio (width/height).
    fov_y: float value specifying output vertical field of views in degrees.
    near_clip: float value specifying near clipping plane distance.
    far_clip: float value specifying far clipping plane distance.

  Returns:
    A [4, 4] float array that maps from right-handed points in eye space to
    left-handed points in clip space.
  """
  # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
  # and adds the half-angle factor of .5.
  focal_length_y = 1.0 / jnp.tan(fov_y * (math.pi / 360.0))
  return perspective_from_intrinsics(focal_length_y / aspect_ratio,
                                     focal_length_y, 0.0, 0.0, near_clip,
                                     far_clip, 2, 2)


def perspective_from_intrinsics(focal_x: float, focal_y: float, center_offset_x,
                                center_offset_y, near_clip, far_clip,
                                image_width, image_height):
  """Computes a perspective matrix from vision-style camera intrisics.

  Follows the pattern found in:
  http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
  of breaking the transform down into a perspective matrix followed by a
  transformation to NDC space.

  Args:
    focal_x: a float value specifying the output focal length in the X dimension
      in pixels.
    focal_y: As focal_x, but for the Y dimension.
    center_offset_x: a float value specifying the offset of the center of
      projection in the X dimension in pixels. A value of 0 puts an object at
      (0,0) in camera space in the center of the image. A positive value puts an
      object at (0,0) in the right half of the image, a negative value puts it
      in the left half.
    center_offset_y: As center_offset_x, but for the Y dimension. A positive
      value puts an object at (0,0) in camera space in the top half of the
      image, a negative value in the bottom half.
    near_clip: a float value specifying near clipping plane distance.
    far_clip: a float value specifying far clipping plane distance.
    image_width: int or float specifying the width of the camera's image
    image_height: int or float specifying the height of the camera's image

  Returns:
    A [4, 4] float32 array that maps from right-handed points in camera space
    to left-handed points in clip space.
  """
  a = near_clip + far_clip
  b = near_clip * far_clip
  # pyformat: disable
  perspective_transform = jnp.array(
      [
          [focal_x, 0.0, -center_offset_x, 0.0],
          [0.0, focal_y, -center_offset_y, 0.0],
          [0.0, 0.0, a, b],
          [0.0, 0.0, -1.0, 0.0]
      ])
  # pyformat: enable
  left = image_width * -0.5
  right = image_width * 0.5
  bottom = image_height * -0.5
  top = image_height * 0.5
  ndc_transform = ortho(left, right, bottom, top, near_clip, far_clip)
  return transforms.hi_prec_matmul(ndc_transform, perspective_transform)


def ortho(left, right, bottom, top, near, far):
  """Computes an orthographic camera transformation matrix.

  Functionality mimes glOrtho (third_party/GL/gl/include/GL/gl.h).

  Args:
    left: float value specifying location of left clipping plane.
    right: float value specifying location of right clipping plane.
    bottom: float value specifying location of bottom clipping plane.
    top: float value specifying location of top clipping plane.
    near: float value specifying location of near clipping plane.
    far: float value specifying location of far clipping plane.

  Returns:
     A [4, 4] float array that maps from right-handed points in eye space to
     left-handed points in clip space.
  """
  depth_range = far - near
  # pyformat: disable
  m = jnp.array([
      [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
      [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
      [0.0, 0.0, -2.0 / (depth_range), -(far + near) / (depth_range)],
      [0.0, 0.0, 0.0, 1.0]
  ])
  # pyformat: enable
  return m


def look_at(eye, center, world_up):
  """Computes camera viewing matrices.

  Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

  Args:
    eye: 3-D float array containing the XYZ world-space position of the camera.
    center: 3-D float array containing a position along the center of the
      camera's gaze.
    world_up: 3-D float array specifying the world's up direction; the output
      camera will have no tilt with respect to this direction.

  Returns:
    A [4, 4] float array containing a right-handed camera extrinsics matrix
    that maps points from world space to points in eye space.

  Raises:
    ValueError: if the input frame is degenerate.
  """
  eye = jnp.array(eye)
  center = jnp.array(center)
  world_up = jnp.array(world_up)

  vector_degeneracy_cutoff = 1e-6
  forward = center - eye
  forward_norm = jnp.linalg.norm(forward)

  try:
    if forward_norm < vector_degeneracy_cutoff:
      raise ValueError(
          'Camera matrix is degenerate because eye and center are close.')
  except jax.errors.ConcretizationTypeError:
    pass

  forward = forward / forward_norm

  to_side = jnp.cross(forward, world_up)
  to_side_norm = jnp.linalg.norm(to_side)
  try:
    if to_side_norm < vector_degeneracy_cutoff:
      raise ValueError(
          'Camera matrix is degenerate because up and gaze are close '
          'or up is degenerate.')
  except jax.errors.ConcretizationTypeError:
    pass

  to_side = to_side / to_side_norm
  cam_up = jnp.cross(to_side, forward)

  # Make a 3x3 rotation matrix:
  view_rotation = jnp.vstack([to_side, cam_up, -forward])
  # Set the upper 3x3 of a 4x4 identity matrix:
  view_rotation = jnp.eye(4).at[:3, :3].set(view_rotation)

  # Make a 4x4 translation matrix:
  view_translation = jnp.eye(4).at[:3, 3].set(-eye)

  camera_matrix = transforms.hi_prec_matmul(view_rotation, view_translation)
  return camera_matrix
