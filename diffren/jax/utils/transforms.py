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

"""Utilities for matrix transformations."""

import jax
import jax.numpy as jnp


def hi_prec_matmul(x, y):
  """Uses HIGHEST precision for full float32 accuracy on TPU."""
  return jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST)


def l2_normalize(x, axis=None, eps=1e-12):
  """Equivalent of tf.math.l2_normalize.

  Args:
    x: A Array.
    axis: Dimension along which to normalize. A scalar or a vector of integers.
    eps: A lower bound value for the norm.

  Returns:
    A Array with the same shape as x.
  """
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def transform_homogeneous(matrices, vertices):
  """Applies 4x4 homogenous matrix transformations to xyz or xyzw vertices.

  The vertices are input and output as as row-major, but are interpreted as
  column vectors multiplied on the right-hand side of the matrices. More
  explicitly, this function computes (MV^T)^T.
  If input vertices are xyz they are extended to xyzw with w=1.

  Args:
    matrices: a [..., 4, 4] Array (or convertible value) of matrices.
    vertices: a [..., N, 3] or [..., N, 4] Array (or convertible value) of xyz
      or xyzw vertices.

  Returns:
    A [..., N, 4] Array of xyzw vertices.

  Raises:
    ValueError: if matrices or vertices have incorrect or mismatched number of
      dimensions.
  """
  matrices = jnp.array(matrices)
  vertices = jnp.array(vertices)

  if len(matrices.shape) != len(vertices.shape):
    raise ValueError(
        'length of matrices and vertices shapes must match but are {} and {}'
        .format(len(matrices.shape), len(vertices.shape)))
  if vertices.shape[-1] not in [3, 4]:
    raise ValueError(
        'vertices must be either 3-D (xyz) or 4-D (xyzw) but is {}'.format(
            vertices.shape[-1]))

  if vertices.shape[-1] == 3:
    homogeneous_coord = jnp.ones_like(vertices[..., 0:1])
    vertices = jnp.concatenate([vertices, homogeneous_coord], -1)

  matrices = jnp.swapaxes(
      matrices, axis1=matrices.ndim - 2, axis2=matrices.ndim - 1)
  return hi_prec_matmul(vertices, matrices)


def quaternion_matrix(q: jnp.array):
  """Creates a 4x4 rotation matrix from a quaternion.

  This function is based on quaternion_matrix() from transformations.py.
  It returns a 4x4 matrix for convenience with multiplying with other
  transformations.

  Note that unlike the transformations.py version, this function does
  not check if q is a zero-length quaternion.

  Args:
    q: a 4-D array representing a quaternion.

  Returns:
    a [4, 4] rotation matrix.
  """
  q = jnp.array(q)

  normalized_q = q * jnp.sqrt(2.0 / jnp.dot(q, q))
  q_mat = jnp.outer(normalized_q, normalized_q)
  return jnp.array(
      ((1.0 - q_mat[1, 1] - q_mat[2, 2], q_mat[0, 1] - q_mat[2, 3],
        q_mat[0, 2] + q_mat[1, 3], 0.0),
       (q_mat[0, 1] + q_mat[2, 3], 1.0 - q_mat[0, 0] - q_mat[2, 2],
        q_mat[1, 2] - q_mat[0, 3], 0.0),
       (q_mat[0, 2] - q_mat[1, 3], q_mat[1, 2] + q_mat[0, 3],
        1.0 - q_mat[0, 0] - q_mat[1, 1], 0.0), (0.0, 0.0, 0.0, 1.0)))


def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
  """Converts roll-pitch-yaw angles to rotation matrices.

  Args:
    roll_pitch_yaw: Array (or convertible value) with shape [..., 3]. The last
      dimension contains the roll, pitch, and yaw angles in radians.  The
      resulting matrix rotates points by first applying roll around the x-axis,
      then pitch around the y-axis, then yaw around the z-axis.

  Returns:
     Array with shape [..., 3, 3]. The 3x3 rotation matrices corresponding to
     the input roll-pitch-yaw angles.
  """
  roll_pitch_yaw = jnp.array(roll_pitch_yaw)

  cosines = jnp.cos(roll_pitch_yaw)
  sines = jnp.sin(roll_pitch_yaw)
  cx, cy, cz = cosines[..., 0], cosines[..., 1], cosines[..., 2]
  sx, sy, sz = sines[..., 0], sines[..., 1], sines[..., 2]
  # pyformat: disable
  rotation = jnp.stack(
      [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
       sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
       -sy, cy * sx, cy * cx], axis=-1)
  # pyformat: enable
  shape = rotation.shape[:-1] + (3, 3)
  rotation = jnp.reshape(rotation, shape)
  return rotation


def make_affine(linear, translation):
  """Combines 3x3 linear transforms and translations into 3x4 affine transforms.

  The resulting matrix will have the linear component on the left, with
  translation as the right column.

  Args:
    linear: a [..., 3, 3] Array (or convertible value) containing 3x3 linear
      transforms.
    translation: a [..., 3] Array (or convertible value) containing
      translations.

  Returns:
    A [..., 3, 4] Tensor of 3x4 transformation matrices.
  """
  linear = jnp.array(linear)
  translation = jnp.array(translation)

  affine = jnp.concatenate([linear, translation[..., jnp.newaxis]], axis=-1)
  return affine


def expand_transform_3x4_to_4x4(transforms_3x4):
  """Appends a [0, 0, 0, 1] row to each 3x4 transform.

  Args:
    transforms_3x4: a [..., 3, 4] Array (or convertible value) containing 3x4
      affine transforms.

  Returns:
    A [..., 4, 4] Array of 4x4 transformation matrices.
  """
  transforms_3x4 = jnp.array(transforms_3x4)

  dtype = transforms_3x4.dtype
  shape_base = transforms_3x4.shape[:-2]
  zeros = jnp.zeros(shape_base + (1, 3), dtype=dtype)
  ones = jnp.ones(shape_base + (1, 1), dtype=dtype)
  last_row = jnp.concatenate([zeros, ones], -1)
  transforms_4x4 = jnp.concatenate([transforms_3x4, last_row], -2)
  return transforms_4x4


def expand_transforms_3to4(transforms_3x3):
  """Expands a 3x3 transformation matrix to a 4x4 transformation matrix.

  The corner elements (output[..., 3, 3]) will be ones, while the other expanded
  elements will be zeros.

  Args:
    transforms_3x3: a [..., 3, 3] Array (or convertible value) containing 3x3
      transforms.

  Returns:
    A [..., 4, 4] Array of 4x4 transformation matrices.
  """
  transforms_3x3 = jnp.array(transforms_3x3)

  translations = jnp.zeros_like(transforms_3x3[..., 0])
  transforms_3x4 = make_affine(transforms_3x3, translations)
  transforms_4x4 = expand_transform_3x4_to_4x4(transforms_3x4)

  return transforms_4x4


def euler_matrices(angles):
  """Computes a XYZ Tait-Bryan (improper Euler angle) rotation.

  This function matches the default 'sxyz' order of the euler_matrix function in
  transformations.py. It returns 4x4 matrices for convenient multiplication with
  other transformations.

  Args:
    angles: a [..., 3] Array (or convertible value) containing X, Y, and Z
      angles in radians.

  Returns:
    A [..., 4, 4] Array of matrices.
  """
  angles = jnp.array(angles)

  rotations_3x3 = roll_pitch_yaw_to_rotation_matrices(angles)
  return expand_transforms_3to4(rotations_3x3)
