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

"""Shaders to apply to geometry buffer output from the rasterizer."""

import io

from diffren.jax.utils import image
from diffren.jax.utils import transforms
from etils import epath
import jax.numpy as jnp
import numpy as np
from PIL import Image as PilImage


def load_texture(texture_filename, lower_left_origin=True):
  """Returns a texture image loaded from a file (float32 in [0,1] range)."""
  texture_bytes = epath.Path(texture_filename).read_bytes()
  texture = (
      np.asarray(PilImage.open(io.BytesIO(texture_bytes))).astype(np.float32)
      / 255.0
  )
  if lower_left_origin:
    # Flip the image to match the OpenGL origin lower-left convention.
    texture = np.flipud(texture)
  return texture


def texture_map(uv_image, texture_image_or_filename):
  """Applies a texture map to a UV image using bilinear interpolation.

  Clamps samples outside the texture to the edge of the texture (analogous to
  OpenGL CLAMP_TO_EDGE).

  Args:
    uv_image: a [height, width, 2] array of UV coordinates with range [0.0,
      1.0].
    texture_image_or_filename: texture as returned by load_texture or a path to
      pass to load_texture.

  Returns:
    a [height, width, 3] array of interpolated RGB values.
  """
  if isinstance(texture_image_or_filename, str):
    texture_image = load_texture(texture_image_or_filename)
  else:
    texture_image = texture_image_or_filename

  texture = jnp.array(texture_image)
  query_points = uv_image * jnp.array((texture.shape[1], texture.shape[0]))
  interpolated = image.bilinear_resample(texture, query_points, pad_mode='edge')
  return interpolated.reshape(
      uv_image.shape[0], uv_image.shape[1], texture.shape[2]
  )


def diffuse_light(
    diffuse_image,
    normals_image,
    light_direction=(0.5, 0.5, 1.0),
    ka=0.5,
    kd=0.5,
):
  """Applies diffuse directional lighting with an ambient component.

  Args:
    diffuse_image: a [height, width, 3] array of diffuse shading colors.
    normals_image: a [height, width, 3] array of normal (x,y,z) coordinates.
    light_direction: an iterable of length 3 containing the (x,y,z) direction of
      the light source.
    ka: the ambient shading coefficient.
    kd: the diffuse shading coefficient.

  Returns:
    a [batch, height, width, 3] array of shaded colors.
  """
  light_direction = jnp.array(light_direction)

  light_direction = light_direction.reshape(1, 1, 3)
  light_direction = transforms.l2_normalize(light_direction, axis=-1)
  n_dot_l = jnp.clip(
      jnp.sum(normals_image * light_direction, axis=2, keepdims=True), 0.0, 1.0
  )
  ambient = diffuse_image * ka
  diffuse = diffuse_image * kd * n_dot_l
  return ambient + diffuse


def point_light(
    diffuse_image,
    normals_image,
    pixel_positions,
    point_locations,
    eye_position=(0.0, 0.0, 0.0),
    ka=0.3,
    kd=1.4,
    ks=0.0,
    ns=0.0,
    attenuation_constant=1.0,
    attenuation_linear=0.7,
    attenuation_quadratic=1.8,
):
  """Applies a point light source with ambient, diffuse, and specular lighting.

    The attenuation is based on the quadratic formulation in OpenGL tutorial
    (see https://learnopengl.com/Lighting/Light-casters). Specifically,
    attenuation = 1.0 / (const + linear*dist + quad*dist*dist).

  Args:
    diffuse_image: a [height, width, 3] array of diffuse shading colors. -1
      indicates background.
    normals_image: a [height, width, 3] array of normal (x,y,z) coordinates. -1
      indicates background.
    pixel_positions: a [height, width, 3] array of pixel (x,y,z) coordinates. -1
      indicates background.
    point_locations: an iterable of length 3 containing the (x,y,z) location of
      light source.
    eye_position: an iterable of length 3 containing the (x,y,z) location of
      camera. If None, light location will be used as eye location.
    ka: the ambient shading coefficient (scalar or length 3 iterable).
    kd: the diffuse shading coefficient (scalar or length 3 iterable).
    ks: the specular shading coefficient (scalar or length 3 iterable).
    ns: the specular shading exponent (scalar).
    attenuation_constant: constantant attentuation coefficient.
    attenuation_linear: linear attentuation coefficient.
    attenuation_quadratic: quadratic attentuation coefficient.

  Returns:
    a [height, width, 3] array of shaded colors.
  """
  eye_position = jnp.array(eye_position)

  light_direction = transforms.l2_normalize(
      point_locations - pixel_positions, axis=-1
  )
  view_direction = transforms.l2_normalize(
      eye_position - pixel_positions, axis=-1
  )
  half_vector = transforms.l2_normalize(
      light_direction + view_direction, axis=-1
  )

  nol = jnp.clip(
      jnp.sum(normals_image * light_direction, axis=2, keepdims=True), 0.0, 1.0
  )
  noh = jnp.clip(
      jnp.sum(normals_image * half_vector, axis=2, keepdims=True), 0.0, 1.0
  )

  distance = jnp.linalg.norm(
      point_locations - pixel_positions, axis=-1, keepdims=True
  )
  decay = 1.0 / (
      attenuation_constant
      + attenuation_linear * distance
      + attenuation_quadratic * distance * distance
  )

  luma = ka + kd * nol + ks * noh**ns
  shading = diffuse_image * luma * decay

  return shading
