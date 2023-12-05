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

"""Interpolation utilities for attributes."""

from typing import Dict, Iterable, Optional, Union

from diffren.jax.internal import framebuffer as fb
import jax
import jax.numpy as jnp


def _interpolate_vertex_attribute_layer(
    flat_corners: jnp.ndarray,
    framebuffer: fb.Framebuffer,
    background_value: Optional[Union[jnp.ndarray, Iterable[float]]] = None
) -> jnp.ndarray:
  """Interpolate a single vertex attribute across the input framebuffer.

  Args:
    flat_corners: 2-D vertex attribute Array with shape [num_vertices,
      num_channels]
    framebuffer: Framebuffer to interpolate across. Expected to contain
      barycentrics, vertex_ids, and foreground_mask.
    background_value: 1-D Array (or convertible value) with shape [num_channels]
      containing the value to use for background pixels. If None, defaults to
      zero.

  Returns:
    An array containing the per-pixel interpolated values.
  """
  num_channels = flat_corners.shape[-1]
  # corners has shape (num_pixels, 3, num_channels)
  corners = jnp.reshape(flat_corners, (-1, 3, num_channels))
  # barycentrics has shape (num_pixels, 3, 1)
  barycentrics = jnp.reshape(framebuffer.barycentrics, (-1, 3, 1))

  weighted_corners = jnp.multiply(corners, barycentrics)
  summed_attributes = jnp.sum(weighted_corners, axis=1)
  attribute_image = jnp.reshape(
      summed_attributes, (framebuffer.height, framebuffer.width, num_channels))

  if background_value is None:
    background_value = jnp.zeros((num_channels,), dtype=flat_corners.dtype)
  else:
    background_value = jnp.array(background_value, dtype=flat_corners.dtype)

  attribute_image = (
      framebuffer.foreground_mask * attribute_image +
      framebuffer.background_mask * background_value)

  return attribute_image


def interpolate_grid_attribute(
    attribute: jnp.ndarray,
    sdf: jnp.ndarray,
    framebuffer: fb.Framebuffer,
    background_value: Optional[Union[jnp.ndarray, Iterable[float]]] = None
) -> jnp.ndarray:
  """Interpolate a grid vertex attribute across the input framebuffer.

  Args:
    attribute: 4-D grid attribute array [depth, height, width, num_channels].
    sdf: 3-D SDF grid array with shape [depth, height, width]. Batching must
      match the attribute.
    framebuffer: Framebuffer to interpolate across. Expected to contain
      barycentrics, vertex_ids, and foreground_mask.
    background_value: 1-D array (or convertible value) with shape [num_channels]
      containing the value to use for background pixels. If None, defaults to
      zero.

  Returns:
    An array containing the per-pixel interpolated values.
  """

  def interpolate_layer(layer):
    num_channels = attribute.shape[-1]

    flat_sdf = sdf.reshape(-1, 1)
    flat_attribute = attribute.reshape(-1, num_channels)

    # Compared to the explicit case, the main difference is here, in getting
    # the 'corner' attribute values:
    # The 'corner ends' are the two 3D grid edge ends of each of the 3 triangle
    # 'corners' rendered at each pixel. The 'corner' attributes are the 3
    # triangle corners for each pixel.
    corner_end_attribute = flat_attribute[layer.vertex_ids]
    corner_end_sdf = flat_sdf[layer.vertex_ids]

    # Interpolate grid corners to vertices
    corner_end_pas, corner_end_pbs = jnp.split(corner_end_attribute, 2, axis=-2)
    corner_end_vas, corner_end_vbs = jnp.split(corner_end_sdf, 2, axis=-2)
    denom = corner_end_vbs - corner_end_vas
    denom = denom + jnp.where(denom < 0, -1e-6, 1e-6)

    alpha = (-1.0 * corner_end_vas) / denom
    corner_attribute = corner_end_pas + alpha * (
        corner_end_pbs - corner_end_pas)

    return _interpolate_vertex_attribute_layer(corner_attribute, layer,
                                               background_value)

  return jax.vmap(interpolate_layer)(framebuffer)


def interpolate_vertex_attribute(
    attribute: jnp.ndarray,
    framebuffer: fb.Framebuffer,
    background_value: Optional[Union[jnp.ndarray, Iterable[float]]] = None
) -> jnp.ndarray:
  """Interpolate a single vertex attribute across the input framebuffer.

  Args:
    attribute: 2-D vertex attribute Array with shape [num_vertices,
      num_channels].
    framebuffer: Framebuffer to interpolate across. Expected to contain
      barycentrics, vertex_ids, and foreground_mask.
    background_value: 1-D Array (or convertible value) with shape [num_layers,
      num_channels] containing the value to use for background pixels. If None,
      defaults to zero.

  Returns:
    An array containing the per-pixel interpolated values.
  """

  attribute = jnp.array(attribute)

  def interpolate_layer(layer):
    flat_corners = attribute[layer.vertex_ids[:], :]
    return _interpolate_vertex_attribute_layer(flat_corners, layer,
                                               background_value)

  return jax.vmap(interpolate_layer)(framebuffer)


def interpolate_vertex_attributes(
    attributes: Dict[str, jnp.ndarray],
    framebuffer: fb.Framebuffer,
    background_value: Optional[Union[jnp.ndarray, Iterable[float]]] = None
) -> Dict[str, jnp.ndarray]:
  """Interpolates multiple attributes across the input framebuffer.

  Args:
    attributes: dictionary of 2-D vertex attribute Arrays with shape
      [num_vertices, attribute_channels].
    framebuffer: Framebuffer to interpolate across. Expected to contain
      barycentrics, vertex_ids, and foreground_mask.
    background_value: 1-D Array (or convertible value) with shape [num_layers,
      num_channels] containing the value to use for background pixels. If None,
      defaults to zero.

  Returns:
    A dictionary of interpolated attribute buffers, each of shape [num_layers,
    height, width, attribute_channels].
  """
  def interpolate_attribute(v):
    return interpolate_vertex_attribute(v, framebuffer, background_value)
  return {k: interpolate_attribute(v) for k, v in attributes.items()}
