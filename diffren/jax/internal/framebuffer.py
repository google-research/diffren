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

"""Storage classes for framebuffers and related data."""

from typing import Dict

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class Framebuffer(object):
  """A framebuffer holding rasterized values required for deferred shading.

  Framebuffers have at least [height, width, channels] dimensions. They may have
  additional batch and layer dimensions, as well. Valid shapes are:
  [h, w, c], [num_layers, h, w, c], [batch_size, num_layers, h, w, c].

  Immutable once created.
  """
  # The barycentric weights of the pixel centers in the covering triangle.
  barycentrics: jnp.ndarray
  # The index of the triangle covering this pixel. Not differentiable.
  triangle_id: jnp.ndarray
  # The indices of the vertices of the triangle covering this pixel.
  # Not differentiable.
  vertex_ids: jnp.ndarray
  # A mask of the pixels covered by a triangle. 1 if covered, 0 if background.
  # Not differentiable.
  foreground_mask: jnp.ndarray

  # Other rasterized attribute values (e.g., colors, UVs, normals, etc.).
  attributes: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)

  def __post_init__(self):
    # Checks that all buffers have rank and same shape up to the
    # number of channels.
    values = [
        self.barycentrics, self.triangle_id, self.vertex_ids,
        self.foreground_mask
    ]
    values += self.attributes.values()

    # During tracing, JAX appears to re-run the constructor for the dataclass
    # with placeholder members. These objects are not actually arrays, and
    # thus, this check won't work. This try-except block returns early in that
    # case.
    #
    # TODO(srinivaskaza): Read over https://github.com/google/jax/issues/2371
    # and figure out what the placeholder objects are.
    try:
      shapes = [v.shape for v in values if v is not None]
    except AttributeError:
      return

    try:
      for i in range(1, len(shapes)):
        if not jnp.array_equal(shapes[0][:-1], shapes[i][:-1]):  # pytype: disable=wrong-arg-types  # jnp-type
          raise ValueError(
              f"Expected all input shapes to match (up to channels), "
              f"but found {shapes}")
    except jax.errors.ConcretizationTypeError:
      pass

  @property
  def is_batched(self):
    return self.barycentrics.shape.ndim == 5  # pytype: disable=attribute-error  # jax-ndarray

  @property
  def is_multi_layer(self):
    return self.barycentrics.shape.ndim in [4, 5]  # pytype: disable=attribute-error  # jax-ndarray

  @property
  def batch_size(self):
    return self.barycentrics.shape[0] if self.is_batched else 1

  @property
  def num_layers(self):
    if self.is_batched:  # pylint: disable=using-constant-test
      return self.barycentrics.shape[1]
    elif self.is_multi_layer:
      return self.barycentrics.shape[0]
    else:
      return 1

  @property
  def height(self):
    return self.barycentrics.shape[-3]

  @property
  def width(self):
    return self.barycentrics.shape[-2]

  @property
  def pixel_count(self):
    return self.num_layers * self.height * self.width

  @property
  def background_mask(self):
    return 1 - self.foreground_mask

  def layer(self, index):
    """Slices at the given layer index, returning a single-layer Framebuffer."""
    if not self.is_multi_layer:
      if index > 0:
        raise ValueError(
            f"Invalid layer index {index} for single-layer Framebuffer.")
      return self

    return Framebuffer(
        triangle_id=self.triangle_id[:, index, ...]
        if self.triangle_id is not None else None,
        vertex_ids=self.vertex_ids[:, index, ...],
        foreground_mask=self.foreground_mask[:, index, ...],
        attributes={k: v.layer(index) for k, v in self.attributes.items()},  # pytype: disable=attribute-error  # jax-ndarray
        barycentrics=self.barycentrics.layer(index))  # pytype: disable=attribute-error  # jax-ndarray
