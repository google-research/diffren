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

"""Main entry point for the Diffren renderer."""

from typing import Callable, Dict

from diffren.jax import composite
from diffren.jax import constants
from diffren.jax.internal import interpolate
from diffren.jax.internal import rasterize
from diffren.jax.internal import splat
import jax
import jax.numpy as jnp


def render_triangles(vertices: jnp.ndarray,
                     attributes: Dict[str, jnp.ndarray],
                     triangles: jnp.ndarray,
                     camera_matrices: jnp.ndarray,
                     image_width: int,
                     image_height: int,
                     shading_function: Callable[[Dict[str, jnp.ndarray]],
                                                jnp.ndarray],
                     num_layers: int = 1,
                     face_culling_mode=constants.FaceCullingMode.BACK,
                     compositing_mode=constants.CompositingMode.SPLAT_OVER,
                     return_accum_buffers=False,
                     compute_diff_barys=False) -> jnp.ndarray:
  """Rendering with differentiable occlusion using rasterize-then-splat.

  Rasterizes the input triangles to produce surface point samples, applies
  a user-specified shading function, then splats the shaded point
  samples onto the pixel grid and composites the result if `num_layers` > 1.
  If `compositing_mode` is `OVER`, the splatting step is skipped. Splatting
  provides smooth derivatives at occlusion boundaries, at the cost of slightly
  blurring the final image and reduced rendering speed. If you don't
  care about rendering derivatives, splatting can be safely turned off.

  The attributes are arbitrary per-vertex quantities (colors, normals, texture
  coordinates, etc.). The rasterization step interpolates these attributes
  across triangles to produce per-pixel interpolated attribute buffers
  with shape [image_height, image_width, attribute_size]. These buffers are
  passed to the user-provided shading_function in a dictionary with the
  corresponding attribute keys. The shading function should produce a
  [image_height, image_width, component_count] buffer of pixel colors. The
  result of the shader is replaced with zeros for background pixels. If
  `compositing_mode` is OVER or SPLAT_OVER, the final element of pixel colors is
  interpreted as alpha for compositing.

  In the common case that the attributes are RGBA vertex colors, the shading
  function would just pass the rasterized attributes through (e.g.,
  shading_function = lambda x: x['vertex_colors']).

  Args:
    vertices: float32 array of xyz positions with shape [vertex_count, d]. If
      camera_matrices is specified, d may be 3 or 4. If camera_matrices is None,
      d must be 4 and the values are assumed to be xyzw homogenous coordinates.
    attributes: dictionary of (str, float32 array) pairs of vertex attributes
      with shape [vertex_count, attribute_size]
    triangles: int32 array or array with shape [triangle_count, 3].
    camera_matrices: camera matrices with size [4, 4]. May be None. If None,
      vertices are assumed to be already be in homogenous clip space (xyzw).
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    shading_function: a function that takes a dictionary of (str, [image_height,
      image_width, attribute_count]) rasterized attribute arrays and returns a
      [image_height, image_width, component_count] color array. component_count
      must be >= 2, where the last component is alpha.
    num_layers: int specifying number of depth layers to composite.
    face_culling_mode: one of FaceCullingMode.[NONE, BACK, FRONT]. Defaults to
      BACK. Must be BACK if rasterize_then_splat is True.
    compositing_mode: one of CompositingMode.[NONE, OVER, SPLAT_OVER].
      Specifies how to composite the output layers. If NONE, no compositing is
      performed and separate layers are returned. If OVER, layers are composited
      using over compositing. If SPLAT_OVER, layers are splatted and then
      composited using over compositing.
    return_accum_buffers: bool specifying whether to return intermediate
      splatting accumulation buffers along with the final output. Intended for
      debugging.
    compute_diff_barys: bool specifying whether derivatives of barycentric
      coordinates w.r.t. vertices should be computed. Not necessary for
      rasterize-then-splat, so False by default. May be useful in cases
      derivatives are needed but splatting is not desirable, however
      differentiable barycentrices do *not* account for occlusion boundaries.

  Returns:
    a [image_height, image_width, component_count] array of colors, or a
    [num_layers, image_height, image_width, component_count] array if
    compositing_mode is NONE.

  Raises:
    ValueError: if `face_culling_mode` is not BACK and `compositing_mode` is
    SPLAT_OVER.
  """
  # Back face culling is necessary when rendering multiple layers with splats
  # so that back faces aren't counted as occluding layers.
  if (compositing_mode == constants.CompositingMode.SPLAT_OVER and
      face_culling_mode != constants.FaceCullingMode.BACK):
    raise ValueError("Backface culling must be enabled when using splatting.")

  rasterized = rasterize.rasterize_triangles(
      vertices,
      triangles,
      camera_matrices,
      image_width,
      image_height,
      num_layers=num_layers,
      face_culling_mode=face_culling_mode,
      compute_diff_barys=compute_diff_barys)

  interpolated = interpolate.interpolate_vertex_attributes(
      attributes, rasterized,
      jnp.zeros((rasterized.triangle_id.shape[-1],), dtype=jnp.float32))

  # Vmap the shader over the layers.
  shaded_buffer = jax.vmap(shading_function)(interpolated)

  if compositing_mode == constants.CompositingMode.SPLAT_OVER:
    return splat.splat_shaded_pixels(vertices, camera_matrices, rasterized,
                                     shaded_buffer, return_accum_buffers)

  # Zero out shader result outside of foreground mask for non-splatting modes.
  shaded_buffer = shaded_buffer * rasterized.foreground_mask

  if compositing_mode == constants.CompositingMode.OVER:
    return composite.over(shaded_buffer)

  return shaded_buffer
