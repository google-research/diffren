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

"""Differentiable point splatting functions for rasterize-then-splat."""

import math

from diffren.jax import composite
from diffren.jax.internal import framebuffer
from diffren.jax.internal import interpolate
from diffren.jax.utils import transforms
import jax.numpy as jnp


def splat_at_pixel_centers(xyz_layers: jnp.ndarray, color_layers: jnp.ndarray,
                           foreground_mask: jnp.ndarray) -> jnp.ndarray:
  """Splat a buffer of XYZ, color samples onto a pixel grid of the same size.

  This is a specialized splatting function that takes a multi-layer buffer of
  screen-space XYZ positions and colors and splats each sample into
  a buffer of the same size, using a 3x3 Gaussian kernel of variance 0.25.
  The accumulated layers are then composited back-to-front.

  The specialized part is that the 3x3 kernel is always centered on the
  pixel-coordinates of the sample in the input buffer, *not* the XY position
  stored at that sample, but the weights are defined by using the XY position.
  Computing weights w.r.t. the XY positions, rather than the pixel-centers,
  allows gradients to flow from the output colors back to the XY positions. When
  used in rasterize-then-splat, XY positions will always coincide with the pixel
  centers, so the forward computation is the same as if the XY positions defined
  the position of the splat.

  When splatting, the Z of the splat is compared with the Z of the layers under
  the splat sample. The sample is accumulated into the layer with the Z closest
  to the Z of the splat itself.

  Colors are encoded into an array of color components where the final component
  is alpha. A typical example of this is the standard RGBA encoding.

  Args:
    xyz_layers: a float32 array of rasterized XYZ positions with shape
      [num_layers, height, width, 3]
    color_layers: a array of attributes [n, h, w, component_count]. For RGBA
      color encoding, component_count is 4.
    foreground_mask: a mask of rendered pixels in each layer [n, h, w, 1].

  Returns:
    a [h, w, component_count] array of output color values, as well as [n,
    h, w, component_count] arrays of accumulated and normalized colors for
    visualization and debugging.
  """
  extra_accumulation_epsilon = 0.05

  gaussian_variance = 0.5**2
  gaussian_exp_scale = -1.0 / (2 * gaussian_variance)

  # The normalization coefficient for the Gaussian must be computed with care
  # so that a full accumulation of neighboring splats adds up to 1.0 +
  # epsilon. We need to trigger normalization when the splats accumulate to a
  # full surface in order to avoid a spurious "spread-splats-to-darken-color"
  # derivative, but we do not want to normalize otherwise (e.g., at the
  # boundary with the background), so we use a small epsilon here.
  weight_sum = 0
  for u in (-1, 0, 1):
    for v in (-1, 0, 1):
      weight_sum += math.exp(gaussian_exp_scale * (u**2 + v**2))
  gaussian_coef = (1.0 + extra_accumulation_epsilon) / weight_sum

  # Accumulation buffers need a 1 pixel border because of 3x3 splats.
  padding = ((0, 0), (1, 1), (1, 1), (0, 0))
  # 3 accumulation layers (fg, surface, bg) of the same size as the image.
  accumulation_shape = [3, color_layers.shape[1], color_layers.shape[2]]
  accumulate_color = jnp.pad(
      jnp.zeros(
          accumulation_shape + [color_layers.shape[3]],
          dtype=color_layers.dtype), padding)
  accumulate_weights = jnp.pad(
      jnp.zeros(accumulation_shape + [1], dtype=color_layers.dtype), padding)
  padded_center_z = jnp.pad(xyz_layers[..., 2:3], padding, constant_values=1.0)

  num_layers = color_layers.shape[0]

  # TODO(srinivaskaza): vmap this as well?
  for l in range(num_layers):
    color = color_layers[l, ...]
    xyz = xyz_layers[l, ...]
    foreground = foreground_mask[l, ...]

    # Computes the offset from the splat to the pixel underneath the splat.
    # Note that in the forward pass, splat_to_center_pixel will always be zero
    # to within numerical precision, but it is necessary to define the filter
    # tap weights as a function of the splat position so derivatives will flow
    # to the splat. As the splat moves right, the pixel moves left relative to
    # it, so the splat position xy is negated here.
    splat_to_center_pixel = jnp.floor(xyz[..., :2]) + jnp.array(
        [0.5, 0.5]) - xyz[..., :2]

    # TODO(srinivaskaza): Vectorize this as well?
    for u in (-1, 0, 1):
      for v in (-1, 0, 1):
        splat_to_pixel = splat_to_center_pixel + jnp.array([u, v])
        dist_sqr = jnp.sum(splat_to_pixel**2, axis=-1, keepdims=True)
        tap_weights = foreground * gaussian_coef * jnp.exp(
            gaussian_exp_scale * dist_sqr)

        tap_color = tap_weights * color

        padding = [[max(v + 1, 0), abs(min(v - 1, 0))],
                   [max(u + 1, 0), abs(min(u - 1, 0))], [0, 0]]
        tap_color = jnp.pad(tap_color, padding)
        tap_weights = jnp.pad(tap_weights, padding)

        # Find the layer index of the first surface shared by the center of
        # the splat and the splat filter tap (i.e., sample position). The
        # first surface must appear as the top layer either at center or at
        # tap. The best matching Z between the top center layer and the tap
        # layers is compared against the best match between the center layers
        # and the top tap layer, and the pair of layers with smallest
        # difference in Z is the estimated surface.
        tap_z_layers = jnp.pad(
            xyz_layers[..., 2:3], [[0, 0]] + padding, constant_values=1.0)
        dist_center_to_tap_layers = jnp.abs(tap_z_layers -
                                            padded_center_z[0, ...])
        best_center_surface_idx = jnp.argmin(dist_center_to_tap_layers, axis=0)
        best_center_surface_z = jnp.min(dist_center_to_tap_layers, axis=0)
        dist_tap_to_center_layers = jnp.abs(padded_center_z -
                                            tap_z_layers[0, ...])
        best_tap_surface_idx = jnp.argmin(dist_tap_to_center_layers, axis=0)
        best_tap_surface_z = jnp.min(dist_tap_to_center_layers, axis=0)
        # surface_idx is 0 if the first surface is the top layer for both
        # center and tap, a negative number (of layers) if the surface is
        # occluded at center, and a positive number if occluded at tap.
        surface_idx = jnp.where(best_tap_surface_z < best_center_surface_z,
                                -best_tap_surface_idx, best_center_surface_idx)

        # If the current layer is in front of the surface, accumulate into fg.
        # If at the surface, accumulate into surf. If behind, accumulate into
        # bg. We use a masked accumulation here rather than a scatter, though
        # scatter could also work if there are a lot of layers.
        fg_mask = (surface_idx > l).astype(jnp.float32)
        surf_mask = (surface_idx == l).astype(jnp.float32)
        bg_mask = (surface_idx < l).astype(jnp.float32)
        layer_mask = jnp.stack((fg_mask, surf_mask, bg_mask), axis=0)

        masked_tap_color = jnp.tile(
            jnp.expand_dims(tap_color, axis=0), (3, 1, 1, 1)) * layer_mask
        masked_tap_weights = jnp.tile(
            jnp.expand_dims(tap_weights, axis=0), (3, 1, 1, 1)) * layer_mask

        accumulate_color = accumulate_color + masked_tap_color
        accumulate_weights = accumulate_weights + masked_tap_weights

  # Normalize the accumulated colors by the accumulated weights. Normalization
  # only happens if the accumulate weights are > 1.0.
  accumulate_color = accumulate_color[:, 1:-1, 1:-1, :]
  accumulate_weights = accumulate_weights[:, 1:-1, 1:-1, :]
  normalization_scales = 1.0 / (
      jnp.maximum(accumulate_weights - 1.0, 0.0) + 1.0)
  normalized_color = accumulate_color * normalization_scales

  # Composite the foreground, surface, and background layers back-to-front.
  output_color = composite.over(normalized_color)

  return output_color, accumulate_color, normalized_color  # pytype: disable=bad-return-type  # jax-ndarray


def splat_shaded_pixels(vertices: jnp.ndarray, camera_matrices: jnp.ndarray,
                        rasterized_fb: framebuffer.Framebuffer,
                        shaded_pixels: jnp.ndarray,
                        return_accum_buffers: bool) -> jnp.ndarray:
  """Splats shaded pixels back onto the image grid.

  This function implements the splatting stage of rasterize-then-splat. It
  constructs screen-space positions for each splat based on the vertices and
  camera matrices. These screen-space positions line up exactly with the pixel
  centers, but are linked to the vertices and camera matrices for automatic
  differentiation.

  Args:
    vertices: float32 array of xyz positions with shape [vertex_count, d]. If
      camera_matrices is specified, d may be 3 or 4. If camera_matrices is None,
      d must be 4 and the values are assumed to be xyzw homogenous coordinates.
    camera_matrices: camera matrices with size [4, 4]. May be None. If None,
      vertices are assumed to be already be in homogenous clip space (xyzw).
    rasterized_fb: a Framebuffer containing ids, barycentrics, and mask output
      from rasterize_triangles().
    shaded_pixels: float32 array of shaded pixel colors with shape [num_layers,
      height, width, component_count].
    return_accum_buffers: bool specifying whether to return intermediate
      splatting accumulation buffers along with the final output. Intended for
      debugging.

  Returns:
    a [image_height, image_width, component_count] array of colors
  """
  image_height, image_width = shaded_pixels.shape[1:3]

  projected_vertices = vertices
  if camera_matrices is not None:
    projected_vertices = transforms.transform_homogeneous(
        camera_matrices, projected_vertices)

  z_background = (0, 0, 1, 1)  # Background Z = 1, W = 1
  clip_space_buffer = interpolate.interpolate_vertex_attribute(
      projected_vertices, rasterized_fb, z_background)

  ndc_xyz = clip_space_buffer[..., :3] / clip_space_buffer[..., 3:4]
  viewport_xyz = (ndc_xyz + 1.0) * jnp.array([image_width, image_height, 1],
                                             dtype=jnp.float32).reshape(
                                                 1, 1, 1, 3) * 0.5

  output, accum, norm_accum = splat_at_pixel_centers(
      viewport_xyz, shaded_pixels, rasterized_fb.foreground_mask)

  if return_accum_buffers:
    return output, accum, norm_accum  # pytype: disable=bad-return-type  # jax-ndarray

  return output
