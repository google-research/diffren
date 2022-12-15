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

"""Utility functions for manipulating images."""

import jax
import jax.numpy as jnp


def bilinear_resample(image,
                      locations,
                      image_order='HWC',
                      locations_order='HWC',
                      pad_mode='constant'):
  """Resamples image data at the provided locations using gather.

  The domain of the image is (0,0) to (height, width). The image data samples
  are considered to lie at half-integer coordinates: a sample at location
  (0.5, 0.5) will take the exact value of image[0,0].

  Args:
    image: A [H1, W1, C] or [C, H1, W1] array of image data.
    locations: A [H2, W2, 2] or [2, H2, W2] array of floating point locations to
      sample. The coordinate order is 'xy', not 'ij'.
    image_order: Either 'HWC' or 'CHW'.
    locations_order: Either 'HWC' or 'CHW'.
    pad_mode: one of 'constant' or 'edge'.

  Returns:
    A tensor of shape [H2, W2, C] or [C, H2, W2] depending on image_order.
  """

  image = jnp.array(image)
  locations = jnp.array(locations)
  if image_order not in ['HWC', 'CHW'] or locations_order not in ['HWC', 'CHW']:
    raise ValueError(f'Bad ordering constant: Got image order {image_order}, '
                     f'locations order {locations_order}')
  image_channel_axis = 2 if image_order == 'HWC' else 0
  location_channel_axis = 2 if locations_order == 'HWC' else 0
  if (image.ndim != 3 or locations.ndim != 3 or
      locations.shape[location_channel_axis] != 2):
    raise ValueError(
        f'Bad shapes for bilinear resample. Got image shape {image.shape}, '
        f'locations shape {locations.shape}')
  if pad_mode not in ['constant', 'edge']:
    raise ValueError(f'Unsupported padding mode: {pad_mode}')

  def single_bilinear_sample(data, location):
    """Compute a single bilinear sample from a [H, W] single-channel image."""
    # Flip location because coordinate order is xy
    location = jnp.flip(location)

    # Add padding to handle out-of-domain pixels.
    data = jnp.pad(data, ((1, 1), (1, 1)), mode=pad_mode)
    location = location + 1.0
    floor = jnp.floor(location)

    # Clip floor to avoid out-of-bound accesses and to handle boundary cases.
    floor = jnp.clip(floor, 0, jnp.array(data.shape) - 2).astype(jnp.int32)

    # Out-of-bounds (OOB) pixels above or to the right have ceil_weight = 1 and
    # floor_weight = 0. OOB pixels below or to the left have floor_weight = 1
    # and ceil_weight = 0.
    ceil_weight = jnp.clip(location - floor, 0, 1)
    floor_weight = 1.0 - ceil_weight

    if str(jax.devices()[0]).lower().startswith('tpu'):
      # Use standard data lookup on TPU.
      ceil = floor + 1
      ll = data[floor[0], floor[1]] * floor_weight[0] * floor_weight[1]
      ul = data[ceil[0], floor[1]] * ceil_weight[0] * floor_weight[1]
      ur = data[ceil[0], ceil[1]] * ceil_weight[0] * ceil_weight[1]
      lr = data[floor[0], ceil[1]] * floor_weight[0] * ceil_weight[1]
    else:
      # On GPU, dynamic_slice may be more efficient than individual indexing.
      footprint = jax.lax.dynamic_slice(data, floor, (2, 2))
      ll = footprint[0, 0] * floor_weight[0] * floor_weight[1]
      ul = footprint[1, 0] * ceil_weight[0] * floor_weight[1]
      ur = footprint[1, 1] * ceil_weight[0] * ceil_weight[1]
      lr = footprint[0, 1] * floor_weight[0] * ceil_weight[1]

    return ll + ul + ur + lr

  def sample_single_channel(data, locations):
    """Samples a [H1,W1] image at [H2,W2,2] or [2,H2,W2] locations."""
    if locations_order == 'HWC':
      flat_locations_shape = (-1, 2)
      flat_locations_axis = 0
      full_samples_shape = locations.shape[:-1]
    else:
      flat_locations_shape = (2, -1)
      flat_locations_axis = 1
      full_samples_shape = locations.shape[1:]

    flat_locations = jnp.reshape(locations, flat_locations_shape)
    flat_samples = jax.vmap(
        single_bilinear_sample, in_axes=(None, flat_locations_axis),
        out_axes=0)(data, flat_locations)
    samples = jnp.reshape(flat_samples, full_samples_shape)
    return samples

  locations = locations - 0.5

  return jax.vmap(
      sample_single_channel,
      in_axes=(image_channel_axis, None),
      out_axes=image_channel_axis)(image, locations)
