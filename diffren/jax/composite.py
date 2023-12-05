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

"""Image compositing functions for Diffren."""

import jax.numpy as jnp


def over(color_layers: jnp.ndarray) -> jnp.ndarray:
  """Perform 'over' compositing for a stack of RGBA layers.

  Alpha is assumed to be premultiplied. The layers are assumed to be in
  front-to-back order where layer 0 is closest to the camera and layer k is
  farthest from the camera.

  Args:
    color_layers: a [l, h, w, 4] array of RGBA layers.

  Returns:
    a [h, w, 4] array of composited colors.
  """
  output_color = color_layers[-1, ...]
  for i in range(2, color_layers.shape[0] + 1):
    alpha = color_layers[-i, :, :, -1:]
    output_color = color_layers[-i, ...] + (1.0 - alpha) * output_color
  return output_color
