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

from absl.testing import absltest
from absl.testing import parameterized
import chex
from diffren.jax.utils import image

import numpy as np


class ImageTest(chex.TestCase, parameterized.TestCase):

  @parameterized.parameters(('HWC'), ('CHW'))
  def test_bilinear_resample_recovers_input(self, data_format):
    image_size = 64
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))

    input_image_shape = ((image_size, image_size,
                          3) if data_format == 'HWC' else
                         (3, image_size, image_size))
    input_image = np.reshape(np.arange(3 * image_size**2), (input_image_shape))
    samples_axis = -1 if data_format == 'HWC' else 0
    samples = np.stack((x, y), axis=samples_axis) + 0.5
    output_image = image.bilinear_resample(
        input_image,
        samples,
        image_order=data_format,
        locations_order=data_format)
    np.testing.assert_array_equal(input_image, output_image)

  @parameterized.parameters(('HWC'), ('CHW'))
  def test_bilinear_resample_boundary_cases(self, data_format):
    # Input image.
    # Each channel in the input_image contains the following simple example:
    # [[1, 2, 3, ..., image_width-1, image_width],
    #  [1, 2, 3, ..., image_width-1, image_width],
    #  ...,
    #  [1, 2, 3, ..., image_width-1, image_width]]
    image_height, image_width = 30, 10
    input_image = np.arange(1, image_width + 1, step=1, dtype=np.float32)
    input_image = np.reshape(input_image, [1, image_width])
    input_image = input_image.repeat(image_height, axis=0)

    channel_axis = 2 if data_format == 'HWC' else 0
    input_image = np.expand_dims(
        input_image, axis=channel_axis).repeat(
            3, axis=channel_axis)

    # 'samples' contains a pixel coordinate where it wants to sample a value
    # in the input_image. 'samples' is initialized with its pixel coordinate.
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    samples = np.stack((x, y), axis=channel_axis) + 0.5

    input_image = input_image.astype(dtype=np.float32)
    samples = samples.astype(dtype=np.float32)

    # Test cases. Each row has [[sample coordinate (x, y)], expected output]].
    # pyformat: disable
    test_case_list = [[[0.5, 0.5], 1],
                      [[0.1, 1.5], 0.6],
                      [[-0.5, 2], 0],
                      [[-1.3, 3], 0],
                      [[-200, 3], 0],
                      [[-float('inf'), 4], 0],
                      [[image_width - 0.5, 0.5], image_width],
                      [[image_width - 0.3, 0.5], (image_width) * 0.8],
                      [[image_width + 0.5, 1], 0],
                      [[image_width + 0.52, 2], 0],
                      [[image_width + 200, 3], 0],
                      [[float('inf'), 1], 0],
                      [[0.5, -0.4], 0.1],
                      [[2, -0.5], 0],
                      [[3, -1.3], 0],
                      [[2, -200], 0],
                      [[1, -float('inf')], 0],
                      [[0.5, image_height - 0.5], 1],
                      [[1.5, image_height - 0.2], 0.7 * 2],
                      [[2, image_height + 0.5], 0],
                      [[3, image_height + 0.52], 0],
                      [[2, image_height + 200], 0],
                      [[1, float('inf')], 0]]
    # pyformat: enable
    samples_custom = [sample for sample, _ in test_case_list]
    output_custom = [output for _, output in test_case_list]
    samples_custom = np.stack(samples_custom)
    output_custom = np.stack(output_custom)

    # Initialize and update the expected output.
    output_expected = input_image.copy()
    number_samples = samples_custom.shape[0]
    if data_format == 'HWC':
      samples[:number_samples, 0, :] = samples_custom
      output_expected[:number_samples, 0, :] = output_custom[:, np.newaxis]
    else:  # 'CHW'
      samples_custom = np.swapaxes(samples_custom, 0, 1)
      samples[:, :number_samples, 0] = samples_custom
      output_expected[:, :number_samples, 0] = output_custom

    output_image = image.bilinear_resample(
        input_image,
        samples,
        image_order=data_format,
        locations_order=data_format)

    # Floating-point comparison (the test cases above cause errors up to 2e-6).
    np.testing.assert_array_almost_equal(
        output_expected, output_image, decimal=5)


if __name__ == '__main__':
  absltest.main()
