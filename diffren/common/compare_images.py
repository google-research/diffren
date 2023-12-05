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

"""Test utility functions for comparing image buffers with error tolerance.
"""

import io
import os

import numpy as np
from PIL import Image as PilImage
from PIL import ImageChops


def images_are_near(baseline_image,
                    result_image,
                    max_outlier_fraction=0.005,
                    pixel_error_threshold=0.04):
  """Compares two image arrays.

  The comparison is soft: the images are considered identical if fewer than
  max_outlier_fraction of the pixels differ by more than pixel_error_threshold
  of the full color value.

  Differences in JPEG encoding can produce pixels with pretty large variation,
  so by default we use 0.04 (4%) for pixel_error_threshold and 0.005 (0.5%) for
  max_outlier_fraction.

  Args:
    baseline_image: a numpy array containing the baseline image.
    result_image: a numpy array containing the result image.
    max_outlier_fraction: fraction of pixels that may vary by more than the
      error threshold. 0.005 means 0.5% of pixels.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.

  Returns:
    A (boolean, string) tuple where the first value is whether the images
    matched, and the second is a pretty-printed summary of the differences.
  """
  if baseline_image.shape != result_image.shape:
    return False, ("Image shapes %s and %s do not match" %
                   (np.array_str(np.array(baseline_image.shape)),
                    np.array_str(np.array(result_image.shape))))

  float_base = baseline_image.astype(float) / 255.0
  float_result = result_image.astype(float) / 255.0

  outlier_channels = np.abs(float_base - float_result) > pixel_error_threshold
  if len(baseline_image.shape) > 2:
    outlier_pixels = np.any(outlier_channels, axis=2)
  else:
    outlier_pixels = outlier_channels
  outlier_fraction = np.count_nonzero(outlier_pixels) / np.prod(
      baseline_image.shape[:2])
  images_match = outlier_fraction <= max_outlier_fraction
  message = (" (%f of pixels are outliers, maximum allowed is %f) " %
             (outlier_fraction, max_outlier_fraction))
  return images_match, message


def expect_images_are_near(test,
                           baseline_image,
                           result_image,
                           max_outlier_fraction=0.005,
                           pixel_error_threshold=0.04):
  """A convenience wrapper around ImagesAreNear that adds a test assertion."""
  images_match, message = images_are_near(baseline_image, result_image,
                                          max_outlier_fraction,
                                          pixel_error_threshold)
  test.assertTrue(images_match, msg=message)


def _save_image(image, outputs_dir, comparison_name, suffix, save_format):
  path = os.path.join(outputs_dir, "{}_{}{}".format(comparison_name, suffix,
                                                    save_format))
  image.save(path)


def expect_images_are_near_and_save_comparison(test,
                                               baseline_image,
                                               result_image,
                                               comparison_name,
                                               images_differ_message,
                                               max_outlier_fraction=0.005,
                                               pixel_error_threshold=0.04,
                                               save_format=".png"):
  """A convenience wrapper around ImagesAreNear that saves comparison images.

  If the images differ, this function writes the
  baseline and result images into the test's outputs directory.

  Args:
    test: a python unit test instance.
    baseline_image: baseline image as a numpy array.
    result_image: the result image as a numpy array.
    comparison_name: a string naming this comparison. Names outputs for viewing
      in sponge.
    images_differ_message: the test message to display if the images differ.
    max_outlier_fraction: fraction of pixels that may vary by more than the
      error threshold. 0.005 means 0.5% of pixels.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.
    save_format: a text string defining the image format to save.
  """
  images_match, comparison_message = images_are_near(baseline_image,
                                                     result_image,
                                                     max_outlier_fraction,
                                                     pixel_error_threshold)

  if not images_match:
    outputs_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
    test.assertNotEmpty(outputs_dir)

    if baseline_image.ndim == 2 or baseline_image.shape[2] == 1:
      image_mode = "L"
    elif baseline_image.shape[2] == 3:
      image_mode = "RGB"
    elif baseline_image.shape[2] == 4:
      image_mode = "RGBA"
    else:
      raise ValueError("Unsupported image mode.")

    baseline_pil_image = PilImage.fromarray(baseline_image, mode=image_mode)
    _save_image(baseline_pil_image, outputs_dir, comparison_name, "baseline",
                save_format)

    result_pil_image = PilImage.fromarray(result_image, mode=image_mode)
    _save_image(result_pil_image, outputs_dir, comparison_name, "result",
                save_format)

    diff_pil_image = ImageChops.difference(baseline_pil_image, result_pil_image)
    if image_mode == "L" or image_mode == "RGB":
      _save_image(diff_pil_image, outputs_dir, comparison_name, "diff",
                  save_format)
    else:
      _save_image(
          diff_pil_image.convert("RGB"), outputs_dir, comparison_name,
          "diff_rgb", save_format)
      _save_image(
          diff_pil_image.getchannel("A"), outputs_dir, comparison_name,
          "diff_alpha", save_format)

  test.assertEqual(baseline_image.shape, result_image.shape)
  test.assertTrue(images_match, msg=images_differ_message + comparison_message)


def expect_image_file_and_image_are_near(test,
                                         baseline_path,
                                         result_image_bytes_or_numpy,
                                         comparison_name,
                                         images_differ_message,
                                         max_outlier_fraction=0.005,
                                         pixel_error_threshold=0.04,
                                         resize_baseline_image=None):
  """Compares the input image bytes with an image on disk.

  The comparison is soft: the images are considered identical if fewer than
  max_outlier_fraction of the pixels differ by more than pixel_error_threshold
  of the full color value. If the images differ, the function writes the
  baseline and result images into the test's outputs directory.

  Uses ImagesAreNear for the actual comparison.

  Args:
    test: a python unit test instance.
    baseline_path: path to the reference image on disk.
    result_image_bytes_or_numpy: the result image, as either a bytes object
      or a numpy array.
    comparison_name: a string naming this comparison. Names outputs for
      viewing in sponge.
    images_differ_message: the test message to display if the images differ.
    max_outlier_fraction: fraction of pixels that may vary by more than the
      error threshold. 0.005 means 0.5% of pixels.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.
    resize_baseline_image: a (width, height) tuple giving a new size to apply
      to the baseline image, or None.
  """
  try:
    result_image = np.array(
        PilImage.open(io.BytesIO(result_image_bytes_or_numpy)))
  except IOError:
    result_image = result_image_bytes_or_numpy
  baseline_pil_image = PilImage.open(baseline_path)
  baseline_format = ("." + baseline_pil_image.format).lower()

  if resize_baseline_image:
    baseline_pil_image = baseline_pil_image.resize(
        resize_baseline_image, PilImage.Resampling.LANCZOS
    )
  baseline_image = np.array(baseline_pil_image)

  expect_images_are_near_and_save_comparison(test, baseline_image, result_image,
                                             comparison_name,
                                             images_differ_message,
                                             max_outlier_fraction,
                                             pixel_error_threshold,
                                             baseline_format)


def get_pil_formatted_image(image, add_transparency=True):
  """Converts a [0,1] scaled numpy array containing an image to the PIL format.

  All channels are expected to be in the range [0,1]. Elements are mapped to
  uint8s. Any values that map outside [0.0, 255.0] will be clipped.

  If the image has only intensity information, then the intensity channel will
  be replicated to the R, G, and B channels. If the image has no alpha channel,
  and add_transparency is enabled, an opaque alpha channel will be concatenated
  with the image.

  The underlying storage order is changed to C-contiguous order (i.e.
  channels are directly adjacent in memory, and then grouped by rows before
  columns).

  Args:
    image: numpy array with shape [h, w], [h, w, 1], [h, w, 3], or [h, w, 4],
      where h and w are the height and width in pixels. The input image that
      will be converted.
    add_transparency: boolean specifying whether to add an opaque alpha
      channel to input images that do not have transparency information.
  Raises:
    ValueError: If the function catches an invalid argument.
  Returns:
    numpy array with shape [height, width, 3 (4 if add_transparency)]. Suitable
      for input to PilImage.fromarray() in either "RGB" or "RGBA" mode,
      depending on the value of add_transparency.
  """
  if not isinstance(image, np.ndarray):
    raise ValueError(
        "Input image must be a numpy array, but has type %s" % type(image))
  if len(image.shape) not in [2, 3]:
    raise ValueError(
        "Image tensor rank must be 2 or 3, but is %i" % len(image.shape))
  height, width = image.shape[0:2]
  if len(image.shape) == 2:
    image = np.expand_dims(image, axis=2)
  channel_count = image.shape[2]
  if channel_count not in [1, 3, 4]:
    raise ValueError(
        "Image channel count must be in [1, 3, 4], but was detected to be %i" %
        channel_count)
  if channel_count == 1:
    image = np.tile(image, [1, 1, 3])
  needs_alpha = channel_count in [1, 3] and add_transparency
  if needs_alpha:
    alpha_channel = np.ones([height, width, 1], dtype=np.float32)
    image = np.concatenate([image, alpha_channel], axis=2)
  return np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8).copy(order="C")
