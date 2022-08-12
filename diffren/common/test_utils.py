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

"""Common functions for the rasterizer and mesh renderer tests."""

import math
import os

from diffren.common import compare_images
from etils import epath
import numpy as np
from PIL import Image as PilImage


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TEST_DATA_DIRECTORY = 'common/test_data'


def make_resource_path(resource_name):
  return os.path.join(
      epath.resource_path('diffren'), TEST_DATA_DIRECTORY, resource_name)


def make_look_at_matrix(index):
  """Predefined lookat matrices that match test golden images."""
  center = np.array((0.0, 0.0, 0.0))
  world_up = np.array((0.0, 1.0, 0.0))
  if index == 0:
    eye = np.array((2.0, 3.0, 6.0))
  else:
    eye = np.array((-3.0, 1.0, 6.0))

  forward = center - eye
  forward_norm = np.linalg.norm(forward)
  forward = forward / forward_norm

  to_side = np.cross(forward, world_up)
  to_side_norm = np.linalg.norm(to_side)
  to_side = to_side / to_side_norm
  cam_up = np.cross(to_side, forward)

  view_rotation = np.eye(4)
  view_rotation[0, :3] = to_side
  view_rotation[1, :3] = cam_up
  view_rotation[2, :3] = -forward

  view_translation = np.eye(4)
  view_translation[:3, 3] = -np.array(eye)

  return np.matmul(view_rotation, view_translation)


def make_perspective_matrix(index=0):
  """Predefined perspective matrices that match test golden images."""
  fov_y = [40.0, 60.0]
  aspect = IMAGE_WIDTH / IMAGE_HEIGHT
  near = [0.01, 0.1]
  far = [10.0, 20.0]

  focal_y = 1.0 / np.tan(fov_y[index] * (math.pi / 360.0))
  depth_range = far[index] - near[index]
  a = -(far[index] + near[index]) / depth_range
  b = -2.0 * (far[index] * near[index] / depth_range)
  # pyformat: disable
  return np.array([[focal_y / aspect, 0.0, 0.0, 0.0],
                   [0.0, focal_y, 0.0, 0.0],
                   [0.0, 0.0, a, b],
                   [0.0, 0.0, -1.0, 0.0]])
  # pyformat: enable


def color_map_ids(ids_image, id_zero_is_white=False):
  """Maps an id image to a color cycle."""
  color_cycle = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255],
                          [0, 255, 255], [255, 255, 0],
                          [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 0, 128],
                          [0, 128, 128], [128, 128, 0]]) / 255.0

  mod_ids = ids_image % color_cycle.shape[0]
  flat_mapped_ids = color_cycle[np.reshape(mod_ids, (-1,)), :]
  if id_zero_is_white:
    id_is_zero = np.reshape(ids_image, (-1,)) == 0
    flat_mapped_ids[id_is_zero, :] = np.array([1.0, 1.0, 1.0])
  return np.reshape(flat_mapped_ids,
                    (ids_image.shape[0], ids_image.shape[1], 3))


def check_image(test,
                image_tensor,
                target_image_name,
                resize_image_to=None,
                exact=False,
                add_transparency=True):
  """Checks the input image against an image file and saves the result."""
  image = compare_images.get_pil_formatted_image(image_tensor, add_transparency)
  if resize_image_to:
    image = PilImage.fromarray(image).resize(resize_image_to, PilImage.NEAREST)
    image = np.array(image)

  error_threshold = 0 if exact else 0.04
  baseline_image_path = make_resource_path(target_image_name)
  compare_images.expect_image_file_and_image_are_near(
      test,
      baseline_image_path,
      image,
      target_image_name,
      '%s does not match.' % target_image_name,
      pixel_error_threshold=error_threshold)


def check_jacobians_are_nearly_equal(theoretical,
                                     numerical,
                                     outlier_relative_error_threshold,
                                     outlier_absolute_error_threshold,
                                     max_outlier_fraction,
                                     include_jacobians_in_error_message=False):
  """Compares two Jacobian matrices, allowing for some fraction of outliers.

  Args:
    theoretical: 2D numpy array containing a Jacobian matrix with entries
      computed via gradient functions. The layout should be as in the output of
      gradient_checker.
    numerical: 2D numpy array of the same shape as theoretical containing a
      Jacobian matrix with entries computed via finite difference
      approximations. The layout should be as in the output of gradient_checker.
    outlier_relative_error_threshold: float prescribing the maximum relative
      error between theoretical and numerical gradients before an entry is
      considered an outlier.
    outlier_absolute_error_threshold: float prescribing the maximum absolute
      error between theoretical and numerical gradients before an entry is
      considered an outlier.
    max_outlier_fraction: float defining the maximum fraction of entries in
      theoretical that may be outliers before the check returns False.
    include_jacobians_in_error_message: bool defining whether the jacobian
      matrices should be included in the return message should the test fail.

  Returns:
    A tuple where the first entry is a boolean describing whether
    max_outlier_fraction was exceeded, and where the second entry is a string
    containing an error message if one is relevant.
  """
  outliers = np.logical_not(
      np.isclose(theoretical, numerical, outlier_relative_error_threshold,
                 outlier_absolute_error_threshold))
  outlier_fraction = np.count_nonzero(outliers) / np.prod(numerical.shape[:2])
  jacobians_match = outlier_fraction <= max_outlier_fraction

  message = (
      ' %f of theoretical gradients are relative outliers, but the maximum'
      ' allowable fraction is %f ' % (outlier_fraction, max_outlier_fraction))
  if include_jacobians_in_error_message:
    # the gradient_checker convention is the typical Jacobian transposed:
    message += ('\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s' %
                (repr(numerical.T), repr(theoretical.T)))
  return jacobians_match, message
