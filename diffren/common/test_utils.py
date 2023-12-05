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

"""Common functions for the rasterizer and mesh renderer tests."""

import math
import os

from diffren.common import compare_images
from diffren.common import obj_loader
from etils import epath
import numpy as np
from PIL import Image as PilImage


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TEST_DATA_DIRECTORY = 'common/test_data'


def make_resource_path(resource_name):
  return os.path.join(
      epath.resource_path('diffren'), TEST_DATA_DIRECTORY, resource_name
  )


def eye_position(view_name):
  """Returns test eye position and up vectors by name."""
  if view_name.startswith('view_1'):
    eye = np.array((2.0, 3.0, 6.0))
    world_up = np.array((0.0, 1.0, 0.0))
  elif view_name.startswith('view_2'):
    eye = np.array((-3.0, 1.0, 6.0))
    world_up = np.array((0.0, 1.0, 0.0))
  elif view_name.startswith('ycb_toy_airplane'):
    eye = np.array((0.2, 0.3, 0.5))
    world_up = np.array((0.0, 1.0, 0.0))
  elif view_name.startswith('spot'):
    eye = np.array((0.2, 0.3, -0.5))
    world_up = np.array((0.0, -1.0, 0.0))
  return eye, world_up


def make_look_at_matrix(view_name):
  """Predefined lookat matrices that match test golden images."""
  center = np.array((0.0, 0.0, 0.0))
  eye, world_up = eye_position(view_name)

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


def load_test_obj(obj_name):
  """Loads and applies rescaling to test .obj files."""
  vertices, triangles = obj_loader.load_and_flatten_obj(
      make_resource_path(obj_name)
  )

  obj_scale = 0.2 if obj_name.startswith('spot') else 1.0
  vertices[:, :3] *= obj_scale

  return vertices, triangles


def color_map_ids(ids_image, id_zero_is_white=False):
  """Maps an id image to a color cycle."""
  color_cycle = (
      np.array([
          [255, 0, 0],
          [0, 255, 0],
          [0, 0, 255],
          [255, 0, 255],
          [0, 255, 255],
          [255, 255, 0],
          [128, 0, 0],
          [0, 128, 0],
          [0, 0, 128],
          [128, 0, 128],
          [0, 128, 128],
          [128, 128, 0],
      ])
      / 255.0
  )

  mod_ids = ids_image % color_cycle.shape[0]
  flat_mapped_ids = color_cycle[np.reshape(mod_ids, (-1,)), :]
  if id_zero_is_white:
    id_is_zero = np.reshape(ids_image, (-1,)) == 0
    flat_mapped_ids[id_is_zero, :] = np.array([1.0, 1.0, 1.0])
  return np.reshape(
      flat_mapped_ids, (ids_image.shape[0], ids_image.shape[1], 3)
  )


def check_image(
    test,
    image_tensor,
    target_image_name,
    resize_image_to=None,
    exact=False,
    add_transparency=True,
):
  """Checks the input image against an image file and saves the result."""
  image = compare_images.get_pil_formatted_image(image_tensor, add_transparency)
  if resize_image_to:
    image = PilImage.fromarray(image).resize(
        resize_image_to, PilImage.Resampling.NEAREST
    )
    image = np.array(image)

  error_threshold = 0 if exact else 0.04
  baseline_image_path = make_resource_path(target_image_name)
  compare_images.expect_image_file_and_image_are_near(
      test,
      baseline_image_path,
      image,
      target_image_name,
      '%s does not match.' % target_image_name,
      pixel_error_threshold=error_threshold,
  )
