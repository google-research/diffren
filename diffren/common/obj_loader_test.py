# Copyright 2023 The diffren Authors.
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

"""Tests for diffren.common.obj_loader."""

import os
import tempfile
from absl.testing import absltest
from diffren.common import obj_loader
from diffren.common import test_utils
import numpy as np


class ObjLoaderTest(absltest.TestCase):

  def test_loads_v_obj(self):
    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('sphere.obj'))
    self.assertEqual(vertices.shape, (92, 3))
    self.assertEqual(triangles.shape, (180, 3))
    # Check a few values from the .obj file. Note that the loader does not
    # preserve the order of vertices even if there is only one "v" list, so
    # these indices differ from the order in the .obj file.
    self.assertEqual(triangles[:3, :].tolist(),
                     [[0, 1, 2], [3, 2, 4], [5, 0, 3]])
    np.testing.assert_array_almost_equal(
        vertices[:3, :],
        np.array([[0.343073219, 0.111471429, -0.932670832], [0, 0, -1],
                  [0, 0.360728621, -0.932670832]]))

  def test_loads_v_vt_vn_obj(self):
    vertices, triangles = obj_loader.load_and_flatten_obj(
        test_utils.make_resource_path('ycb_toy_airplane.obj'))
    self.assertEqual(vertices.shape, (10504, 8))
    self.assertEqual(triangles.shape, (16384, 3))

    self.assertEqual(triangles[:3, :].tolist(),
                     [[0, 1, 2], [3, 4, 5], [6, 1, 7]])
    # Check the first three flattened vertices. These contain position, uv,
    # and normal in that order.
    np.testing.assert_array_almost_equal(
        vertices[:3, :],
        np.array([[
            0.017613, -0.094105, 0.003374, 0.568048, 0.320247, -0.093812,
            -0.594589, -0.798538
        ],
                  [
                      0.015891, -0.09501, 0.00506, 0.56525, 0.318785, -0.673264,
                      -0.482648, -0.560148
                  ],
                  [
                      0.016777, -0.090737, 0.001438, 0.571667, 0.317413,
                      -0.14963, -0.485498, -0.861338
                  ]]))

  def test_loads_v_vn_obj(self):
    temp_file, temp_path = tempfile.mkstemp()
    with os.fdopen(temp_file, 'w') as f:
      f.write("""
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
""")
    vertices, triangles = obj_loader.load_and_flatten_obj(temp_path)
    self.assertEqual(
        vertices.tolist(),
        [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]])
    self.assertEqual(triangles.tolist(), [[0, 1, 2]])

  def test_loads_v_vt_obj(self):
    temp_file, temp_path = tempfile.mkstemp()
    with os.fdopen(temp_file, 'w') as f:
      f.write("""
v 0 0 0
v 1 0 0
v 0 1 0
vt 0.5 0.5
f 1/1 2/1 3/1
""")
    vertices, triangles = obj_loader.load_and_flatten_obj(temp_path)
    self.assertEqual(
        vertices.tolist(),
        [[0, 0, 0, 0.5, 0.5], [1, 0, 0, 0.5, 0.5], [0, 1, 0, 0.5, 0.5]])
    self.assertEqual(triangles.tolist(), [[0, 1, 2]])


if __name__ == '__main__':
  absltest.main()
