# Copyright 2026 The diffren Authors.
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
from absl.testing import parameterized
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



class RelativeIndicesTest(parameterized.TestCase):

  def _load(self, contents):
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, 'mesh.obj')
      with open(path, 'w') as f:
        f.write(contents)
      return obj_loader.load_and_flatten_obj(path)

  @parameterized.parameters('v', 'v/vt', 'v//vn', 'v/vt/vn')
  def test_relative_attributes_match_absolute_indices(self, layout):
    header = 'v 0 0 0\nv 1 0 0\nv 0 1 0\n' 'vt 0.25 0.5\nvt 0.75 1\nvn 0 0 1\n'
    relative = {
        'v': '-3 -2 -1',
        'v/vt': '-3/-2 -2/-1 -1/-2',
        'v//vn': '-3//-1 -2//-1 -1//-1',
        'v/vt/vn': '-3/-2/-1 -2/-1/-1 -1/-2/-1',
    }
    absolute = {
        'v': '1 2 3',
        'v/vt': '1/1 2/2 3/1',
        'v//vn': '1//1 2//1 3//1',
        'v/vt/vn': '1/1/1 2/2/1 3/1/1',
    }
    expected = self._load(header + 'f ' + absolute[layout] + '\n')
    actual = self._load(header + 'f ' + relative[layout] + '\n')
    for a, b in zip(actual, expected):
      np.testing.assert_array_equal(a, b)

  def test_relative_indices_resolve_at_each_face(self):
    vertices, triangles = self._load(
        'v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n' 'v 0 0 1\nf -3 -2 -1\n'
    )
    np.testing.assert_array_equal(
        vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    np.testing.assert_array_equal(triangles, [[0, 1, 2], [1, 2, 3]])

  def test_equivalent_absolute_and_relative_vertices_are_shared(self):
    vertices, triangles = self._load(
        'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf -3 2 -1\n'
    )
    self.assertEqual(vertices.shape, (3, 3))
    np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 1, 2]])

  def test_relative_attributes_can_change_without_new_positions(self):
    vertices, triangles = self._load(
        'v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 0.5\nvn 0 0 1\n'
        'f 1/-1/-1 2/-1/-1 3/-1/-1\n'
        'vt 0.75 1\nvn 0 1 0\nf 1/-1/-1 2/-1/-1 3/-1/-1\n'
    )
    self.assertEqual(vertices.shape, (6, 8))
    np.testing.assert_array_equal(vertices[:3, :3], vertices[3:, :3])
    np.testing.assert_array_equal(vertices[:3, 3:], [[0.25, 0.5, 0, 0, 1]] * 3)
    np.testing.assert_array_equal(vertices[3:, 3:], [[0.75, 1, 0, 1, 0]] * 3)
    np.testing.assert_array_equal(triangles, [[0, 1, 2], [3, 4, 5]])

  @parameterized.parameters('0', '-4', '4', '1/0', '1/-2', '1//0', '1//-2')
  def test_invalid_indices_raise(self, vertex):
    with self.assertRaises(ValueError):
      self._load(
          'v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n'
          + 'f '
          + vertex
          + ' 2 3\n'
      )


if __name__ == '__main__':
  absltest.main()
