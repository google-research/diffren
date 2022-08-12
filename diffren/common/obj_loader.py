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

"""Simple loader for Wavefront .obj files."""

from etils import epath
import numpy as np


VERTEX_TYPES = ['v', 'vt', 'vn']


def load_and_flatten_obj(obj_path):
  """Loads an .obj and flattens the vertex lists into a single array.

  .obj files may contain separate lists of positions, texture coordinates, and
  normals. In this case, a triangle vertex will have three values: indices into
  each of the position, texture, and normal lists. This function flattens those
  lists into a single vertex array by looking for unique combinations of
  position, texture, and normal, adding those to list, and then reindexing the
  triangles.

  This function processes only 'v', 'vt', 'vn', and 'f' .obj lines.

  Args:
    obj_path: the path to the Wavefront .obj file.

  Returns:
    a numpy array of vertices and a Mx3 numpy array of triangle indices.

    The vertex array will have shape Nx3, Nx5, Nx6, or Nx8, depending on whether
    position, position + texture, position + normals, or
    position + texture + normals are present.

    Unlike .obj, the triangle vertex indices are 0-based.
  """
  vertex_lists = {n: [] for n in VERTEX_TYPES}
  flat_vertices_list = []
  flat_vertices_indices = {}
  flat_triangles = []
  # Keep track of encountered vertex types.
  has_type = {t: False for t in VERTEX_TYPES}

  file_str = epath.Path(obj_path).read_text('utf-8')

  for line in file_str.splitlines():
    tokens = line.split()
    if not tokens:
      continue
    line_type = tokens[0]
    # We skip lines not starting with v, vt, vn, or f.
    if line_type in VERTEX_TYPES:
      vertex_lists[line_type].append([float(x) for x in tokens[1:]])
    elif line_type == 'f':
      triangle = []
      for i in range(3):
        # The vertex name is one of the form: 'v', 'v/vt', 'v//vn', or
        # 'v/vt/vn'.
        vertex_name = tokens[i + 1]
        if vertex_name in flat_vertices_indices:
          triangle.append(flat_vertices_indices[vertex_name])
          continue
        # Extract all vertex type indices ('' for unspecified).
        vertex_indices = vertex_name.split('/')
        while len(vertex_indices) < 3:
          vertex_indices.append('')
        flat_vertex = []
        for vertex_type, index in zip(VERTEX_TYPES, vertex_indices):
          if index:
            # obj triangle indices are 1 indexed, so subtract 1 here.
            flat_vertex += vertex_lists[vertex_type][int(index) - 1]
            has_type[vertex_type] = True
          else:
            # Append zeros for missing attributes.
            flat_vertex += [0, 0] if vertex_type == 'vt' else [0, 0, 0]
        flat_vertex_index = len(flat_vertices_list)

        flat_vertices_list.append(flat_vertex)
        flat_vertices_indices[vertex_name] = flat_vertex_index
        triangle.append(flat_vertex_index)
      flat_triangles.append(triangle)

  # Keep only vertex types that are used in at least one vertex.
  flat_vertices_array = np.float32(flat_vertices_list)
  flat_vertices = flat_vertices_array[:, :3]
  if has_type['vt']:
    flat_vertices = np.concatenate((flat_vertices, flat_vertices_array[:, 3:5]),
                                   axis=-1)
  if has_type['vn']:
    flat_vertices = np.concatenate((flat_vertices, flat_vertices_array[:, -3:]),
                                   axis=-1)

  return flat_vertices, np.int32(flat_triangles)
