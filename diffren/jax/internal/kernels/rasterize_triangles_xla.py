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

"""JAX primitive for rasterizing triangles on CPU and GPU."""

import functools

import jax
import numpy as np

from . import rasterize_triangles_cpu
from . import rasterize_triangles_gpu

for _name, _value in rasterize_triangles_cpu.registrations().items():
  jax.ffi.register_ffi_target(_name, _value, platform='cpu')

for _name, _value in rasterize_triangles_gpu.registrations().items():
  jax.ffi.register_ffi_target(_name, _value, platform='gpu')


def rasterize_triangles(vertices, triangles, image_width, image_height,
                        num_layers, face_culling_mode):
  """Rasterizes triangles using XLA.

  Args:
    vertices: a (num_vertices, 4) array clip-space vertices (XYZW).
    triangles: a (num_triangles, 3) array of triangle indices.
    image_width: width of the output image.
    image_height: height of the output image.
    num_layers: number of depth layers in the output.
    face_culling_mode: an integer setting the face culling mode. See
      rasterize_triangles_impl.h for valid values.

  Returns:
    a tuple of three arrays: triangle id (n, h, w), NDC Z buffer (n, h, w),
      and barycentric coordinates (n, h, w, 3).
  """
  # Using stop_gradient here removes the need to define a JVP for
  # the rasterize_triangles primitive.
  vertices = jax.lax.stop_gradient(vertices)
  triangles = jax.lax.stop_gradient(triangles)

  assert len(vertices.shape) == 2
  assert vertices.shape[1] == 4
  assert len(triangles.shape) == 2
  assert triangles.shape[1] == 3

  out_type = (
      jax.ShapeDtypeStruct((num_layers, image_height, image_width),
                           dtype=np.int32),
      jax.ShapeDtypeStruct((num_layers, image_height, image_width),
                           dtype=np.float32),
      jax.ShapeDtypeStruct((num_layers, image_height, image_width, 3),
                           dtype=np.float32),
  )

  cpu_rule = functools.partial(
      jax.ffi.ffi_call('rasterize_triangles', out_type,
                       vmap_method='sequential_unrolled'),
      face_culling_mode=np.int32(face_culling_mode))
  cuda_rule = functools.partial(
      jax.ffi.ffi_call('rasterize_triangles', out_type,
                       vmap_method='sequential_unrolled'),
      face_culling_mode=np.int32(face_culling_mode))

  return jax.lax.platform_dependent(vertices, triangles, cpu=cpu_rule,
                                    cuda=cuda_rule)
