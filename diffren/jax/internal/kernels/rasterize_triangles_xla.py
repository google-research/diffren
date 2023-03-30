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

"""JAX primitive for rasterizing triangles on CPU and GPU."""

import functools

from . import rasterize_triangles_cpu
from . import rasterize_triangles_gpu
from diffren.jax.internal.kernels import descriptors_pb2
import jax
from jax import abstract_arrays
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np


xla_client = jax.lib.xla_client
xops = xla_client.ops

for _name, _value in rasterize_triangles_cpu.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform='cpu')  # pytype: disable=module-attr

for _name, _value in rasterize_triangles_gpu.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform='gpu')  # pytype: disable=module-attr


def _get_shape(builder, element):
  return xla_client.Shape.array_shape(
      builder.GetShape(element).numpy_dtype(),
      builder.GetShape(element).dimensions(),
      list(reversed(range(len(builder.GetShape(element).dimensions())))))


def _translation_rule_cpu(builder, vertices, triangles, *, image_width,
                          image_height, num_layers, face_culling_mode):
  """Rasterize triangles translation rule for CPU using XLA."""

  def get_int_op_and_shape(value):
    operand = xops.Constant(builder, np.int32(value))
    shape = _get_shape(builder, operand)
    return operand, shape

  def get_count_op_and_shape(array):
    operand = xops.Constant(builder,
                            np.int32(builder.GetShape(array).dimensions()[0]))

    shape = _get_shape(builder, operand)
    return operand, shape

  vertices_shape = _get_shape(builder, vertices)
  triangles_shape = _get_shape(builder, triangles)
  vertex_count_operand, vertex_count_shape = get_count_op_and_shape(vertices)
  triangle_count_operand, triangle_count_shape = get_count_op_and_shape(
      triangles)

  height_operand, height_shape = get_int_op_and_shape(image_height)
  width_operand, width_shape = get_int_op_and_shape(image_width)
  num_layers_operand, num_layers_shape = get_int_op_and_shape(num_layers)
  culling_mode_operand, culling_mode_shape = get_int_op_and_shape(
      face_culling_mode)

  operand_shapes = [
      vertices_shape, vertex_count_shape, triangles_shape, triangle_count_shape,
      width_shape, height_shape, num_layers_shape, culling_mode_shape
  ]

  triangle_ids_shape = xla_client.Shape.array_shape(
      np.dtype(np.int32), (num_layers, image_height, image_width), [2, 1, 0])
  z_buffer_shape = xla_client.Shape.array_shape(
      np.dtype(np.float32), (num_layers, image_height, image_width), [2, 1, 0])
  barycentrics_shape = xla_client.Shape.array_shape(
      np.dtype(np.float32), (num_layers, image_height, image_width, 3),
      [3, 2, 1, 0])
  output_shapes = xla_client.Shape.tuple_shape(
      [triangle_ids_shape, z_buffer_shape, barycentrics_shape])

  out = xops.CustomCallWithLayout(
      builder,
      b'rasterize_triangles',
      operands=[
          vertices, vertex_count_operand, triangles, triangle_count_operand,
          width_operand, height_operand, num_layers_operand,
          culling_mode_operand
      ],
      operand_shapes_with_layout=operand_shapes,
      shape_with_layout=output_shapes,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING)
  return out


def _translation_rule_gpu(builder, vertices, triangles, *, image_width,
                          image_height, num_layers, face_culling_mode):
  """Rasterize triangles translation rule for GPU (CUDA) using XLA."""

  vertices_shape = _get_shape(builder, vertices)
  triangles_shape = _get_shape(builder, triangles)

  triangle_ids_shape = xla_client.Shape.array_shape(
      np.dtype(np.int32), (num_layers, image_height, image_width), [2, 1, 0])
  z_buffer_shape = xla_client.Shape.array_shape(
      np.dtype(np.float32), (num_layers, image_height, image_width), [2, 1, 0])
  barycentrics_shape = xla_client.Shape.array_shape(
      np.dtype(np.float32), (num_layers, image_height, image_width, 3),
      [3, 2, 1, 0])
  output_shapes = xla_client.Shape.tuple_shape(
      [triangle_ids_shape, z_buffer_shape, barycentrics_shape])

  descriptor = descriptors_pb2.RasterizeTrianglesConfig()
  descriptor.image_width = image_width
  descriptor.image_height = image_height
  descriptor.num_layers = num_layers
  descriptor.face_culling_mode = face_culling_mode
  descriptor.triangle_count = triangles_shape.dimensions()[0]

  out = xops.CustomCallWithLayout(
      builder,
      b'rasterize_triangles',
      operands=[
          vertices,
          triangles,
      ],
      operand_shapes_with_layout=[vertices_shape, triangles_shape],
      shape_with_layout=output_shapes,
      opaque=descriptor.SerializeToString(),
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING)
  return out


def rasterize_triangles_abstract_eval(vertices, triangles, *, image_width,
                                      image_height, num_layers,
                                      face_culling_mode):
  """Abstract evaluation of rasterize_triangles."""
  assert len(vertices.shape) == 2
  assert vertices.shape[1] == 4
  assert len(triangles.shape) == 2
  assert triangles.shape[1] == 3
  del face_culling_mode

  return (abstract_arrays.ShapedArray((num_layers, image_height, image_width),
                                      dtype=np.int32),
          abstract_arrays.ShapedArray((num_layers, image_height, image_width),
                                      dtype=np.float32),
          abstract_arrays.ShapedArray(
              (num_layers, image_height, image_width, 3), dtype=np.float32))


rasterize_triangles_p = jax.core.Primitive('rasterize_triangles')
rasterize_triangles_p.multiple_results = True
rasterize_triangles_p.def_impl(
    functools.partial(jax.xla.apply_primitive, rasterize_triangles_p))
rasterize_triangles_p.def_abstract_eval(rasterize_triangles_abstract_eval)

jax.xla.backend_specific_translations['cpu'][
    rasterize_triangles_p] = _translation_rule_cpu
jax.xla.backend_specific_translations['gpu'][
    rasterize_triangles_p] = _translation_rule_gpu


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

  return rasterize_triangles_p.bind(
      vertices,
      triangles,
      image_width=image_width,
      image_height=image_height,
      num_layers=num_layers,
      face_culling_mode=face_culling_mode)


def rasterize_triangles_batch(args, batch_axes, **params):
  """Batching version of rasterize triangles."""
  vertices, triangles = args[0:2]
  v_axis, t_axis = batch_axes[0:2]
  v_is_batched = v_axis is not None
  t_is_batched = t_axis is not None

  for other_arg_axis in batch_axes[2:]:
    # Only vertices and triangles should ever be batched.
    assert other_arg_axis is None

  batch_size = (
      vertices.shape[v_axis] if v_is_batched else triangles.shape[t_axis])

  results = []
  for i in range(batch_size):
    v_slice = jnp.take(vertices, i, v_axis) if v_is_batched else vertices
    t_slice = jnp.take(triangles, i, t_axis) if t_is_batched else triangles

    results.append(rasterize_triangles_p.bind(v_slice, t_slice, **params))
  # Reorder to create ((id1,id2,id3...), (z1,z2,z3...), (b1,b2,b3...)) tuples
  results = zip(*results)
  # Stack to create (id123, z123, b123) batched outputs
  outputs = map(jnp.stack, results)
  return outputs, (0, 0, 0)


batching.primitive_batchers[rasterize_triangles_p] = rasterize_triangles_batch
