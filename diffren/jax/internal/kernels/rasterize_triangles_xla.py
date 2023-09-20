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
from jax.interpreters import batching
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np


xla_client = jax.lib.xla_client
xops = xla_client.ops

for _name, _value in rasterize_triangles_cpu.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform='cpu')  # pytype: disable=module-attr

for _name, _value in rasterize_triangles_gpu.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform='gpu')  # pytype: disable=module-attr


def _major_to_minor_layout(shape):
  return tuple(range(len(shape) - 1, -1, -1))


def _translation_rule_cpu(ctx, vertices, triangles, *, image_width,
                          image_height, num_layers, face_culling_mode):
  """Rasterize triangles translation rule for CPU using XLA."""
  vertices_aval, triangles_aval = ctx.avals_in

  def get_int_op_and_layout(value):
    value = np.array(value, dtype=np.int32)
    operand = mlir.ir_constant(value)
    layout = _major_to_minor_layout(value.shape)
    return operand, layout

  def get_count_op_and_layout(array_shape):
    operand = mlir.ir_constant(np.array(array_shape[0], dtype=np.int32))
    return operand, ()

  vertices_layout = _major_to_minor_layout(vertices_aval.shape)
  triangles_layout = _major_to_minor_layout(triangles_aval.shape)
  vertex_count_operand, vertex_count_layout = get_count_op_and_layout(
      vertices_aval.shape)
  triangle_count_operand, triangle_count_layout = get_count_op_and_layout(
      triangles_aval.shape)

  height_operand, height_layout = get_int_op_and_layout(image_height)
  width_operand, width_layout = get_int_op_and_layout(image_width)
  num_layers_operand, num_layers_layout = get_int_op_and_layout(num_layers)
  culling_mode_operand, culling_mode_layout = get_int_op_and_layout(
      face_culling_mode)

  return mlir.custom_call(
      'rasterize_triangles',
      operands=[
          vertices, vertex_count_operand, triangles, triangle_count_operand,
          width_operand, height_operand, num_layers_operand,
          culling_mode_operand
      ],
      operand_layouts=[
          vertices_layout, vertex_count_layout, triangles_layout,
          triangle_count_layout, width_layout, height_layout, num_layers_layout,
          culling_mode_layout
      ],
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      result_layouts=[[2, 1, 0], [2, 1, 0], [3, 2, 1, 0]],
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING).results


def _translation_rule_gpu(ctx, vertices, triangles, *, image_width,
                          image_height, num_layers, face_culling_mode):
  """Rasterize triangles translation rule for GPU (CUDA) using XLA."""
  vertices_aval, triangles_aval = ctx.avals_in

  descriptor = descriptors_pb2.RasterizeTrianglesConfig()
  descriptor.image_width = image_width
  descriptor.image_height = image_height
  descriptor.num_layers = num_layers
  descriptor.face_culling_mode = face_culling_mode
  descriptor.triangle_count = triangles_aval.shape[0]

  return mlir.custom_call(
      'rasterize_triangles',
      operands=[vertices, triangles],
      operand_layouts=[_major_to_minor_layout(vertices_aval.shape),
                       _major_to_minor_layout(triangles_aval.shape)],
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      result_layouts=[[2, 1, 0], [2, 1, 0], [3, 2, 1, 0]],
      backend_config=descriptor.SerializeToString(),
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING
  ).results


def rasterize_triangles_abstract_eval(vertices, triangles, *, image_width,
                                      image_height, num_layers,
                                      face_culling_mode):
  """Abstract evaluation of rasterize_triangles."""
  assert len(vertices.shape) == 2
  assert vertices.shape[1] == 4
  assert len(triangles.shape) == 2
  assert triangles.shape[1] == 3
  del face_culling_mode

  return (jax.core.ShapedArray((num_layers, image_height, image_width),
                               dtype=np.int32),
          jax.core.ShapedArray((num_layers, image_height, image_width),
                               dtype=np.float32),
          jax.core.ShapedArray(
              (num_layers, image_height, image_width, 3), dtype=np.float32))


rasterize_triangles_p = jax.core.Primitive('rasterize_triangles')
rasterize_triangles_p.multiple_results = True
rasterize_triangles_p.def_impl(functools.partial(
    jax.interpreters.xla.apply_primitive, rasterize_triangles_p))
rasterize_triangles_p.def_abstract_eval(rasterize_triangles_abstract_eval)

mlir.register_lowering(
    rasterize_triangles_p, _translation_rule_cpu, platform='cpu'
)
mlir.register_lowering(
    rasterize_triangles_p, _translation_rule_gpu, platform='cuda'
)


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
