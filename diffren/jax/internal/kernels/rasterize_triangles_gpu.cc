/* Copyright 2024 The diffren Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.*/

#include <string>

#include "diffren/common/kernels/rasterize_triangles_impl_cuda.cu.h"
#include "diffren/common/kernels/rasterize_triangles_types.h"

#include "absl/base/casts.h"
#include "include/pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(diffren::FaceCullingMode);

namespace diffren {
namespace cuda {

// Calls RasterizeTriangleImpl after unpacking the arguments and setting up
// output buffers.
//
// Inputs:
// vertices (0): float array with 4*vertex_count elements.
// triangles (1): integer array of 3*triangle_count elements.
// face_culling_mode (2): integer, see rasterize_triangles_impl.h.
//
// Outputs:
// triangle_ids (0): integer array of size num_layers*image_height*image_width.
// z_buffer (1): float array of size num_layers*image_height*image_width.
// barycentric_coordinates (2): float array of size n*h*w*3.
ffi::Error rasterize_triangles_impl(
    cudaStream_t stream, diffren::FaceCullingMode face_culling_mode,
    ffi::BufferR2<ffi::F32> vertices, ffi::BufferR2<ffi::S32> triangles,
    ffi::ResultBufferR3<ffi::S32> triangle_ids,
    ffi::ResultBufferR3<ffi::F32> z_buffer,
    ffi::ResultBufferR4<ffi::F32> barycentric_coordinates) {
  int triangle_count = static_cast<int>(triangles.dimensions()[0]);
  auto out_dims = triangle_ids->dimensions();
  int num_layers = static_cast<int>(out_dims[0]);
  int image_height = static_cast<int>(out_dims[1]);
  int image_width = static_cast<int>(out_dims[2]);

  auto impl_status = RasterizeTrianglesImpl(
      vertices.typed_data(), triangles.typed_data(), triangle_count,
      image_width, image_height, num_layers, face_culling_mode,
      triangle_ids->typed_data(), z_buffer->typed_data(),
      barycentric_coordinates->typed_data(), stream);
  if (!impl_status.ok()) {
    return ffi::Error(static_cast<ffi::ErrorCode>(impl_status.code()),
                      std::string(impl_status.message()));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rasterize_triangles, rasterize_triangles_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<diffren::FaceCullingMode>("face_culling_mode")
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR3<ffi::S32>>()
        .Ret<ffi::BufferR3<ffi::F32>>()
        .Ret<ffi::BufferR4<ffi::F32>>());

py::dict Registrations() {
  pybind11::dict dict;
  dict["rasterize_triangles"] = pybind11::capsule(
      absl::bit_cast<void*>(&rasterize_triangles), "xla._CUSTOM_CALL_TARGET");
  return dict;
}

PYBIND11_MODULE(rasterize_triangles_gpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace cuda
}  // namespace diffren
