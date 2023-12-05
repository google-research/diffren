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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "diffren/common/kernels/rasterize_triangles_impl_cuda.cu.h"

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "diffren/jax/internal/kernels/descriptors.pb.h"
#include "include/pybind11/pybind11.h"
#include "xla/service/custom_call_status.h"

namespace py = pybind11;

namespace diffren {
namespace cuda {

// Calls RasterizeTriangleImpl after unpacking the arguments and setting up
// output buffers.
//
// Inputs:
// vertices (0): float array with 4*vertex_count elements.
// vertex_count (1): integer, number of vertices (currently unused).
// triangles (2): integer array of 3*triangle_count elements.
// triangle_count (3): integer, number of triangles.
// image_width (4): integer, width of output image.
// image_height (5): integer, height of output image.
// num_layers (6): integer, number of layers to render.
// face_culling_mode (7): integer, see rasterize_triangles_impl.h.
//
// Outputs:
// triangle_ids (0): integer array of size num_layers*image_height*image_width.
// z_buffer (1): float array of size num_layers*image_height*image_width.
// barycentric_coordinates (2): float array of size n*h*w*3.

void rasterize_triangles(cudaStream_t stream, void** buffers,
                         const char* opaque, std::size_t opaque_len,
                         XlaCustomCallStatus* status) {
  RasterizeTrianglesConfig descriptor;
  auto did_parse = descriptor.ParseFromString(std::string(opaque, opaque_len));
  if (!did_parse) {
    const char message[] = "RasterizeTrianglesConfig parsing failed.";
    XlaCustomCallStatusSetFailure(status, message, strlen(message));
    return;
  }

  int triangle_count = descriptor.triangle_count();
  int image_width = descriptor.image_width();
  int image_height = descriptor.image_height();
  int num_layers = descriptor.num_layers();
  auto face_culling_mode =
      static_cast<diffren::FaceCullingMode>(descriptor.face_culling_mode());

  const float* vertices = reinterpret_cast<const float*>(buffers[0]);
  const int* triangles = reinterpret_cast<const int*>(buffers[1]);
  int* triangle_ids = reinterpret_cast<int*>(buffers[2]);
  float* z_buffer = reinterpret_cast<float*>(buffers[3]);
  float* barycentric_coordinates = reinterpret_cast<float*>(buffers[4]);

  auto impl_status = RasterizeTrianglesImpl(
      vertices, triangles, triangle_count, image_width, image_height,
      num_layers, face_culling_mode, triangle_ids, z_buffer,
      barycentric_coordinates, stream);
  if (!impl_status.ok()) {
    XlaCustomCallStatusSetFailure(status, impl_status.message().data(),
                                  impl_status.message().size());
  }
}

py::dict Registrations() {
  pybind11::dict dict;
  dict["rasterize_triangles"] = pybind11::capsule(
      absl::bit_cast<void *>(&rasterize_triangles), "xla._CUSTOM_CALL_TARGET");
  return dict;
}

PYBIND11_MODULE(rasterize_triangles_gpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace cuda
}  // namespace diffren
