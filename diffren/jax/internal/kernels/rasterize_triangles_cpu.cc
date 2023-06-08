/* Copyright 2023 The diffren Authors.

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

#include "diffren/common/kernels/rasterize_triangles_impl.h"

#include "absl/base/casts.h"
#include "include/pybind11/pybind11.h"
#include "xla/service/custom_call_status.h"

namespace py = pybind11;

namespace diffren {

namespace {
template <typename T>
const T *Input(const void **args, int idx) {
  return reinterpret_cast<const T *>(args[idx]);
}

template <typename T>
T ScalarInput(const void **args, int idx) {
  return Input<T>(args, idx)[0];
}

template <typename T>
T *Output(void **args, int idx) {
  return reinterpret_cast<T *>(args[idx]);
}
}  // namespace

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
void rasterize_triangles(void **out, const void **in,
                         XlaCustomCallStatus *status) {
  int triangle_count = ScalarInput<int>(in, 3);
  int image_width = ScalarInput<int>(in, 4);
  int image_height = ScalarInput<int>(in, 5);
  int num_layers = ScalarInput<int>(in, 6);
  auto face_culling_mode =
      static_cast<diffren::FaceCullingMode>(ScalarInput<int>(in, 7));

  auto vertices = Input<float>(in, 0);
  auto triangles = Input<int>(in, 2);

  auto triangle_ids = Output<int>(out, 0);
  auto z_buffer = Output<float>(out, 1);
  auto barycentric_coordinates = Output<float>(out, 2);

  // Clear the triangle id to -1, barycentrics to zero, and the Z buffer to 1
  // (far plane in NDC coordinates).
  int num_pixels = num_layers * image_width * image_height;
  std::fill_n(triangle_ids, num_pixels, -1);
  std::fill_n(barycentric_coordinates, num_pixels * 3, 0.0);
  std::fill_n(z_buffer, num_pixels, 1.0);

  RasterizeTrianglesImpl(vertices, triangles, triangle_count, image_width,
                         image_height, num_layers, face_culling_mode,
                         triangle_ids, z_buffer, barycentric_coordinates);
}

py::dict Registrations() {
  pybind11::dict dict;
  dict["rasterize_triangles"] = pybind11::capsule(
      absl::bit_cast<void *>(&rasterize_triangles), "xla._CUSTOM_CALL_TARGET");
  return dict;
}

PYBIND11_MODULE(rasterize_triangles_cpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace diffren
