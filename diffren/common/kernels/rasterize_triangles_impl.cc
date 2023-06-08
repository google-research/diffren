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

#include "diffren/common/kernels/rasterize_triangles_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "diffren/common/kernels/rasterize_utils.h"

namespace diffren {

using fixed_t = int64_t;

void RasterizeTrianglesImpl(const float* vertices, const int32_t* triangles,
                            int32_t triangle_count, int32_t image_width,
                            int32_t image_height, int32_t num_layers,
                            FaceCullingMode face_culling_mode,
                            int32_t* triangle_ids, float* z_buffer,
                            float* barycentric_coordinates) {
  fixed_t unnormalized_matrix_inverse[9];
  fixed_t b_over_w[3];
  int left, right, bottom, top;

  for (int32_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
    const int32_t v0_x_id = 4 * triangles[3 * triangle_id];
    const int32_t v1_x_id = 4 * triangles[3 * triangle_id + 1];
    const int32_t v2_x_id = 4 * triangles[3 * triangle_id + 2];

    const float v0x = vertices[v0_x_id];
    const float v0y = vertices[v0_x_id + 1];
    const float v0z = vertices[v0_x_id + 2];
    const float v0w = vertices[v0_x_id + 3];

    const float v1x = vertices[v1_x_id];
    const float v1y = vertices[v1_x_id + 1];
    const float v1z = vertices[v1_x_id + 2];
    const float v1w = vertices[v1_x_id + 3];

    const float v2x = vertices[v2_x_id];
    const float v2y = vertices[v2_x_id + 1];
    const float v2z = vertices[v2_x_id + 2];
    const float v2w = vertices[v2_x_id + 3];

    const bool is_valid = ComputeTriangleBoundingBox(
        v0x, v0y, v0z, v0w, v1x, v1y, v1z, v1w, v2x, v2y, v2z, v2w, image_width,
        image_height, &left, &right, &bottom, &top);

    // Ignore triangles that do not overlap with any screen pixels.
    if (!is_valid) continue;

    ComputeUnnormalizedMatrixInverse(
        ToFixedPoint(v0x), ToFixedPoint(v1x), ToFixedPoint(v2x),
        ToFixedPoint(v0y), ToFixedPoint(v1y), ToFixedPoint(v2y),
        ToFixedPoint(v0w), ToFixedPoint(v1w), ToFixedPoint(v2w),
        face_culling_mode, unnormalized_matrix_inverse);

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy) {
      for (int ix = left; ix < right; ++ix) {
        ComputeEdgeFunctions(ix, iy, image_width, image_height,
                             unnormalized_matrix_inverse, b_over_w);
        if (!PixelIsInsideTriangle(b_over_w)) {
          continue;
        }

        const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
        const float b0 = b_over_w[0] / one_over_w;
        const float b1 = b_over_w[1] / one_over_w;
        const float b2 = b_over_w[2] / one_over_w;

        // Since we computed an unnormalized w above, we need to recompute
        // a properly scaled clip-space w value and then divide clip-space z
        // by that.
        const float clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
        const float clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
        const float z = clip_z / clip_w;

        // Skip the pixel if it is beyond the near or far clipping plane.
        if (z < -1.0f || z > 1.0f) continue;

        // Insert into appropriate depth layer with insertion sort.
        float z_next = z;
        int32_t id_next = triangle_id;
        float b0_next = b0;
        float b1_next = b1;
        float b2_next = b2;
        const int pixel_idx0 = iy * image_width + ix;
        for (int layer = 0; layer < num_layers; ++layer) {
          const int pixel_idx = pixel_idx0 + image_height * image_width * layer;
          if (z_next < z_buffer[pixel_idx]) {
            std::swap(z_next, z_buffer[pixel_idx]);
            std::swap(id_next, triangle_ids[pixel_idx]);
            if (barycentric_coordinates != nullptr) {
              std::swap(b0_next, barycentric_coordinates[3 * pixel_idx + 0]);
              std::swap(b1_next, barycentric_coordinates[3 * pixel_idx + 1]);
              std::swap(b2_next, barycentric_coordinates[3 * pixel_idx + 2]);
            }
          }
          // Exit the loop early if the clear depth (z == 1) is reached.
          if (z_next == 1) break;
        }
      }
    }
  }
}

}  // namespace diffren
