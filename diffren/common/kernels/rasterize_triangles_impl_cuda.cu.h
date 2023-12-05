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

#ifndef DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_IMPL_CUDA_CU_H_
#define DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_IMPL_CUDA_CU_H_

#include "diffren/common/kernels/rasterize_triangles_types.h"

#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace diffren::cuda {

// The main CUDA rasterizer entry point. Shares a signature with the CPU
// version, although all pointers point to GPU memory and it has an optional
// stream argument. Note that this function call is asynchronous with respect
// to the host- it does not block waiting for the results of the computation.
// It returns as soon as the GPU kernel launches have been requested
// successfully. Note this means that later cuda errors could have come from
// this call.
// Important note- the barycentric coordinates pointer must be at least 8-byte
// aligned (this is guaranteed by cudaMalloc). The code checks for this
// condition.
absl::Status RasterizeTrianglesImpl(const float* vertices_d,
                                    const int32_t* triangles_d,
                                    int32_t triangle_count, int32_t image_width,
                                    int32_t image_height, int32_t num_layers,
                                    diffren::FaceCullingMode face_culling_mode,
                                    int32_t* triangle_ids_d, float* z_buffer_d,
                                    float* barycentric_coordinates_d,
                                    cudaStream_t stream = nullptr);

// A utility function that shares a call signature with the CPU rasterizer.
// All pointers are standard CPU pointers. This function handles allocating
// GPU buffers and copying back and forth between device and host. It is also
// synchronous, so upon return of this function all CPU-side pointers are
// available and filled with the result of the rasterizer call. This function
// is mainly intended for testing/one-off calls.
void RasterizeTrianglesImplWithHostPointers(
    const float* vertices_h, const int32_t* triangles_h, int32_t triangle_count,
    int32_t image_width, int32_t image_height, int32_t num_layers,
    diffren::FaceCullingMode face_culling_mode, int32_t* triangle_ids,
    float* z_buffer, float* barycentric_coordinates);

}  // namespace diffren::cuda

#endif  // DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_IMPL_CUDA_CU_H_
