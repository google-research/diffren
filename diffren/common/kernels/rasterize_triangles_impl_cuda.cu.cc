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


#include "diffren/common/kernels/rasterize_triangles_impl_cuda.cu.h"

// CHECK_OK and friends from glog/logging.h do not seem to place nicely with
// nvcc, so we need to change them. These simpler versions just log and do
// not crash the program if the CHECK fails.
#define CHECK_OK(x) if (!(x.ok())) std::cerr << "CHECK_OK failed: "
#define CHECK_EQ(x, y) if (x != y) std::cerr << "CHECK_EQ failed: "
#define CHECK_NE(x, y) if (x == y) std::cerr << "CHECK_NE failed: "
#define CHECK_GE(x, y) if (x < y) std::cerr << "CHECK_GE failed: "
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "diffren/common/kernels/rasterize_utils.h"

namespace {

using diffren::ComputeEdgeFunctions;
using diffren::ComputeUnnormalizedMatrixInverse;
using diffren::fixed_t;
using diffren::ToFixedPoint;

// Highest exactly representable IEEE-754 float below uint32_t max / 2 (CUDA
// from 1.4 onwards uses IEEE-754)
// __device__ constexpr float kDiscretizationScale = 2147483520.0f;
// For now we use a double, which requires less care because it has more
// significant figures (~15) than the uint32_t we store the discretized values
// in (10).
__device__ constexpr double kDiscretizationScale =
    std::numeric_limits<uint32_t>::max() / 2;

// The clipping plane values in clip space. Note that the implementation depends
// heavily on these values (e.g., the discretization scale above is chosen to be
// valid in the input range [-1, 1]). Changing it without adjusting all code
// related to z or clipping operations could break the rasterizer.
__device__ constexpr float kFarClippingPlane = 1.0f;
__device__ constexpr float kNearClippingPlane = -1.0f;

// The initial triangle id for the buffer. Note that 0 is a valid triangle id,
// it is distinguished from the background because the barycentrics will be
// 0 on the background.
__device__ constexpr int32_t kInitialTriangleId = 0;

std::string PrintCudaError(cudaError_t cuda_error_code) {
  return std::string(cudaGetErrorString(cuda_error_code));
}

absl::Status CreateUninitializedDeviceBuffer(size_t arr_byte_count,
                                             void** arr_d,
                                             const std::string& name) {
  cudaError_t error_code = cudaMalloc(arr_d, arr_byte_count);
  if (error_code != cudaSuccess) {
    return absl::UnknownError("cudaMalloc " + name + " failed with error " +
                              PrintCudaError(error_code));
  }
  return absl::OkStatus();
}

absl::Status ClearDeviceBuffer(void* arr_d, size_t arr_byte_count,
                               const std::string& name) {
  cudaError_t error_code = cudaMemsetAsync(arr_d, 0, arr_byte_count);
  if (error_code != cudaSuccess) {
    return absl::UnknownError("cudaMemset " + name + " failed with error " +
                              PrintCudaError(error_code));
  }
  return absl::OkStatus();
}

absl::Status CreateDeviceCopyOfArray(const void* arr_h, size_t arr_byte_count,
                                     void** arr_d, const std::string& name) {
  absl::Status status =
      CreateUninitializedDeviceBuffer(arr_byte_count, arr_d, name);
  if (!status.ok()) return status;
  cudaError_t error_code =
      cudaMemcpy(*arr_d, arr_h, arr_byte_count, cudaMemcpyHostToDevice);
  if (error_code != cudaSuccess) {
    return absl::UnknownError("cudaMemcpy " + name +
                              " h->g failed with error " +
                              PrintCudaError(error_code));
  }
  if (arr_d == nullptr) {
    return absl::UnknownError("cudaMemcpy " + name +
                              " did not fill out device ptr.");
  }
  return absl::OkStatus();
}

void* CreateDeviceCopyOfArrayOrDie(const void* arr_h, size_t arr_byte_count,
                                   const std::string& name) {
  void* arr_d = nullptr;
  CHECK_OK(CreateDeviceCopyOfArray(arr_h, arr_byte_count, &arr_d, name));
  CHECK_NE(arr_d, nullptr)
      << "CreateDeviceCopyOfArray did not fill out device ptr for " << name;
  return arr_d;
}

absl::Status CopyDeviceToHostAndFree(void* arr_h, size_t arr_byte_count,
                                     void* arr_d, const std::string& name) {
  if (arr_h == nullptr) {
    return absl::UnknownError("Input host buffer is nullptr");
  }
  cudaError_t error_code =
      cudaMemcpy(arr_h, arr_d, arr_byte_count, cudaMemcpyDeviceToHost);
  if (error_code != cudaSuccess || cudaFree(arr_d) != cudaSuccess) {
    return absl::UnknownError("cudaMemcpy+free " + name +
                              " d->h failed with error " +
                              PrintCudaError(error_code));
  }
  return absl::OkStatus();
}

absl::Status FreeDeviceBuffer(void* arr_d, const std::string& name) {
  cudaError_t error_code = cudaFree(arr_d);
  if (error_code != cudaSuccess) {
    return absl::UnknownError("cudaFree " + name + " failed with error " +
                              PrintCudaError(error_code));
  }
  return absl::OkStatus();
}

// We inline all the cuda-specific device utility functions rather than
// spreading them across translation units (like the C++ rasterizer does) or
// sharing them, because it is difficult to expose device functions in
// library code using cuda build rules.

// Discretize the floating point z value in [-1.0f, 1.0f] to an integer in
// [0, uint32 max]. Note the parenthesis are important because 0.5f * uint
// limit is exactly representable. So -1.0f becomes exactly 0, 1.0f becomes
// exactly uint32 max. The uint32 limit is not exactly representable.
// Important: Assumes -1.0f <= z <= 1.0f.
__device__ constexpr uint32_t DiscretizeZ(float z) {
  // 1.0 as a double is intentional- promotes z to double precision before
  // discretizing to an integer.
  uint32_t discretized = (z + 1.0) * kDiscretizationScale;
  return discretized;
}

// Returns a properly scale floating point z value from a discretized one.
// Values inside [-1.0f, 1.0f] that are discretized are restored to the same
// range. Not guaranteed to undo the discretization exactly (but the depth
// buffer accuracy is not critical at this point).
__device__ constexpr float UndiscretizeZ(uint32_t discretized_z) {
  // Implicitly promotes discretized_z to double, then back down to float.
  // The divide is heavy (could consider reciprocal) but we don't need to unpack
  // in the inner loop, only pack.
  return ((discretized_z / kDiscretizationScale) - 1);
}

// Packs two 32-bit values into a uint64 so that they can be atomically written
// to global memory together with the cuda atomic op. Importantly, the
// discretized z value resides in the MSB, so that min/max z buffer comparisons
// can happen directly on the 64 bit value. Note- this function should not be
// called with a negative triangle id.
__device__ constexpr uint64_t PackDepthAndTriangle(float z,
                                                   int32_t triangle_id) {
  uint32_t discretized_z = DiscretizeZ(z);
  uint64_t val = discretized_z;
  uint32_t unsigned_id = triangle_id;  // Undefined if triangle id is negative!
  val = (val << 32) + unsigned_id;
  return val;
}

// Reverses the PackDepthAndTriangle() operation. Fills triangle_id and z with
// the contents of the previously packed uint64 value.
__device__ constexpr void UnpackDepthAndTriangle(uint64_t packed, float* z,
                                                 int32_t* triangle_id) {
  *triangle_id = 0xFFFFFFFF & packed;     // Truncate away MSB (64 -> 32 bit).
  uint32_t discretized_z = packed >> 32;  // Extract MSB.
  *z = UndiscretizeZ(discretized_z);
}

// The initial value that is written to the packed buffer for an empty
// pixel. Normally constants would go at the top, but this has to go after
// the Packing function (forward declaration doesn't work with constexpr).
__device__ constexpr uint64_t kInitialBufferValue =
    PackDepthAndTriangle(kFarClippingPlane, kInitialTriangleId);

// Inverts the matrix associated with a single triangle (in fixed point, without
// normalizing)
__device__ void ComputeTriangleUnnormalizedMatrixInverse(
    const float* vertices_d, const int32_t* triangle,
    diffren::FaceCullingMode face_culling_mode,
    fixed_t* unnormalized_matrix_inverse) {
  const int32_t v0_x_id = 4 * triangle[0];
  const int32_t v1_x_id = 4 * triangle[1];
  const int32_t v2_x_id = 4 * triangle[2];

  const float v0x = vertices_d[v0_x_id];
  const float v0y = vertices_d[v0_x_id + 1];
  const float v0w = vertices_d[v0_x_id + 3];

  const float v1x = vertices_d[v1_x_id];
  const float v1y = vertices_d[v1_x_id + 1];
  const float v1w = vertices_d[v1_x_id + 3];

  const float v2x = vertices_d[v2_x_id];
  const float v2y = vertices_d[v2_x_id + 1];
  const float v2w = vertices_d[v2_x_id + 3];

  ComputeUnnormalizedMatrixInverse(
      ToFixedPoint(v0x), ToFixedPoint(v1x), ToFixedPoint(v2x),
      ToFixedPoint(v0y), ToFixedPoint(v1y), ToFixedPoint(v2y),
      ToFixedPoint(v0w), ToFixedPoint(v1w), ToFixedPoint(v2w),
      face_culling_mode, unnormalized_matrix_inverse);
}

// Computes the barycentric coordinates at a point (assumes the point is valid).
__device__ void ComputeBarycentrics(int ix, int iy, int image_width,
                                    int image_height,
                                    const fixed_t* unnormalized_matrix_inverse,
                                    float* barycentric_coordinates) {
  fixed_t b_over_w[3];
  ComputeEdgeFunctions(ix, iy, image_width, image_height,
                       unnormalized_matrix_inverse, b_over_w);
  const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
  barycentric_coordinates[0] = b_over_w[0] / one_over_w;
  barycentric_coordinates[1] = b_over_w[1] / one_over_w;
  barycentric_coordinates[2] = b_over_w[2] / one_over_w;
}

}  // namespace

namespace diffren::cuda {

__global__ void InitializePackedBufferKernel(uint64_t* packed_buffer_d,
                                             int pixel_count) {
  int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (pixel_idx >= pixel_count) return;
  packed_buffer_d[pixel_idx] = kInitialBufferValue;
}

__global__ void RasterizeTrianglesImplKernel(
    const float* vertices_d, const int32_t* triangles_d, int32_t triangle_count,
    int32_t image_width, int32_t image_height, int32_t num_layers,
    FaceCullingMode face_culling_mode, uint64_t* packed_buffer_d) {
  // The actual device code. For most uses cases, don't call this directly.
  // Instead, use RasterizeTrianglesImpl() with a device pointer. Only call this
  // function if you want to manage the kernel launch parameters yourself and
  // will ensure that all triangles are rasterized.
  fixed_t unnormalized_matrix_inverse[9];
  fixed_t b_over_w[3];
  int left, right, bottom, top;

  int triangle_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (triangle_id >= triangle_count) return;

  const int32_t v0_x_id = 4 * triangles_d[3 * triangle_id];
  const int32_t v1_x_id = 4 * triangles_d[3 * triangle_id + 1];
  const int32_t v2_x_id = 4 * triangles_d[3 * triangle_id + 2];

  const float v0x = vertices_d[v0_x_id];
  const float v0y = vertices_d[v0_x_id + 1];
  const float v0z = vertices_d[v0_x_id + 2];
  const float v0w = vertices_d[v0_x_id + 3];

  const float v1x = vertices_d[v1_x_id];
  const float v1y = vertices_d[v1_x_id + 1];
  const float v1z = vertices_d[v1_x_id + 2];
  const float v1w = vertices_d[v1_x_id + 3];

  const float v2x = vertices_d[v2_x_id];
  const float v2y = vertices_d[v2_x_id + 1];
  const float v2z = vertices_d[v2_x_id + 2];
  const float v2w = vertices_d[v2_x_id + 3];

  const bool is_valid = ComputeTriangleBoundingBox(
      v0x, v0y, v0z, v0w, v1x, v1y, v1z, v1w, v2x, v2y, v2z, v2w, image_width,
      image_height, &left, &right, &bottom, &top);

  // Ignore triangles that do not overlap with any screen pixels.
  if (!is_valid) return;

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
      if (z < kNearClippingPlane || z > kFarClippingPlane) continue;

      // Insert into appropriate depth layer with insertion sort (requires
      // contending with other threads)
      uint64_t to_insert = PackDepthAndTriangle(z, triangle_id);

      // We want to push this value into its proper layer in the z buffer for
      // the pixel. At each step, we atomically min the value and receive
      // either a previous replaced value or the current one. By max-ing the
      // two values, we merge the possible cases: 1) the value was successfully
      // replaced and we received an old value bigger than ours, which needs to
      // get pushed later in the list. 2) the value was not greater that what
      // was in the list, so we want to push our value later in the list. In
      // both cases, the larger of the two values in question should be inserted
      // into the next layer. Note that this implementation assumes all values
      // are unique. This is currently guaranteed because the LSB contains the
      // triangle id, which is present at most once per pixel.
      for (int layer = 0; layer < num_layers; layer++) {
        const int pixel_idx =
            image_height * image_width * layer + image_width * iy + ix;
        // NOLINTBEGIN (unsigned long long conversion required by CUDA)
        uint64_t old = atomicMin(
            reinterpret_cast<unsigned long long*>(&packed_buffer_d[pixel_idx]),
            to_insert);  // uint64 -> unsigned long long (should be a no-op)
        // NOLINTEND
        to_insert = max(to_insert, old);
        // We don't need to push down the default value.
        if (old == kInitialBufferValue) break;
      }
    }
  }
}

__global__ void FinalizeDepthBufferKernel(const uint64_t* packed_buffer_d,
                                          int32_t pixel_count,
                                          int32_t* triangle_ids_d,
                                          float* z_buffer_d) {
  // Currently the depth and triangle id buffers are interleaved so that
  // the 64-bit atomic operations work. They are not useful in that form.
  // So now split them into the z buffer and triangle ids.
  int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (pixel_idx >= pixel_count) return;
  uint64_t buf_value = packed_buffer_d[pixel_idx];
  float buf_z;
  int32_t buf_triangle_id;
  UnpackDepthAndTriangle(buf_value, &buf_z, &buf_triangle_id);
  z_buffer_d[pixel_idx] = buf_z;
  triangle_ids_d[pixel_idx] = buf_triangle_id;
}

__global__ void SetBarycentricsKernel(
    const float* vertices_d, const int32_t* triangles_d,
    const int32_t* triangle_ids_d, const float* z_buffer_d,
    int32_t triangle_count, int32_t image_width, int32_t image_height,
    int32_t num_layers, FaceCullingMode face_culling_mode,
    float* barycentric_coordinates_d) {
  // Recompute the barycentrics for the final triangle ids that were chosen.
  // This time, we parallelize over pixels rather than triangles, so even
  // though we have to recompute, it is much less time consuming (plus total
  // work done is less, since we don't compute them any occluded fragments).
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int layer = blockDim.z * blockIdx.z + threadIdx.z;
  if (ix >= image_width) return;
  if (iy >= image_height) return;
  if (layer >= num_layers) return;

  fixed_t unnormalized_matrix_inverse[9];

  int pixel_idx = layer * image_width * image_height + iy * image_width + ix;
  int pixel_count = image_width * image_height * num_layers;
  if (pixel_idx < 0 || pixel_idx >= pixel_count) return;
  float z = z_buffer_d[pixel_idx];
  if (z == kFarClippingPlane) return;
  int32_t triangle_id = triangle_ids_d[pixel_idx];
  // Now that we have the necessary info, recompute the barycentrics:
  ComputeTriangleUnnormalizedMatrixInverse(
      vertices_d, &triangles_d[3 * triangle_id], face_culling_mode,
      unnormalized_matrix_inverse);
  ComputeBarycentrics(ix, iy, image_width, image_height,
                      unnormalized_matrix_inverse,
                      &barycentric_coordinates_d[3 * pixel_idx]);
}

absl::Status RasterizeTrianglesImpl(
    const float* vertices_d, const int32_t* triangles_d, int32_t triangle_count,
    int32_t image_width, int32_t image_height, int32_t num_layers,
    FaceCullingMode face_culling_mode, int32_t* triangle_ids_d,
    float* z_buffer_d, float* barycentric_coordinates_d, cudaStream_t stream) {
  // The main entry point, assuming you already have your arrays on the device.
  CHECK_GE(triangle_count, 0) << "Invalid input length.";
  CHECK_NE(barycentric_coordinates_d, nullptr)
      << "Barycentric coordinate buffer is required.";

  // In order to avoid an extra buffer, we use the barycentrics buffer as the
  // packed buffer (we could consider adding support for a variant that does not
  // compute the barycentric buffer, but it would need an extra input device
  // buffer for scratch space). We shouldn't have to worry about strict aliasing
  // issues because there is never a strict aliasing violation in host code, nor
  // in any individual CUDA kernel. Note that memory alignment is also not a
  // problem because cudaMalloc ensures returned memory is suitable for any
  // object type, but we double check.
  CHECK_EQ(reinterpret_cast<uintptr_t>(barycentric_coordinates_d) %
               alignof(uint64_t),
           0)
      << "Barycentrics buffer is not correctly aligned; please use cudaMalloc";
  uint64_t* packed_buffer_d =
      reinterpret_cast<uint64_t*>(barycentric_coordinates_d);

  int threads_per_block = 256;  // One block fits on a SM; gets chunked into
                                // warps and shares shared memory/SM caches.
  int pixel_count = image_width * image_height * num_layers;
  int blocks_per_grid =
      (pixel_count + threads_per_block - 1) / threads_per_block;
  // First kernel- initialize the packed buffer to clear the z buffer.
  InitializePackedBufferKernel<<<blocks_per_grid, threads_per_block, 0,
                                 stream>>>(packed_buffer_d, pixel_count);
  cudaError_t error_code = cudaGetLastError();
  if (error_code != cudaSuccess) {
    return absl::UnknownError(
        "InitializePackedBufferKernel kernel launch failed with error " +
        PrintCudaError(error_code));
  }

  // Second kernel- the actual rasterization to correctly fill the packed buffer
  // with triangle ids and z values.
  blocks_per_grid =
      (triangle_count + threads_per_block - 1) / threads_per_block;
  RasterizeTrianglesImplKernel<<<blocks_per_grid, threads_per_block, 0,
                                 stream>>>(
      vertices_d, triangles_d, triangle_count, image_width, image_height,
      num_layers, face_culling_mode, packed_buffer_d);
  error_code = cudaGetLastError();
  if (error_code != cudaSuccess) {
    return absl::UnknownError(
        "RasterizeTrianglesImplKernel kernel launch failed with error " +
        PrintCudaError(error_code));
  }

  // Third kernel- fill the z and triangle id buffers using the information
  // in the packed buffer.
  blocks_per_grid = (pixel_count + threads_per_block - 1) / threads_per_block;
  FinalizeDepthBufferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      packed_buffer_d, pixel_count, triangle_ids_d, z_buffer_d);
  error_code = cudaGetLastError();
  if (error_code != cudaSuccess) {
    return absl::UnknownError(
        "FinalizeDepthBufferKernel kernel launch failed with error " +
        PrintCudaError(error_code));
  }

  cudaMemsetAsync(barycentric_coordinates_d, 0, 3 * pixel_count * sizeof(float),
                  stream);

  // Fourth kernel- fill the barycentrics buffer.
  dim3 block_size(256, 1, 1);
  int32_t grid_x = (image_width + block_size.x - 1) / block_size.x;
  int32_t grid_y = (image_height + block_size.y - 1) / block_size.y;
  int32_t grid_z = (num_layers + block_size.z - 1) / block_size.z;
  dim3 grid_size(grid_x, grid_y, grid_z);
  SetBarycentricsKernel<<<grid_size, block_size, 0, stream>>>(
      vertices_d, triangles_d, triangle_ids_d, z_buffer_d, triangle_count,
      image_width, image_height, num_layers, face_culling_mode,
      barycentric_coordinates_d);
  error_code = cudaGetLastError();
  if (error_code != cudaSuccess) {
    return absl::UnknownError(
        "SetBarycentricsKernel kernel launch failed with error " +
        PrintCudaError(error_code));
  }
  return absl::OkStatus();
}

void RasterizeTrianglesImplWithHostPointers(
    const float* vertices_h, const int32_t* triangles_h, int32_t triangle_count,
    int32_t image_width, int32_t image_height, int32_t num_layers,
    FaceCullingMode face_culling_mode, int32_t* triangle_ids_h,
    float* z_buffer_h, float* barycentric_coordinates_h) {
  // A utility for launching the kernel that first copies the data to the device
  // and then copies back to the CPU afterwards, so its signature is a standard
  // C++ function that takes in standard CPU-side pointers. Easier to use but
  // inefficient for any use cases that involve calling the rasterizer more
  // than once. If GPU operations fail, will CHECK fail.

  // Vertices (We could require a vertex count to avoid the totally unnecessary
  // array scan, but this function is already just a slow convenience wrapper
  // around the device code, so we prefer to keep an identical interface to the
  // standard CPU version instead).
  size_t n_verts =
      *std::max_element(triangles_h, triangles_h + 3 * triangle_count) + 1;
  size_t vertices_byte_count = n_verts * 4 * sizeof(float);
  float* vertices_d = static_cast<float*>(CreateDeviceCopyOfArrayOrDie(
      vertices_h, vertices_byte_count, "vertices"));
  // End vertices

  // Triangles
  size_t triangles_byte_count = triangle_count * 3 * sizeof(int32_t);
  int32_t* triangles_d = static_cast<int32_t*>(CreateDeviceCopyOfArrayOrDie(
      triangles_h, triangles_byte_count, "triangles"));
  // End triangles

  // Triangle ids:
  size_t num_pixels = image_height * image_width * num_layers;
  size_t triangle_ids_byte_count = num_pixels * sizeof(int32_t);
  int32_t* triangle_ids_d = static_cast<int32_t*>(CreateDeviceCopyOfArrayOrDie(
      triangle_ids_h, triangle_ids_byte_count, "triangle_ids"));
  // End triangle ids

  // z buffer
  size_t z_buffer_byte_count = num_pixels * sizeof(float);
  float* z_buffer_d = static_cast<float*>(CreateDeviceCopyOfArrayOrDie(
      z_buffer_h, z_buffer_byte_count, "z_buffer"));
  // End z buffer.

  // Barycentrics
  CHECK_NE(barycentric_coordinates_h, nullptr);  // Allowed in C++ kernel.
  size_t barycentrics_byte_count = num_pixels * 3 * sizeof(float);
  float* barycentric_coordinates_d =
      static_cast<float*>(CreateDeviceCopyOfArrayOrDie(
          barycentric_coordinates_h, barycentrics_byte_count, "barycentrics"));
  // End barycentrics.

  // Kernel Launches
  absl::Status kernel_status = RasterizeTrianglesImpl(
      vertices_d, triangles_d, triangle_count, image_width, image_height,
      num_layers, face_culling_mode, triangle_ids_d, z_buffer_d,
      barycentric_coordinates_d);
  CHECK_OK(kernel_status);
  // End kernels

  // Copy the result back to the input host buffers and free the device buffers.
  CHECK_OK(CopyDeviceToHostAndFree(triangle_ids_h, triangle_ids_byte_count,
                                   triangle_ids_d, "triangle_ids"));
  CHECK_OK(CopyDeviceToHostAndFree(z_buffer_h, z_buffer_byte_count, z_buffer_d,
                                   "z_buffer"));
  CHECK_OK(CopyDeviceToHostAndFree(barycentric_coordinates_h,
                                   barycentrics_byte_count,
                                   barycentric_coordinates_d, "barycentrics"));
  CHECK_OK(FreeDeviceBuffer(vertices_d, "vertices"));
  CHECK_OK(FreeDeviceBuffer(triangles_d, "triangles"));
}

}  // namespace diffren::cuda
