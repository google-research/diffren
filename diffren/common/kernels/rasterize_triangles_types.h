#ifndef DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_TYPES_H_
#define DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_TYPES_H_

namespace diffren {
// We define the face culling mode type here in a shared types file because:
// 1) We need to share the function call signature between the C++ and CUDA
// kernels, so this type needs to be exposed to both.
// 2) We don't want to share other things, like utility functions, as those are
// related to the implementation details of each device's kernel.
enum class FaceCullingMode { kNone = 0, kBack, kFront };

}  // namespace diffren

#endif  // DIFFREN_COMMON_KERNELS_RASTERIZE_TRIANGLES_TYPES_H_
