# Copyright 2022 The diffren Authors.
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
