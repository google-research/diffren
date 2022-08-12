#include <string>
#include <vector>

#include "diffren/common/kernels/rasterize_triangles_impl_cuda.cu.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "diffren/common/kernels/rasterize_test_utils.h"

namespace diffren {

namespace {

using diffren::FaceCullingMode;
using diffren::cuda::RasterizeTrianglesImplWithHostPointers;
using diffren::test_utils::CUBE_TRIANGLES;
using diffren::test_utils::CUBE_VERTICES;
using diffren::test_utils::ExpectImageFileAndImageAreNear;
using diffren::test_utils::ImageBuffer4_b;
using diffren::test_utils::LARGE_CUBE_TRIANGLES;
using diffren::test_utils::LARGE_CUBE_VERTICES;

// TODO(kgenova) We could probably merge the cuda and c++ rasterizer tests
// by using some base class API or templating and shared utility functions.
class RasterizeTrianglesImplCudaTest : public testing::Test {
 protected:
  void CallRasterizeTrianglesImplCudaWithParameters(
      const float* vertices_h, const int32_t* triangles_h,
      int32_t triangle_count, int num_layers,
      diffren::FaceCullingMode culling_mode) {
    const int num_pixels = num_layers * image_height_ * image_width_;
    triangle_ids_buffer_.resize(num_pixels);

    constexpr float kClearDepth = 1.0;
    z_buffer_.resize(num_pixels, kClearDepth);

    barycentrics_buffer_.resize(num_pixels * 3);

    RasterizeTrianglesImplWithHostPointers(
        vertices_h, triangles_h, triangle_count, image_width_, image_height_,
        num_layers, culling_mode, triangle_ids_buffer_.data(), z_buffer_.data(),
        barycentrics_buffer_.data());
  }

  void CallRasterizeTrianglesImplCuda(const float* vertices,
                                      const int32_t* triangles,
                                      int32_t triangle_count) {
    CallRasterizeTrianglesImplCudaWithParameters(
        vertices, triangles, triangle_count, 1, FaceCullingMode::kNone);
  }

  ImageBuffer4_b EncodeBarycentricsBuffer(const int layer_idx = 0) const {
    return test_utils::EncodeBarycentricsBuffer(
        barycentrics_buffer_, image_height_, image_width_, layer_idx);
  }

  ImageBuffer4_b EncodeIdBuffer(const int layer_idx = 0) const {
    return test_utils::EncodeIdBuffer(triangle_ids_buffer_, image_height_,
                                      image_width_, 1, layer_idx);
  }

  // Expects that the sum of barycentric weights at a pixel is close to a
  // given value.
  void ExpectBarycentricSumIsNear(int x, int y, float expected) const {
    constexpr float kEpsilon = 1e-6f;
    auto it = barycentrics_buffer_.begin() + y * image_width_ * 3 + x * 3;
    EXPECT_NEAR(*it + *(it + 1) + *(it + 2), expected, kEpsilon);
  }
  // Expects that a pixel is covered by verifying that its barycentric
  // coordinates sum to one.
  void ExpectIsCovered(int x, int y) const {
    ExpectBarycentricSumIsNear(x, y, 1.0);
  }
  // Expects that a pixel is not covered by verifying that its barycentric
  // coordinates sum to zero.
  void ExpectIsNotCovered(int x, int y) const {
    ExpectBarycentricSumIsNear(x, y, 0.0);
  }

  int image_height_ = 480;
  int image_width_ = 640;
  std::vector<float> barycentrics_buffer_;
  std::vector<int32_t> triangle_ids_buffer_;
  std::vector<float> z_buffer_;
};

class RasterizeTrianglesImplBarysTest
    : public RasterizeTrianglesImplCudaTest,
      public ::testing::WithParamInterface<bool> {
 protected:
  void CallRasterizeTrianglesImpl(const float* vertices,
                                  const int32_t* triangles,
                                  int32_t triangle_count, int num_layers) {
    CallRasterizeTrianglesImplCudaWithParameters(vertices, triangles,
                                                 triangle_count, num_layers,
                                                 FaceCullingMode::kNone);
  }
};

class RasterizeTrianglesImplCullTest
    : public RasterizeTrianglesImplCudaTest,
      public ::testing::WithParamInterface<FaceCullingMode> {
 protected:
  void CallRasterizeTrianglesImpl(const float* vertices,
                                  const int32_t* triangles,
                                  int32_t triangle_count, int num_layers,
                                  FaceCullingMode culling_mode) {
    CallRasterizeTrianglesImplCudaWithParameters(
        vertices, triangles, triangle_count, num_layers, culling_mode);
  }
};

TEST_F(RasterizeTrianglesImplCudaTest, CanRasterizeTriangle) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
                                       0.3,  1.0,  0.5, -0.5, 0.3, 1.0};
  const std::vector<int32_t> triangles = {0, 1, 2};

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 1);

  ExpectImageFileAndImageAreNear("Simple_Triangle.png",
                                 EncodeBarycentricsBuffer(), "simple triangle",
                                 "simple triangle does not match");
}

TEST_F(RasterizeTrianglesImplCudaTest, CanRasterizeExternalTriangle) {
  const std::vector<float> vertices = {-0.5,  -0.5, 0.99, 1.0,  0.0,  -0.5,
                                       -1.01, -1.0, 0.5,  -0.5, 0.99, 1.0};
  const std::vector<int32_t> triangles = {0, 1, 2};

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 1);

  ExpectImageFileAndImageAreNear(
      "External_Triangle.png", EncodeBarycentricsBuffer(), "external triangle",
      "external triangle does not match");
}

TEST_F(RasterizeTrianglesImplCudaTest, CanRasterizeCameraInsideBox) {
  CallRasterizeTrianglesImplCuda(LARGE_CUBE_VERTICES, LARGE_CUBE_TRIANGLES, 12);

  ExpectImageFileAndImageAreNear("Inside_Box.png", EncodeBarycentricsBuffer(),
                                 "camera inside box",
                                 "camera inside box does not match");
}

TEST_F(RasterizeTrianglesImplCudaTest, CanRasterizeTetrahedron) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
                                       0.3,  1.0,  0.5, -0.5, 0.3, 1.0,
                                       0.0,  0.0,  0.0, 1.0};
  const std::vector<int32_t> triangles = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 4);

  ExpectImageFileAndImageAreNear(
      "Simple_Tetrahedron.png", EncodeBarycentricsBuffer(),
      "simple tetrahedron", "simple tetrahedron does not match");
}

TEST_F(RasterizeTrianglesImplCudaTest, CanRasterizeSphereWithManyTriangles) {
  image_width_ = 1600;   // Large to avoid pixel-sized/missing triangles
  image_height_ = 1600;  // Square so NDC-sphere is round.
  std::vector<float> vertices_xyz;
  std::vector<int32_t> triangles;
  diffren::test_utils::UVSphere(230, 230, 0.9f, &vertices_xyz, &triangles);
  std::vector<float> vertices =
      diffren::test_utils::AddHomogeneousCoord(vertices_xyz);

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(),
                                 triangles.size() / 3);

  ExpectImageFileAndImageAreNear("UV_Sphere_Ids.png", EncodeIdBuffer(),
                                 "230x230 uv sphere",
                                 "uv sphere does not match");
}

TEST_F(RasterizeTrianglesImplCudaTest, WorksWhenPixelIsOnTriangleEdge) {
  // Verifies that a pixel that lies exactly on a triangle edge is considered
  // inside the triangle.
  image_width_ = 641;
  const int x_pixel = image_width_ / 2;
  const float x_ndc = 0.0;
  constexpr int yPixel = 5;

  const std::vector<float> vertices = {x_ndc, -1.0, 0.5, 1.0,  x_ndc, 1.0,
                                       0.5,   1.0,  0.5, -1.0, 0.5,   1.0};
  {
    const std::vector<int32_t> triangles = {0, 1, 2};

    CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
  {
    // Test the triangle with the same vertices in reverse order.
    const std::vector<int32_t> triangles = {2, 1, 0};

    CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
}

TEST_F(RasterizeTrianglesImplCudaTest, CoversEdgePixelsOfImage) {
  // Verifies that the pixels along image edges are correctly covered.

  const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0, 1.0, -1.0,
                                       0.0,  1.0,  1.0, 1.0, 0.0, 1.0,
                                       -1.0, 1.0,  0.0, 1.0};
  const std::vector<int32_t> triangles = {0, 1, 2, 0, 2, 3};

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 2);

  ExpectIsCovered(0, 0);
  ExpectIsCovered(image_width_ - 1, 0);
  ExpectIsCovered(image_width_ - 1, image_height_ - 1);
  ExpectIsCovered(0, image_height_ - 1);
}

TEST_F(RasterizeTrianglesImplCudaTest, PixelOnDegenerateTriangleIsNotInside) {
  // Verifies that a pixel lying exactly on a triangle with zero area is
  // counted as lying outside the triangle.
  image_width_ = 1;
  image_height_ = 1;
  const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
                                       0.0,  1.0,  0.0, 0.0, 0.0, 1.0};
  const std::vector<int32_t> triangles = {0, 1, 2};

  CallRasterizeTrianglesImplCuda(vertices.data(), triangles.data(), 1);

  ExpectIsNotCovered(0, 0);
}

TEST_F(RasterizeTrianglesImplBarysTest, CanRasterizeCubeWithTwoLayers) {
  CallRasterizeTrianglesImpl(CUBE_VERTICES, CUBE_TRIANGLES, 12, 2);

  // First layer ids.
  ExpectImageFileAndImageAreNear("Ids_Cube.png", EncodeIdBuffer(), "cube_ids",
                                 "cube ids do not match");

  // Second layer ids.
  ExpectImageFileAndImageAreNear("Ids_Cube_Back.png", EncodeIdBuffer(1),
                                 "cube_back_ids", "cube back ids do not match");

  // First layer barycentrics.
  ExpectImageFileAndImageAreNear("Barycentrics_Cube.png",
                                 EncodeBarycentricsBuffer(), "cube_barys",
                                 "cube barycentrics do not match");

  // Second layer barycentrics.
  ExpectImageFileAndImageAreNear("Barycentrics_Cube_Back.png",
                                 EncodeBarycentricsBuffer(1), "cube_back_barys",
                                 "cube back barycentrics do not match");
}

TEST_P(RasterizeTrianglesImplCullTest, CanCullFaces) {
  const FaceCullingMode culling_mode = GetParam();
  CallRasterizeTrianglesImpl(CUBE_VERTICES, CUBE_TRIANGLES, 12, 2,
                             culling_mode);

  const std::vector<int32_t> empty_ids(image_width_ * image_height_);
  const std::vector<int32_t> back_layer_ids(
      triangle_ids_buffer_.begin() + image_width_ * image_height_,
      triangle_ids_buffer_.end());

  if (culling_mode == FaceCullingMode::kBack) {
    // First layer ids should be the front-facing faces, which are actually the
    // back layer of this cube.
    ExpectImageFileAndImageAreNear("Ids_Cube_Back.png", EncodeIdBuffer(),
                                   "cube_front_facing_ids",
                                   "cube front-facing ids do not match");
    // Second layer ids should be empty.
    EXPECT_THAT(back_layer_ids, testing::ContainerEq(empty_ids));
  } else if (culling_mode == FaceCullingMode::kFront) {
    // First layer ids should be the back facing faces, which are the front
    // layer of this cube.
    ExpectImageFileAndImageAreNear("Ids_Cube.png", EncodeIdBuffer(),
                                   "cube_back_facing_ids",
                                   "cube back-facing ids do not match");
    // Second layer ids should be empty.
    EXPECT_THAT(back_layer_ids, testing::ContainerEq(empty_ids));
  }
}

// Only instantiate kBack and kFront to save a bit of time. kNone is tested in
// the previous test.
INSTANTIATE_TEST_SUITE_P(TwoLayerCubeFrontAndBackCulling,
                         RasterizeTrianglesImplCullTest,
                         testing::Values(FaceCullingMode::kBack,
                                         FaceCullingMode::kFront));

}  // namespace

}  // namespace diffren
