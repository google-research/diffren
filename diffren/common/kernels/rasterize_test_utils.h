#ifndef DIFFREN_COMMON_KERNELS_RASTERIZE_TEST_UTILS_H_
#define DIFFREN_COMMON_KERNELS_RASTERIZE_TEST_UTILS_H_

#include <string>
#include <vector>

namespace diffren::test_utils {

// Utilities for generating unit test images.
constexpr int NUM_COLORS = 12;
constexpr uint8_t COLOR_CYCLE[] = {
    255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0,
    128, 0, 0, 0, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 0};

// Vertex values were obtained by dumping the clip-space vertex values from
// the renderSimpleCube test in ../rasterize_triangles_test.py.
// Note that this cube actually has winding order inverted: the CCW faces
// are behind the CW faces.
constexpr float CUBE_VERTICES[] = {
    -2.60648608, -3.22707772,  6.85085106, 6.85714293,
    -1.30324292, -0.992946863, 8.56856918, 8.5714283,
    -1.30324292, 3.97178817,   7.70971,    7.71428585,
    -2.60648608, 1.73765731,   5.991992,   6,
    1.30324292,  -3.97178817,  6.27827835, 6.28571415,
    2.60648608,  -1.73765731,  7.99599648, 8,
    2.60648608,  3.22707772,   7.13713741, 7.14285707,
    1.30324292,  0.992946863,  5.41941929, 5.4285717};

constexpr int32_t CUBE_TRIANGLES[] = {0, 1, 2, 2, 3, 0, 3, 2, 6, 6, 7, 3,
                                      7, 6, 5, 5, 4, 7, 4, 5, 1, 1, 0, 4,
                                      5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7};

constexpr float LARGE_CUBE_VERTICES[] = {
    -1.0,  -1.0, 1.98, 2.0,  1.0,   -1.0, 1.98, 2.0,   1.0,   1.0, 1.98,
    2.0,   -1.0, 1.0,  1.98, 2.0,   -1.0, -1.0, -2.02, -2.0,  1.0, -1.0,
    -2.02, -2.0, 1.0,  1.0,  -2.02, -2.0, -1.0, 1.0,   -2.02, -2.0};

constexpr int32_t LARGE_CUBE_TRIANGLES[] = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
                                            2, 3, 7, 2, 7, 6, 1, 0, 4, 1, 4, 5,
                                            0, 3, 7, 0, 7, 4, 1, 2, 6, 1, 6, 5};

// A simple wrapper around a 4-channel image.
struct ImageBuffer4_b {
  std::vector<unsigned char> data;
  unsigned width, height;

  ImageBuffer4_b(const std::vector<unsigned char>& input_data,
                 unsigned input_width, unsigned input_height)
      : data(input_data), width(input_width), height(input_height) {}
  ImageBuffer4_b(unsigned input_width, unsigned input_height)
      : data(input_width * input_height * 4),
        width(input_width),
        height(input_height) {}

  unsigned char* at(unsigned x, unsigned y) {
    return &data[(x + y * width) * 4];
  }
  const unsigned char* at(unsigned x, unsigned y) const {
    return &data[(x + y * width) * 4];
  }
};

// Generates a UV sphere mesh at the origin. Contains n_rows intermediate rows
// of vertices, n_cols columns of vertices, and 2 extra vertices (top and
// bottom). Y is up. Triangles are CCW.
void UVSphere(int n_rows, int n_cols, float radius, std::vector<float>* vs,
              std::vector<int32_t>* ts);

// Adds a 4-th 1.0 homogeneous coordinate to a flattened array of xyz positions.
std::vector<float> AddHomogeneousCoord(const std::vector<float>& in);

ImageBuffer4_b EncodeBarycentricsBuffer(const std::vector<float>& barycentrics,
                                        int height, int width, int layer);

ImageBuffer4_b EncodeIdBuffer(const std::vector<int32_t>& image, int height,
                              int width, int num_channels = 1, int layer = 0,
                              int channel = 0, bool id_zero_is_white = false);

void ExpectImageFileAndImageAreNear(const std::string& baseline_file_path,
                                    const ImageBuffer4_b& result_image,
                                    const std::string& comparison_name,
                                    const std::string& failure_message);

}  // namespace diffren::test_utils

#endif  // DIFFREN_COMMON_KERNELS_RASTERIZE_TEST_UTILS_H_
