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

#include "diffren/common/kernels/rasterize_test_utils.h"

#include <math.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "third_party/lodepng/lodepng.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace diffren::test_utils {

// Generates a UV sphere mesh at the origin. Contains n_rows intermediate rows
// of vertices, n_cols columns of vertices, and 2 extra vertices (top and
// bottom). Y is up. Triangles are CCW.
void UVSphere(int n_rows, int n_cols, float radius, std::vector<float>* vs,
              std::vector<int32_t>* ts) {
  CHECK_EQ(vs->size(), 0);
  CHECK_EQ(ts->size(), 0);
  // 1 vertex at top and bottom, plus n_rows rows with n_cols each.
  vs->reserve(3 * (2 + n_rows * n_cols));
  // 1 triangle for each vertex at the top and bottom, and 1 quad for each
  // vertex in each row of triangles in between intermediate rows:
  ts->reserve(3 * (n_cols + n_cols + 2 * (n_rows - 1) * n_cols));

  // Add vertex on top of the sphere
  vs->push_back(0.0f);
  vs->push_back(radius);
  vs->push_back(0.0f);

  // Add all rows of vertices
  int vi = 1;
  for (int ri = 0; ri < n_rows; ++ri) {
    // Add vertices in each row
    for (int ci = 0; ci < n_cols; ++ci) {
      // phi goes from [0, pi] including top/bottom
      // So smallest is 0, last (n+2th)=n+1 is 1.
      // Row vertices go [1,n], external points are 0, n+1
      // At zero should be 0, at n+1 should be 1.
      float phi = (M_PI * (ri + 1)) / (n_rows + 1);
      // theta goes from [0, 2*pi * (n-1)/n] because endpoint wraps.
      float theta = 2.0f * M_PI * ci / n_cols;
      float x = radius * sinf(phi) * sinf(theta);
      float y = radius * cosf(phi);
      float z = radius * sinf(phi) * cosf(theta);
      vs->push_back(x);
      vs->push_back(y);
      vs->push_back(z);
      vi++;
    }
  }
  // Add bottom point of the sphere:
  vs->push_back(0.0f);
  vs->push_back(-1.0f * radius);
  vs->push_back(0.0f);
  int n_verts = ++vi;  // Need vertex count for last row
  // Add top triangles:
  int top_row_start = 1;  // vertex 0 is the top of the sphere.
  for (int ci = 0; ci < n_cols; ++ci) {
    int top_vert = 0;
    int left_vert = top_row_start + ci;
    // Wrap around for last column:
    int right_vert = top_row_start + ((ci + 1) % n_cols);
    ts->push_back(top_vert);
    ts->push_back(left_vert);
    ts->push_back(right_vert);
  }
  // Add triangles between intermediate rows:
  int first_row_start = 1;  // Move past first vertex
  // For each intermediate row of vertices:
  for (int ri = 0; ri < n_rows - 1; ++ri) {
    // For each column of vertices:
    for (int ci = 0; ci < n_cols; ++ci) {
      // Quad goes between ri and ri+1, and ci and ci+1; wrap around when ci is
      // the last column, so ci+1 should wrap to connect to the first (0-th idx)
      int top_left_idx = first_row_start + ri * n_cols + ci;
      int top_right_idx = first_row_start + ri * n_cols + ((ci + 1) % n_cols);
      int bottom_left_idx = first_row_start + (ri + 1) * n_cols + ci;
      int bottom_right_idx =
          first_row_start + (ri + 1) * n_cols + ((ci + 1) % n_cols);
      // Split quad into two CCW triangles:
      ts->push_back(top_left_idx);
      ts->push_back(bottom_left_idx);
      ts->push_back(bottom_right_idx);
      ts->push_back(top_right_idx);
      ts->push_back(top_left_idx);
      ts->push_back(bottom_right_idx);
    }
  }
  // Add bottom triangles:
  int last_row_start = n_verts - 1 - n_cols;
  for (int ci = 0; ci < n_cols; ++ci) {
    int bottom_vert = n_verts - 1;
    int left_vert = last_row_start + ci;
    int right_vert = last_row_start + ((ci + 1) % n_cols);
    ts->push_back(left_vert);
    ts->push_back(bottom_vert);
    ts->push_back(right_vert);
  }
}

std::vector<float> AddHomogeneousCoord(const std::vector<float>& in) {
  CHECK_EQ(in.size() % 3, 0);
  std::vector<float> out;
  int vertex_count = in.size() / 3;
  out.reserve(vertex_count * 4);
  for (int vi = 0; vi < vertex_count; ++vi) {
    for (int i = 0; i < 3; ++i) {
      out.push_back(in[3 * vi + i]);
    }
    out.push_back(1.0f);
  }
  return out;
}

std::string GetRunfilesRelativePath(const std::string& filename) {
  const std::string srcdir = std::getenv("TEST_SRCDIR");
  const std::string test_data = "/__main__/diffren/common/test_data/";
  return srcdir + test_data + filename;
}

ImageBuffer4_b LoadPng(const std::string& filename) {
  std::vector<unsigned char> decoded;
  unsigned width, height;
  unsigned error = lodepng::decode(decoded, width, height, filename);
  CHECK(error == 0) << "Decoder error: " << lodepng_error_text(error)
                    << " While trying to load: " << filename;
  return ImageBuffer4_b(decoded, width, height);
}

void SavePng(const std::string& filename, const ImageBuffer4_b& image) {
  unsigned error =
      lodepng::encode(filename.c_str(), image.data, image.width, image.height);
  CHECK(error == 0) << "Encoder error: " << lodepng_error_text(error)
                    << " While trying to save: " << filename << " with size "
                    << image.width << "," << image.height;
}

ImageBuffer4_b EncodeBarycentricsBuffer(const std::vector<float>& barycentrics,
                                        int height, int width, int layer) {
  ImageBuffer4_b result(width, height);
  const int o = layer * height * width * 3;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        result.at(x, y)[c] = barycentrics[(y * width + x) * 3 + c + o] * 255.0;
      }
      result.at(x, y)[3] = 255;  // Set alpha to 1.
    }
  }
  return result;
}

ImageBuffer4_b EncodeIdBuffer(const std::vector<int32_t>& image, int height,
                              int width, int num_channels, int layer,
                              int channel, bool id_zero_is_white) {
  ImageBuffer4_b result(width, height);
  const int o = layer * height * width * num_channels;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int32_t id =
          image[o + y * width * num_channels + x * num_channels + channel];
      for (int c = 0; c < 3; ++c) {
        if (id_zero_is_white && id == 0) {
          result.at(x, y)[c] = 255;
          continue;
        }
        result.at(x, y)[c] = COLOR_CYCLE[(id % NUM_COLORS) * 3 + c];
      }
      result.at(x, y)[3] = 255;  // Set alpha to 1.
    }
  }
  return result;
}

bool ImagesAreNear(const ImageBuffer4_b& a, const ImageBuffer4_b& b) {
  if (a.width != b.width) {
    LOG(ERROR) << "Not equal width: " << a.width << " != " << b.width;
    return false;
  }
  if (a.height != b.height) {
    LOG(ERROR) << "Not equal height: " << a.height << " != " << b.height;
    return false;
  }
  // Differences in JPEG encoding can produce pixels with pretty large
  // variation, so we need to set a 10 / 255 ~= 4% error threshold here.
  const int per_channel_tolerance = 10;
  // Allow 0.5% of the pixels to differ by more than the per-channel tolerance.
  const int max_outlier_pixels = 0.005 * a.width * a.height;

  int outlier_count = 0;
  for (int j = 0; j < a.height; ++j) {
    for (int i = 0; i < a.width; ++i) {
      for (int k = 0; k < 4; ++k) {
        if (abs(static_cast<int>(a.at(i, j)[k]) -
                static_cast<int>(b.at(i, j)[k])) > per_channel_tolerance) {
          ++outlier_count;
        }
      }
    }
  }

  if (outlier_count > max_outlier_pixels) {
    LOG(ERROR) << "Found " << outlier_count << " pixels outside tolerance "
               << per_channel_tolerance << ". Maximum allowed is "
               << max_outlier_pixels;
    return false;
  }

  return true;
}

void ExpectImageFileAndImageAreNear(const std::string& baseline_file,
                                    const ImageBuffer4_b& result_image,
                                    const std::string& comparison_name,
                                    const std::string& failure_message) {
  ImageBuffer4_b baseline_image =
      LoadPng(GetRunfilesRelativePath(baseline_file));

  const bool images_match = ImagesAreNear(baseline_image, result_image);

  if (!images_match) {
    const char* outputs_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    CHECK_NE(outputs_dir, nullptr);

    // Avoid using file::JoinPath here because it is not available open source.
    const std::string baseline_output_path =
        std::string(outputs_dir) + "/" + comparison_name + "_baseline.png";
    SavePng(baseline_output_path, baseline_image);

    const std::string result_output_path =
        std::string(outputs_dir) + "/" + comparison_name + "_result.png";
    SavePng(result_output_path, result_image);
  }

  EXPECT_TRUE(images_match) << failure_message;
}

}  // namespace diffren::test_utils
