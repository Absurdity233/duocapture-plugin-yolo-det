#pragma once

// 原始帧像素 ↔ 模型输入像素 的仿射映射。
//
// 两种预处理模式共用一套公式，只是参数不同：
//
//   letterbox    crop = 整帧，pad = 灰边
//   center_crop  crop = 中心正方形，pad = 0
//
// 统一形式（align_corners=False，像素中心对齐）：
//   model = (src - crop) * scale + pad
//   src   = (model - pad) / scale + crop
//
// 这样插件输出永远是原始帧像素坐标，调用方不用关心用了哪种预处理。

#include <algorithm>
#include <cmath>

namespace duo_yolo {

enum class Preprocess {
  kCenterCrop,  // 裁中心正方形再缩放（默认）：目标更大，但边缘被丢弃
  kLetterbox,   // 整帧等比缩放 + 居中补灰边：看得全，但目标更小
};

struct FrameGeometry {
  int src_w = 0;
  int src_h = 0;
  int dst_w = 0;
  int dst_h = 0;

  // 从原始帧里取哪一块（像素）。letterbox 恒为整帧。
  float crop_x = 0.0f;
  float crop_y = 0.0f;

  // 缩放到模型输入后的缩放比与偏移（模型像素）。center_crop 恒无偏移。
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;

  // 裁剪区尺寸，用来判断一个采样点是否落在有效区域内。
  int crop_w = 0;
  int crop_h = 0;
};

// 模型坐标 → 原始帧坐标。
inline float to_source_x(const FrameGeometry& g, float model_x) {
  return g.scale > 0.0f ? (model_x - g.pad_x) / g.scale + g.crop_x : 0.0f;
}

inline float to_source_y(const FrameGeometry& g, float model_y) {
  return g.scale > 0.0f ? (model_y - g.pad_y) / g.scale + g.crop_y : 0.0f;
}

// 按模式算出几何参数。
inline FrameGeometry make_geometry(Preprocess mode, int src_w, int src_h, int dst_w, int dst_h) {
  FrameGeometry g;
  g.src_w = src_w;
  g.src_h = src_h;
  g.dst_w = dst_w;
  g.dst_h = dst_h;
  g.crop_w = src_w;
  g.crop_h = src_h;

  if (mode == Preprocess::kCenterCrop) {
    // 取与模型输入同长宽比的中心区域
    const double target_aspect = static_cast<double>(dst_w) / static_cast<double>(dst_h);
    const double src_aspect = static_cast<double>(src_w) / static_cast<double>(src_h);
    if (src_aspect > target_aspect) {
      g.crop_w = static_cast<int>(std::lround(src_h * target_aspect));  // 左右裁
    } else if (src_aspect < target_aspect) {
      g.crop_h = static_cast<int>(std::lround(src_w / target_aspect));  // 上下裁
    }
    g.crop_w = std::max(1, std::min(g.crop_w, src_w));
    g.crop_h = std::max(1, std::min(g.crop_h, src_h));
    g.crop_x = static_cast<float>((src_w - g.crop_w) / 2);
    g.crop_y = static_cast<float>((src_h - g.crop_h) / 2);
    g.scale = static_cast<float>(dst_w) / static_cast<float>(g.crop_w);
    g.pad_x = 0.0f;
    g.pad_y = 0.0f;
    return g;
  }

  // letterbox：短边撑满，长边补灰边
  g.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                     static_cast<float>(dst_h) / static_cast<float>(src_h));
  const int resized_w = std::max(1, static_cast<int>(std::lround(src_w * g.scale)));
  const int resized_h = std::max(1, static_cast<int>(std::lround(src_h * g.scale)));
  g.pad_x = static_cast<float>((dst_w - resized_w) / 2);
  g.pad_y = static_cast<float>((dst_h - resized_h) / 2);
  g.crop_x = 0.0f;
  g.crop_y = 0.0f;
  return g;
}

}  // namespace duo_yolo
