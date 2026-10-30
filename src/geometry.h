#pragma once

// 原始帧像素 ↔ 模型输入像素 的映射。
//
// 预处理只有一种：center_crop —— 取与模型输入同长宽比的中心区域，
// 等比缩放到模型尺寸。不补边，模型输入的每一格都来自真实像素。
//
//   src = model / scale + crop
//
// 插件输出因此永远是原始帧像素坐标。

#include <algorithm>
#include <cmath>

namespace duo_yolo {

struct FrameGeometry {
  int src_w = 0;
  int src_h = 0;
  int dst_w = 0;
  int dst_h = 0;

  // 从原始帧里取哪一块（像素），以及它的尺寸。
  float crop_x = 0.0f;
  float crop_y = 0.0f;
  int crop_w = 0;
  int crop_h = 0;

  // 裁剪区 → 模型输入的缩放比。
  float scale = 1.0f;
};

// 模型坐标 → 原始帧坐标。
inline float to_source_x(const FrameGeometry& g, float model_x) {
  return g.scale > 0.0f ? model_x / g.scale + g.crop_x : 0.0f;
}

inline float to_source_y(const FrameGeometry& g, float model_y) {
  return g.scale > 0.0f ? model_y / g.scale + g.crop_y : 0.0f;
}

// 取中心正方形再缩放到模型输入尺寸。
inline FrameGeometry make_geometry(int src_w, int src_h, int dst_w, int dst_h) {
  FrameGeometry g;
  g.src_w = src_w;
  g.src_h = src_h;
  g.dst_w = dst_w;
  g.dst_h = dst_h;

  // 取与模型输入同长宽比的中心区域
  const double target_aspect = static_cast<double>(dst_w) / static_cast<double>(dst_h);
  const double src_aspect = static_cast<double>(src_w) / static_cast<double>(src_h);
  g.crop_w = src_w;
  g.crop_h = src_h;
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
  return g;
}

}  // namespace duo_yolo
