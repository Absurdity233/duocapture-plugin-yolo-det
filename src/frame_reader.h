#pragma once

// 把解码器给出的 NV12 纹理读成模型输入（NCHW float，RGB，/255）。
//
// 路径只有一条：D3D11 暂存纹理（STAGING）→ Map → CPU 采样。
// 之所以不做 GPU 转换，是因为分析 tap 本来就是「忙则丢帧」，
// CPU 这一趟的代价是少分析几帧，不会拖慢呈现。

#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d11.h>
#include <wrl/client.h>

#include "geometry.h"

namespace duo_yolo {

class Nv12FrameReader {
 public:
  // 读取 src，按 mode 预处理后写满 out_nchw（3 * dst_w * dst_h 个 float）。
  // geom 回传本次使用的几何参数，用来把检测框映射回原始帧。
  // 失败时返回 false 并填 error。
  bool read(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* src,
            Preprocess mode, int dst_w, int dst_h, std::vector<float>& out_nchw,
            FrameGeometry& geom, std::string& error);

  // 分辨率变化 / 流重置时调用，丢掉暂存纹理让它按新尺寸重建。
  void reset();

 private:
  Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_;
  unsigned int width_ = 0;
  unsigned int height_ = 0;
};

}  // namespace duo_yolo
