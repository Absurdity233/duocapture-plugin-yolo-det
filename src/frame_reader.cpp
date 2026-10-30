#include "frame_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace duo_yolo {
namespace {

// BT.709 limited range（Y 16-235，Cb/Cr 16-240）。
// 与采集端的编码设置一致，见 src/encode/nvenc_encoder.cpp。
inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

struct YuvToRgb {
  float y_scale = 1.164383f;
  float cr_r = 1.792741f;
  float cb_g = 0.213249f;
  float cr_g = 0.532909f;
  float cb_b = 2.112402f;

  void convert(float y, float u, float v, float& r, float& g, float& b) const {
    const float yy = y_scale * (y - 16.0f);
    const float cb = u - 128.0f;
    const float cr = v - 128.0f;
    r = clamp01((yy + cr_r * cr) * (1.0f / 255.0f));
    g = clamp01((yy - cb_g * cb - cr_g * cr) * (1.0f / 255.0f));
    b = clamp01((yy + cb_b * cb) * (1.0f / 255.0f));
  }
};

}  // namespace

void Nv12FrameReader::reset() {
  staging_.Reset();
  width_ = 0;
  height_ = 0;
}

bool Nv12FrameReader::read(ID3D11Device* device, ID3D11DeviceContext* context,
                           ID3D11Texture2D* src, int dst_w, int dst_h,
                           std::vector<float>& out_nchw, FrameGeometry& geom,
                           std::string& error) {
  if (device == nullptr || context == nullptr || src == nullptr) {
    error = "D3D11 设备或纹理为空";
    return false;
  }
  if (dst_w <= 0 || dst_h <= 0) {
    error = "模型输入尺寸非法";
    return false;
  }

  D3D11_TEXTURE2D_DESC src_desc{};
  src->GetDesc(&src_desc);
  if (src_desc.Format != DXGI_FORMAT_NV12) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "输入纹理不是 NV12（格式 %d）",
                  static_cast<int>(src_desc.Format));
    error = buf;
    return false;
  }
  if (src_desc.Width == 0 || src_desc.Height == 0) {
    error = "输入纹理尺寸为 0";
    return false;
  }

  if (staging_ == nullptr || width_ != src_desc.Width || height_ != src_desc.Height) {
    staging_.Reset();
    D3D11_TEXTURE2D_DESC desc = src_desc;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging_))) {
      error = "创建 NV12 暂存纹理失败";
      return false;
    }
    width_ = src_desc.Width;
    height_ = src_desc.Height;
  }

  context->CopyResource(staging_.Get(), src);

  // NV12 在 D3D11 里是**一个**子资源：Y 占据前 height 行，UV 占据接下来的
  // height/2 行，两段都在同一个子资源里。所以 Map(0) 一次就能拿到全部数据，
  // UV 的起点是 RowPitch * height。
  D3D11_MAPPED_SUBRESOURCE y_map{};
  if (FAILED(context->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &y_map))) {
    error = "Map NV12 暂存纹理失败";
    return false;
  }

  const auto* y_plane = static_cast<const unsigned char*>(y_map.pData);
  const std::size_t pitch = y_map.RowPitch;
  const int src_w = static_cast<int>(src_desc.Width);
  const int src_h = static_cast<int>(src_desc.Height);
  const auto* uv_plane = y_plane + pitch * static_cast<std::size_t>(src_h);

  geom = make_geometry(src_w, src_h, dst_w, dst_h);

  out_nchw.resize(static_cast<std::size_t>(3) * dst_w * dst_h);
  float* out_r = out_nchw.data();
  float* out_g = out_r + static_cast<std::size_t>(dst_w) * dst_h;
  float* out_b = out_g + static_cast<std::size_t>(dst_w) * dst_h;

  const YuvToRgb conv;
  const float inv_scale = 1.0f / geom.scale;

  const int chroma_w = (src_w + 1) / 2;
  const int chroma_h = (src_h + 1) / 2;

  // 采样坐标只跟输出列有关，每一列算一次。公式与 to_source_x 一致：
  //   src_x = model_x / scale + crop_x
  std::vector<int> col_x0(static_cast<std::size_t>(dst_w));
  std::vector<float> col_wx(static_cast<std::size_t>(dst_w));
  for (int i = 0; i < dst_w; ++i) {
    float sx = (static_cast<float>(i) + 0.5f) * inv_scale + geom.crop_x - 0.5f;
    sx = std::clamp(sx, 0.0f, static_cast<float>(src_w - 1));
    const int x0 = static_cast<int>(sx);
    col_x0[static_cast<std::size_t>(i)] = std::min(x0, src_w - 1);
    col_wx[static_cast<std::size_t>(i)] = sx - static_cast<float>(x0);
  }

  for (int y = 0; y < dst_h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * dst_w;

    float sy = (static_cast<float>(y) + 0.5f) * inv_scale + geom.crop_y - 0.5f;
    sy = std::clamp(sy, 0.0f, static_cast<float>(src_h - 1));
    const int y0 = std::min(static_cast<int>(sy), src_h - 1);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = sy - static_cast<float>(y0);

    const unsigned char* row_y0 = y_plane + static_cast<std::size_t>(y0) * pitch;
    const unsigned char* row_y1 = y_plane + static_cast<std::size_t>(y1) * pitch;

    // 色度是 2x2 亚采样：两个源行共用一个色度行，双线性只在色度行内做。
    const int cy = std::clamp(y0 / 2, 0, chroma_h - 1);
    const unsigned char* row_uv = uv_plane + static_cast<std::size_t>(cy) * pitch;

    for (int x = 0; x < dst_w; ++x) {
      const int x0 = col_x0[static_cast<std::size_t>(x)];
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = col_wx[static_cast<std::size_t>(x)];

      // 亮度：4 点双线性
      const float top = row_y0[x0] * (1.0f - wx) + row_y0[x1] * wx;
      const float bottom = row_y1[x0] * (1.0f - wx) + row_y1[x1] * wx;
      const float yy = top * (1.0f - wy) + bottom * wy;

      // 色度：水平双线性，行已按 2 取整
      const int cx0 = std::clamp(x0 / 2, 0, chroma_w - 1);
      const int cx1 = std::clamp(cx0 + 1, 0, chroma_w - 1);
      const float u = row_uv[cx0 * 2] * (1.0f - wx) + row_uv[cx1 * 2] * wx;
      const float v = row_uv[cx0 * 2 + 1] * (1.0f - wx) + row_uv[cx1 * 2 + 1] * wx;

      float r = 0.0f;
      float g = 0.0f;
      float b = 0.0f;
      conv.convert(yy, u, v, r, g, b);

      const std::size_t idx = row + static_cast<std::size_t>(x);
      out_r[idx] = r;
      out_g[idx] = g;
      out_b[idx] = b;
    }
  }

  context->Unmap(staging_.Get(), 0);
  return true;
}

}  // namespace duo_yolo
