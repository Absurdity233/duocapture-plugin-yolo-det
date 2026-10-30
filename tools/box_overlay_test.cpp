// BoxOverlay 的端到端验证：真的把框画到了目标纹理上吗？
//
// 做法：建一张全黑 BGRA 纹理 → 用已知坐标的框调用 draw() →
// 回读像素 → 检查框边缘确实有颜色、框内没被涂满。

#include <cstdio>
#include <cstring>
#include <tuple>
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

#include "duocapture/common/status.h"
#include "duocapture/present/box_overlay.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT kW = 640;
constexpr UINT kH = 360;

bool create_device(ComPtr<ID3D11Device>& device, ComPtr<ID3D11DeviceContext>& context) {
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_11_0;
  return SUCCEEDED(::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels,
                                       2, D3D11_SDK_VERSION, device.GetAddressOf(), &got,
                                       context.GetAddressOf()));
}

}  // namespace

int wmain() {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  if (!create_device(device, context)) {
    std::printf("FAIL: 设备创建失败\n");
    return 1;
  }

  // 目标表面：全黑 BGRA
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = kW;
  desc.Height = kH;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> target;
  if (FAILED(device->CreateTexture2D(&desc, nullptr, &target))) {
    std::printf("FAIL: 目标纹理创建失败\n");
    return 1;
  }
  const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  {
    ComPtr<ID3D11RenderTargetView> rtv;
    if (FAILED(device->CreateRenderTargetView(target.Get(), nullptr, &rtv))) {
      std::printf("FAIL: RTV 创建失败\n");
      return 1;
    }
    context->ClearRenderTargetView(rtv.Get(), clear);
  }

  duo::BoxOverlay overlay;
  if (!duo::ok(overlay.initialize(device.Get(), context.Get()))) {
    std::printf("FAIL: BoxOverlay 初始化失败\n");
    return 1;
  }

  // 源帧 1280x720 → 目标 640x360，缩放正好 0.5。
  // 框放在源坐标 (400,200)-(600,400)，映射到目标应是 (200,100)-(300,200)。
  DuoOverlayBox box{};
  box.x1 = 400.0f;
  box.y1 = 200.0f;
  box.x2 = 600.0f;
  box.y2 = 400.0f;
  box.score = 0.9f;
  box.class_id = 1;
  box.color = 0xFFFF0000u;  // 纯红，便于识别

  const duo::Status st =
      overlay.draw(target.Get(), kW, kH, &box, 1, 1280, 720);
  if (!duo::ok(st)) {
    std::printf("FAIL: draw 返回 %s\n", duo::to_string(st));
    return 1;
  }

  // 回读
  D3D11_TEXTURE2D_DESC sd = desc;
  sd.Usage = D3D11_USAGE_STAGING;
  sd.BindFlags = 0;
  sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ComPtr<ID3D11Texture2D> staging;
  if (FAILED(device->CreateTexture2D(&sd, nullptr, &staging))) {
    std::printf("FAIL: 暂存纹理创建失败\n");
    return 1;
  }
  context->CopyResource(staging.Get(), target.Get());
  D3D11_MAPPED_SUBRESOURCE mapped{};
  if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
    std::printf("FAIL: Map 失败\n");
    return 1;
  }
  const auto* px = static_cast<const unsigned char*>(mapped.pData);
  auto at = [&](int x, int y) {
    const unsigned char* p = px + static_cast<std::size_t>(y) * mapped.RowPitch + x * 4;
    return std::make_tuple(p[0], p[1], p[2]);  // B, G, R
  };

  int failures = 0;
  auto check = [&](bool cond, const char* what) {
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
      ++failures;
    }
  };

  // 期望的边框位置（目标坐标）：左 200、右 300、上 100、下 200
  const auto left_edge = at(201, 150);
  const auto top_edge = at(250, 101);
  const auto inside = at(250, 150);
  const auto outside = at(50, 50);

  check(std::get<2>(left_edge) > 150, "左边框有红色像素");
  check(std::get<2>(top_edge) > 150, "上边框有红色像素");
  check(std::get<2>(inside) < 60, "框内部没有被涂满");
  check(std::get<2>(outside) < 60, "框外部保持透明");

  context->Unmap(staging.Get(), 0);

  std::printf("\n采样值 (B,G,R): 左边框=(%d,%d,%d) 上边框=(%d,%d,%d) 内部=(%d,%d,%d) 外部=(%d,%d,%d)\n",
              std::get<0>(left_edge), std::get<1>(left_edge), std::get<2>(left_edge),
              std::get<0>(top_edge), std::get<1>(top_edge), std::get<2>(top_edge),
              std::get<0>(inside), std::get<1>(inside), std::get<2>(inside),
              std::get<0>(outside), std::get<1>(outside), std::get<2>(outside));

  std::printf("\n%s\n", failures == 0 ? "全部通过" : "有失败项");
  return failures == 0 ? 0 : 1;
}
