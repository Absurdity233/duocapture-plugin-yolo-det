// 插件的端到端自检工具（不属于插件本身，只用来验证）。
//
// 做的事和主程序完全一样：
//   1. 建 D3D11 设备
//   2. 把一张图片转成 NV12 纹理（模拟解码器输出）
//   3. LoadLibrary 插件 → duo_plugin_entry → create → as_analysis → on_frame
//   4. 把插件日志打到控制台
//
// 用法：yolo_selftest <plugin.dll> <image.jpg|png> "<plugin-config>" [画布宽 画布高]
//
// 给了画布尺寸时会把图片居中贴到品红画布上，用来验证 center_crop 的裁剪边界。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>

#include <d3d11.h>
#include <wrl/client.h>

#include "duocapture/plugin/plugin_api.h"

using Microsoft::WRL::ComPtr;

namespace {

void log_cb(int level, const char* message) {
  static const char* names[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
  const char* name = (level >= 0 && level <= 4) ? names[level] : "?";
  std::printf("[%s] %s\n", name, message != nullptr ? message : "");
}

bool create_device(ComPtr<ID3D11Device>& device, ComPtr<ID3D11DeviceContext>& context) {
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_11_0;
  const HRESULT hr = ::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels,
                                         2, D3D11_SDK_VERSION, device.GetAddressOf(), &got,
                                         context.GetAddressOf());
  return SUCCEEDED(hr);
}

// WIC 解码成 BGRA8
bool load_image(const wchar_t* path, std::vector<unsigned char>& bgra, UINT& width, UINT& height) {
  ComPtr<IWICImagingFactory> factory;
  if (FAILED(::CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(factory.GetAddressOf())))) {
    std::printf("WIC 工厂创建失败\n");
    return false;
  }
  ComPtr<IWICBitmapDecoder> decoder;
  if (FAILED(factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ,
                                                WICDecodeMetadataCacheOnLoad,
                                                decoder.GetAddressOf()))) {
    std::printf("打不开图片\n");
    return false;
  }
  ComPtr<IWICBitmapFrameDecode> frame;
  if (FAILED(decoder->GetFrame(0, frame.GetAddressOf()))) {
    return false;
  }
  ComPtr<IWICFormatConverter> converter;
  if (FAILED(factory->CreateFormatConverter(converter.GetAddressOf()))) {
    return false;
  }
  if (FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
                                   WICBitmapDitherTypeNone, nullptr, 0.0,
                                   WICBitmapPaletteTypeCustom))) {
    return false;
  }
  if (FAILED(converter->GetSize(&width, &height))) {
    return false;
  }
  bgra.resize(static_cast<std::size_t>(width) * height * 4);
  return SUCCEEDED(converter->CopyPixels(nullptr, width * 4,
                                         static_cast<UINT>(bgra.size()), bgra.data()));
}

// BT.709 limited range，与采集端一致
void bgra_to_nv12(const std::vector<unsigned char>& bgra, UINT w, UINT h,
                  std::vector<unsigned char>& nv12) {
  const UINT y_stride = w;
  const UINT uv_stride = w;
  nv12.assign(static_cast<std::size_t>(y_stride) * h +
                  static_cast<std::size_t>(uv_stride) * (h / 2),
              0);
  unsigned char* y_plane = nv12.data();
  unsigned char* uv_plane = nv12.data() + static_cast<std::size_t>(y_stride) * h;

  auto clamp_u8 = [](float v) {
    return static_cast<unsigned char>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
  };

  for (UINT y = 0; y < h; ++y) {
    for (UINT x = 0; x < w; ++x) {
      const unsigned char* p = &bgra[(static_cast<std::size_t>(y) * w + x) * 4];
      const float b = p[0];
      const float g = p[1];
      const float r = p[2];
      const float yy = 0.2126f * r + 0.7152f * g + 0.0722f * b;
      y_plane[static_cast<std::size_t>(y) * y_stride + x] = clamp_u8(16.0f + yy * 0.858824f);
    }
  }
  for (UINT y = 0; y < h / 2; ++y) {
    for (UINT x = 0; x < w / 2; ++x) {
      float r = 0.0f;
      float g = 0.0f;
      float b = 0.0f;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const unsigned char* p = &bgra[(static_cast<std::size_t>(y * 2 + dy) * w + (x * 2 + dx)) * 4];
          b += p[0];
          g += p[1];
          r += p[2];
        }
      }
      b *= 0.25f;
      g *= 0.25f;
      r *= 0.25f;
      const float u = -0.1146f * r - 0.3854f * g + 0.5f * b + 128.0f;
      const float v = 0.5f * r - 0.4542f * g - 0.0458f * b + 128.0f;
      uv_plane[static_cast<std::size_t>(y) * uv_stride + x * 2 + 0] = clamp_u8(u);
      uv_plane[static_cast<std::size_t>(y) * uv_stride + x * 2 + 1] = clamp_u8(v);
    }
  }
}

bool create_nv12_texture(ID3D11Device* device, ID3D11DeviceContext* context,
                         const std::vector<unsigned char>& nv12, UINT w, UINT h,
                         ComPtr<ID3D11Texture2D>& out) {
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = w;
  desc.Height = h;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_NV12;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  if (FAILED(device->CreateTexture2D(&desc, nullptr, out.GetAddressOf()))) {
    return false;
  }

  // NV12 在 D3D11 里是**单个**子资源：Y 占据 [0, h)，UV 占据 [h, h*1.5)，
  // 都在同一张图上，用 box.top/bottom 指定。这与 nvdec_cuvid.cpp 的
  // CPU 兜底路径写法一致。
  D3D11_BOX y_box{};
  y_box.left = 0;
  y_box.top = 0;
  y_box.front = 0;
  y_box.right = w;
  y_box.bottom = h;
  y_box.back = 1;
  context->UpdateSubresource(out.Get(), 0, &y_box, nv12.data(), w, 0);

  D3D11_BOX uv_box{};
  uv_box.left = 0;
  uv_box.top = h;
  uv_box.front = 0;
  uv_box.right = w;
  uv_box.bottom = h + h / 2;
  uv_box.back = 1;
  context->UpdateSubresource(out.Get(), 0, &uv_box,
                             nv12.data() + static_cast<std::size_t>(w) * h, w, 0);
  return true;
}

std::string narrow(const char* utf8) { return utf8 != nullptr ? std::string(utf8) : std::string(); }

}  // namespace

int wmain(int argc, wchar_t** argv) {
  // 崩溃时最想知道「走到哪一步了」，所以把 stdout 设成无缓冲。
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::printf("用法: yolo_selftest <plugin.dll> <image> \"<plugin-config>\"\n");
    return 1;
  }
  ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  if (!create_device(device, context)) {
    std::printf("D3D11 设备创建失败\n");
    return 1;
  }

  std::vector<unsigned char> bgra;
  UINT width = 0;
  UINT height = 0;
  if (!load_image(argv[2], bgra, width, height)) {
    return 1;
  }
  std::printf("图片 %ux%u\n", width, height);

  // 可选：把宽高各扩展成指定的画布尺寸（内容居中，余下填品红），
  // 用来验证 center_crop 真的裁掉了边缘。用法：
  //   yolo_selftest dll image.jpg "cfg" 1280 1280
  if (argc >= 6) {
    const int canvas_w = _wtoi(argv[4]);
    const int canvas_h = _wtoi(argv[5]);
    if (canvas_w > 0 && canvas_h > 0) {
      std::vector<unsigned char> canvas(static_cast<std::size_t>(canvas_w) * canvas_h * 4, 0);
      const int off_x = (canvas_w - static_cast<int>(width)) / 2;
      const int off_y = (canvas_h - static_cast<int>(height)) / 2;
      // 品红边框：任何落在裁剪区外的像素都会在 dump 里显示成这个颜色
      for (std::size_t i = 0; i < canvas.size(); i += 4) {
        canvas[i + 0] = 255;
        canvas[i + 1] = 0;
        canvas[i + 2] = 255;
        canvas[i + 3] = 255;
      }
      for (UINT y = 0; y < height; ++y) {
        const int dy = off_y + static_cast<int>(y);
        if (dy < 0 || dy >= canvas_h) {
          continue;
        }
        for (UINT x = 0; x < width; ++x) {
          const int dx = off_x + static_cast<int>(x);
          if (dx < 0 || dx >= canvas_w) {
            continue;
          }
          std::memcpy(&canvas[(static_cast<std::size_t>(dy) * canvas_w + dx) * 4],
                      &bgra[(static_cast<std::size_t>(y) * width + x) * 4], 4);
        }
      }
      bgra.swap(canvas);
      width = static_cast<UINT>(canvas_w);
      height = static_cast<UINT>(canvas_h);
      std::printf("已扩展画布 %ux%u（品红填充边）\n", width, height);
    }
  }

  std::vector<unsigned char> nv12;
  bgra_to_nv12(bgra, width, height, nv12);

  // 调试：把喂进去的 NV12 存下来，方便和插件的回读结果对照。
  wchar_t dump_path[1024] = {};
  size_t dump_len = 0;
  if (::_wgetenv_s(&dump_len, dump_path, 1024, L"DUO_DUMP_NV12") == 0 && dump_len > 0) {
    FILE* fp = nullptr;
    if (::_wfopen_s(&fp, dump_path, L"wb") == 0 && fp != nullptr) {
      std::fwrite(nv12.data(), 1, nv12.size(), fp);
      std::fclose(fp);
      std::printf("NV12 已存到 %ls（%zu 字节）\n", dump_path, nv12.size());
    }
  }

  ComPtr<ID3D11Texture2D> texture;
  if (!create_nv12_texture(device.Get(), context.Get(), nv12, width, height, texture)) {
    std::printf("NV12 纹理创建失败\n");
    return 1;
  }

  HMODULE lib = ::LoadLibraryW(argv[1]);
  if (lib == nullptr) {
    std::printf("加载插件失败，错误码 %lu\n", ::GetLastError());
    return 1;
  }
  auto entry =
      reinterpret_cast<DuoPluginEntryFn>(::GetProcAddress(lib, DUO_PLUGIN_ENTRY_NAME));
  if (entry == nullptr) {
    std::printf("插件缺少 %s\n", DUO_PLUGIN_ENTRY_NAME);
    return 1;
  }
  const DuoPlugin* plugin = entry(DUO_PLUGIN_ABI_VERSION);
  if (plugin == nullptr) {
    std::printf("插件拒绝了 ABI 版本 %u\n", DUO_PLUGIN_ABI_VERSION);
    return 1;
  }
  std::printf("插件 %s v%s [%s]\n", plugin->name, plugin->version, plugin->description);

  DuoPluginHostServices services{};
  services.abi_version = DUO_PLUGIN_ABI_VERSION;
  services.struct_size = sizeof(services);
  services.log = &log_cb;
  services.d3d_device = device.Get();
  services.d3d_context = context.Get();

  void* instance = nullptr;
  const char* config = argc >= 4 ? nullptr : "";
  std::string config_storage;
  if (argc >= 4) {
    // argv[3] 是宽字符，转成 UTF-8
    const int need = ::WideCharToMultiByte(CP_UTF8, 0, argv[3], -1, nullptr, 0, nullptr, nullptr);
    config_storage.resize(static_cast<std::size_t>(need));
    ::WideCharToMultiByte(CP_UTF8, 0, argv[3], -1, config_storage.data(), need, nullptr, nullptr);
    config = config_storage.c_str();
  }

  const int rc = plugin->create(&services, config, &instance);
  if (rc != 0 || instance == nullptr) {
    std::printf("create 失败 rc=%d\n", rc);
    return 1;
  }
  const DuoAnalysisPlugin* api = plugin->as_analysis(instance);
  if (api == nullptr || api->on_frame == nullptr) {
    std::printf("没有分析接口\n");
    return 1;
  }

  DuoFrameView view{};
  view.texture = texture.Get();
  view.width = width;
  view.height = height;
  view.capture_ts_ns = 0;
  view.key = 1;

  // 跑几帧，让 every 之类也能验证
  int last = -1;
  for (int i = 0; i < 3; ++i) {
    last = api->on_frame(instance, &view);
    std::printf("on_frame #%d -> %d\n", i + 1, last);
  }

  // 只走描述符上的 destroy：DuoAnalysisPlugin::destroy 与
  // DuoPlugin::destroy 是同一个函数，两边都调会双重释放。
  plugin->destroy(instance);
  ::FreeLibrary(lib);
  ::CoUninitialize();
  std::printf("完成，最后一帧检出 %d 个目标\n", last);
  return 0;
}
