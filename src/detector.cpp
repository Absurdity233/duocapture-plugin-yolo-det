#include "detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// TensorRT 头文件按 TRT 10 的接口写。如果本机没有 TensorRT，这个文件
// 仍然要能编译（CI 上就没有），整套 API 通过下面的宏做条件编译。
#if __has_include(<NvInfer.h>)
#define DUO_YOLO_HAVE_TENSORRT 1
#endif

#if defined(DUO_YOLO_HAVE_TENSORRT)
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace duo_yolo {
namespace {

// 输出布局：
//   rows —— YOLOv8/v10/v26 导出时就做完 NMS 的 [x1,y1,x2,y2,score,cls] × N
//   chw  —— 原始 YOLO 的 [1, 4+nc, N]，需要自己挑类 + NMS
enum class OutputLayout { kRows, kChw };

float iou(const Detection& a, const Detection& b) {
  const float ix1 = std::max(a.x1, b.x1);
  const float iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2);
  const float iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::size_t dims_volume(const int64_t* d, int n) {
  std::size_t v = 1;
  for (int i = 0; i < n; ++i) {
    if (d[i] <= 0) {
      return 0;
    }
    v *= static_cast<std::size_t>(d[i]);
  }
  return v;
}

}  // namespace

#if defined(DUO_YOLO_HAVE_TENSORRT)

namespace {

// TensorRT 日志接口。只把警告和错误转给插件日志，信息级（每次反序列化
// 都会打印一堆）直接吞掉。
class TrtLogger final : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(LogFn sink) : sink_(sink) {}

  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if (sink_ == nullptr || msg == nullptr) {
      return;
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        sink_(4, msg);
        break;
      case Severity::kWARNING:
        sink_(3, msg);
        break;
      default:
        break;  // kINFO / kVERBOSE：太吵，丢掉
    }
  }

 private:
  LogFn sink_ = nullptr;
};

using CreateInferRuntimeFn = void* (*)(void* logger, int32_t version);

// TRT 接口对象的析构是 protected 的，只能 delete 指针本身。
template <typename T>
void trt_release(T*& ptr) {
  if (ptr != nullptr) {
    delete ptr;
    ptr = nullptr;
  }
}

}  // namespace

struct YoloDetector::Impl {
  HMODULE nvinfer_dll = nullptr;
  TrtLogger* logger = nullptr;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;

  void* device_input = nullptr;
  void* device_output = nullptr;
  cudaStream_t stream = nullptr;
  std::size_t input_elements = 0;
  std::size_t output_elements = 0;

  std::vector<float> host_output;

  std::string input_name;
  std::string output_name;
  std::string io_summary;
  OutputLayout layout = OutputLayout::kRows;
  int chw_channels = 0;
  int chw_count = 0;
  int input_w = 0;
  int input_h = 0;

  ~Impl() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (device_input != nullptr) {
      cudaFree(device_input);
    }
    if (device_output != nullptr) {
      cudaFree(device_output);
    }
    trt_release(context);
    trt_release(engine);
    trt_release(runtime);
    delete logger;
    logger = nullptr;
    if (nvinfer_dll != nullptr) {
      ::FreeLibrary(nvinfer_dll);
    }
  }
};

#else

struct YoloDetector::Impl {};

#endif  // DUO_YOLO_HAVE_TENSORRT

YoloDetector::YoloDetector() = default;
YoloDetector::~YoloDetector() { delete impl_; }

int YoloDetector::input_width() const noexcept {
#if defined(DUO_YOLO_HAVE_TENSORRT)
  return impl_ != nullptr ? impl_->input_w : 0;
#else
  return 0;
#endif
}

int YoloDetector::input_height() const noexcept {
#if defined(DUO_YOLO_HAVE_TENSORRT)
  return impl_ != nullptr ? impl_->input_h : 0;
#else
  return 0;
#endif
}

const std::string& YoloDetector::io_summary() const noexcept {
  static const std::string empty;
#if defined(DUO_YOLO_HAVE_TENSORRT)
  return impl_ != nullptr ? impl_->io_summary : empty;
#else
  return empty;
#endif
}

#if defined(DUO_YOLO_HAVE_TENSORRT)

bool YoloDetector::init(const DetectConfig& config, LogFn log, std::string& error) {
  config_ = config;
  impl_ = new Impl();

  if (config.tensorrt_dll.empty()) {
    error = "缺少 tensorrt_dll 配置（nvinfer_10.dll 的完整路径）";
    return false;
  }
  if (config.engine_path.empty()) {
    error = "缺少 engine_path 配置";
    return false;
  }

  // 用 LOAD_WITH_ALTERED_SEARCH_PATH 让 nvinfer 能找到同目录下的
  // builder resource / plugin DLL。
  impl_->nvinfer_dll =
      ::LoadLibraryExA(config.tensorrt_dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (impl_->nvinfer_dll == nullptr) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "加载 %s 失败（错误码 %lu）", config.tensorrt_dll.c_str(),
                  ::GetLastError());
    error = buf;
    return false;
  }

  // 所有 cudaXxx 调用都会把主上下文绑到当前线程。先选定设备，
  // 否则多卡机器上可能选到错的那块。
  if (config.device_index < 0) {
    error = "device_index 不能为负";
    return false;
  }
  cudaSetDevice(config.device_index);

  // createInferRuntime 是头文件里的 inline 函数，内部调这个导出符号。
  // 只解析它一个，其余全部走虚表，所以不需要 .lib。
  auto create_runtime = reinterpret_cast<CreateInferRuntimeFn>(
      ::GetProcAddress(impl_->nvinfer_dll, "createInferRuntime_INTERNAL"));
  if (create_runtime == nullptr) {
    error = "TensorRT DLL 里找不到 createInferRuntime_INTERNAL（版本不是 TRT 10？）";
    return false;
  }

  // 读引擎文件
  FILE* fp = nullptr;
  if (::fopen_s(&fp, config.engine_path.c_str(), "rb") != 0 || fp == nullptr) {
    error = "打不开引擎文件: " + config.engine_path;
    return false;
  }
  std::fseek(fp, 0, SEEK_END);
  const long size = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  if (size <= 0) {
    std::fclose(fp);
    error = "引擎文件为空: " + config.engine_path;
    return false;
  }
  std::vector<char> blob(static_cast<std::size_t>(size));
  const bool read_ok = std::fread(blob.data(), 1, blob.size(), fp) == blob.size();
  std::fclose(fp);
  if (!read_ok) {
    error = "读取引擎文件失败: " + config.engine_path;
    return false;
  }

  impl_->logger = new TrtLogger(log);
  impl_->runtime =
      static_cast<nvinfer1::IRuntime*>(create_runtime(impl_->logger, NV_TENSORRT_VERSION));
  if (impl_->runtime == nullptr) {
    error = "创建 TensorRT runtime 失败";
    return false;
  }

  impl_->engine = impl_->runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (impl_->engine == nullptr) {
    error = "反序列化引擎失败（引擎与本机 TensorRT / GPU 不匹配？）";
    return false;
  }

  impl_->context = impl_->engine->createExecutionContext();
  if (impl_->context == nullptr) {
    error = "创建执行上下文失败";
    return false;
  }

  // 找出输入/输出张量，同时拼一份 IO 摘要给日志用。
  const int32_t io_count = impl_->engine->getNbIOTensors();
  const char* input_name = nullptr;
  const char* output_name = nullptr;
  std::string summary;

  for (int32_t i = 0; i < io_count; ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    if (name == nullptr) {
      continue;
    }
    const nvinfer1::TensorIOMode mode = impl_->engine->getTensorIOMode(name);
    const nvinfer1::Dims dims = impl_->engine->getTensorShape(name);

    char line[256];
    int written = std::snprintf(line, sizeof(line), "%s %s [",
                                mode == nvinfer1::TensorIOMode::kINPUT ? "in" : "out", name);
    for (int d = 0; d < dims.nbDims; ++d) {
      if (written <= 0 || written >= static_cast<int>(sizeof(line)) - 8) {
        break;
      }
      written += std::snprintf(line + written, sizeof(line) - static_cast<std::size_t>(written),
                               "%s%lld", d == 0 ? "" : ",",
                               static_cast<long long>(dims.d[d]));
    }
    if (written > 0 && written < static_cast<int>(sizeof(line)) - 1) {
      std::snprintf(line + written, sizeof(line) - static_cast<std::size_t>(written), "]");
    }
    summary += line;
    summary += "; ";

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      // 配置里点名了就用它，否则取第一个
      if (input_name == nullptr || input_name != name) {
        if (input_name == nullptr || (!config.input_name.empty() && config.input_name == name)) {
          input_name = name;
        }
      }
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (output_name == nullptr || (!config.output_name.empty() && config.output_name == name)) {
        output_name = name;
      }
    }
  }
  impl_->io_summary = summary;

  if (input_name == nullptr || output_name == nullptr) {
    error = "引擎缺少输入或输出张量: " + summary;
    return false;
  }
  impl_->input_name = input_name;
  impl_->output_name = output_name;

  const nvinfer1::Dims in_dims = impl_->engine->getTensorShape(input_name);
  const nvinfer1::Dims out_dims = impl_->engine->getTensorShape(output_name);
  const std::size_t in_elements = dims_volume(in_dims.d, in_dims.nbDims);
  const std::size_t out_elements = dims_volume(out_dims.d, out_dims.nbDims);
  if (in_elements == 0 || out_elements == 0) {
    error = "引擎用了动态形状，本插件暂不支持（请导出固定 640x640 的引擎）";
    return false;
  }

  // 类型必须是 FP32。整个预处理 / 拷贝链路都按 float 写，遇到 FP16
  // 或 INT8 的 IO 张量会按 4 字节/元素分配，但 TensorRT 只写 2/1 字节，
  // 反过来也会把输入解释错。这里直接拒绝，不要带着错误假设跑。
  const nvinfer1::DataType in_type = impl_->engine->getTensorDataType(input_name);
  const nvinfer1::DataType out_type = impl_->engine->getTensorDataType(output_name);
  if (in_type != nvinfer1::DataType::kFLOAT || out_type != nvinfer1::DataType::kFLOAT) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "张量类型必须是 FP32（输入=%d 输出=%d）。"
                  "请用 trtexec 不指定 --inputIOFormats/--outputIOFormats，或导出 FP32 ONNX",
                  static_cast<int>(in_type), static_cast<int>(out_type));
    error = buf;
    return false;
  }

  if (in_dims.nbDims != 4 || in_dims.d[1] != 3) {
    error = "输入不是 NCHW 的 [1,3,H,W]，无法处理: " + summary;
    return false;
  }
  impl_->input_h = static_cast<int>(in_dims.d[2]);
  impl_->input_w = static_cast<int>(in_dims.d[3]);

  // ---- 输出布局 ----
  const int64_t last = out_dims.nbDims >= 2 ? out_dims.d[out_dims.nbDims - 1] : 0;
  const int64_t prev = out_dims.nbDims >= 2 ? out_dims.d[out_dims.nbDims - 2] : 0;
  if (last >= 5 && last <= 8) {
    impl_->layout = OutputLayout::kRows;
  } else if (prev >= 5 && last > prev) {
    impl_->layout = OutputLayout::kChw;
    impl_->chw_channels = static_cast<int>(prev);
    impl_->chw_count = static_cast<int>(last);
    if (config.num_classes > 0 && impl_->chw_channels != config.num_classes + 4) {
      char buf[192];
      std::snprintf(buf, sizeof(buf), "输出通道 %d 与 num_classes=%d 不一致: %s",
                    impl_->chw_channels, config.num_classes, summary.c_str());
      error = buf;
      return false;
    }
  } else if (config.output_stride > 0 &&
             out_elements % static_cast<std::size_t>(config.output_stride) == 0) {
    impl_->layout = OutputLayout::kRows;  // 形状不典型，按配置的 stride 当 rows
  } else {
    error = "无法识别的输出形状: " + summary;
    return false;
  }

  impl_->input_elements = in_elements;
  impl_->output_elements = out_elements;
  impl_->host_output.assign(out_elements, 0.0f);

  if (cudaMalloc(&impl_->device_input, in_elements * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&impl_->device_output, out_elements * sizeof(float)) != cudaSuccess) {
    error = "cudaMalloc 失败";
    return false;
  }
  // 非默认流：TRT 在默认流上会自己插 cudaStreamSynchronize，白掉一次同步。
  if (cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking) != cudaSuccess) {
    error = "创建 CUDA 流失败";
    return false;
  }

  // 预热。第一次 enqueueV3 要建内核句柄、初始化 CUDA 上下文，
  // 实测首帧 8-14 ms，稳定后 3.9 ms。这几十毫秒不该算在分析线程的
  // 第一帧上——那会白白丢掉开局的一两帧。这里跑两次空推理提前付掉。
  {
    std::vector<float> warm(in_elements, 0.5f);
    if (!impl_->context->setInputTensorAddress(input_name, impl_->device_input) ||
        !impl_->context->setOutputTensorAddress(output_name, impl_->device_output)) {
      error = "预热前设置张量地址失败";
      return false;
    }
    for (int pass = 0; pass < 2; ++pass) {
      if (cudaMemcpyAsync(impl_->device_input, warm.data(), warm.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess ||
          !impl_->context->enqueueV3(impl_->stream) ||
          cudaStreamSynchronize(impl_->stream) != cudaSuccess) {
        error = "预热推理失败";
        return false;
      }
    }
  }
  return true;
}

bool YoloDetector::infer(const std::vector<float>& input_nchw, const FrameGeometry& geom,
                         std::vector<Detection>& out, std::string& error) {
  out.clear();
  if (impl_ == nullptr || impl_->context == nullptr) {
    error = "推理器未初始化";
    return false;
  }
  if (input_nchw.size() != impl_->input_elements) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "输入元素数不符：%zu != %zu", input_nchw.size(),
                  impl_->input_elements);
    error = buf;
    return false;
  }

  if (cudaMemcpyAsync(impl_->device_input, input_nchw.data(), input_nchw.size() * sizeof(float),
                      cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess) {
    error = "cudaMemcpy 上传输入失败";
    return false;
  }

  if (!impl_->context->setInputTensorAddress(impl_->input_name.c_str(), impl_->device_input) ||
      !impl_->context->setOutputTensorAddress(impl_->output_name.c_str(), impl_->device_output)) {
    error = "设置张量地址失败";
    return false;
  }

  if (!impl_->context->enqueueV3(impl_->stream)) {
    error = "enqueueV3 失败";
    return false;
  }

  if (cudaMemcpyAsync(impl_->host_output.data(), impl_->device_output,
                      impl_->host_output.size() * sizeof(float), cudaMemcpyDeviceToHost,
                      impl_->stream) != cudaSuccess) {
    error = "cudaMemcpy 回读输出失败";
    return false;
  }
  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) {
    error = "cudaStreamSynchronize 失败";
    return false;
  }

  const float* data = impl_->host_output.data();
  const auto& keep = config_.keep_classes;
  auto class_allowed = [&keep](int cls) {
    return keep.empty() || std::find(keep.begin(), keep.end(), cls) != keep.end();
  };

  // 模型坐标 → 原始帧像素
  auto to_source = [&](float mx1, float my1, float mx2, float my2, Detection& d) {
    d.x1 = to_source_x(geom, mx1);
    d.y1 = to_source_y(geom, my1);
    d.x2 = to_source_x(geom, mx2);
    d.y2 = to_source_y(geom, my2);
  };

  if (impl_->layout == OutputLayout::kRows) {
    const std::size_t count =
        impl_->output_elements / static_cast<std::size_t>(config_.output_stride);
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const float* row = data + i * static_cast<std::size_t>(config_.output_stride);
      Detection d;
      d.score = row[4];
      d.class_id = static_cast<int>(row[5]);
      if (d.score < config_.conf_threshold || !class_allowed(d.class_id)) {
        continue;
      }
      to_source(row[0], row[1], row[2], row[3], d);
      out.push_back(d);
    }
    return true;
  }

  const int channels = impl_->chw_channels;
  const int count = impl_->chw_count;
  std::vector<Detection> raw;
  raw.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    int best_class = -1;
    float best_score = 0.0f;
    for (int c = 4; c < channels; ++c) {
      const float s = data[static_cast<std::size_t>(c) * count + i];
      if (s > best_score) {
        best_score = s;
        best_class = c - 4;
      }
    }
    if (best_class < 0 || best_score < config_.conf_threshold || !class_allowed(best_class)) {
      continue;
    }
    const float cx = data[i];
    const float cy = data[static_cast<std::size_t>(count) + i];
    const float w = data[static_cast<std::size_t>(2) * count + i];
    const float h = data[static_cast<std::size_t>(3) * count + i];
    Detection d;
    d.score = best_score;
    d.class_id = best_class;
    to_source(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, d);
    raw.push_back(d);
  }

  // chw 布局的模型没做 NMS，这里补一个
  std::sort(raw.begin(), raw.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  out.reserve(raw.size());
  for (const Detection& cand : raw) {
    bool suppressed = false;
    for (const Detection& kept : out) {
      if (kept.class_id == cand.class_id && iou(kept, cand) > config_.iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      out.push_back(cand);
    }
  }
  return true;
}

#else  // !DUO_YOLO_HAVE_TENSORRT

bool YoloDetector::init(const DetectConfig&, LogFn, std::string& error) {
  error = "本插件编译时没有找到 TensorRT 头文件（NvInfer.h），推理未启用";
  return false;
}

bool YoloDetector::infer(const std::vector<float>&, const FrameGeometry&, std::vector<Detection>&,
                         std::string& error) {
  error = "推理未启用";
  return false;
}

#endif  // DUO_YOLO_HAVE_TENSORRT

}  // namespace duo_yolo
