#pragma once

// TensorRT 推理封装。
//
// 刻意不链接 nvinfer_10.lib：TensorRT 是个 400 MB 的 DLL，装在哪台机器上
// 不确定，也不该强制用户在 PATH 里加它。这里用 LoadLibrary 按配置里的路径
// 加载，只解析 `createInferRuntime_INTERNAL` 一个导出符号——其余全部走
// 头文件里的纯虚接口，不需要导入库。
//
// 编译时找不到 NvInfer.h 也能过：所有 TensorRT 相关代码都在
// DUO_YOLO_HAVE_TENSORRT 下面，没有头文件就退化成「init 直接失败」。

#include <string>
#include <vector>

#include "geometry.h"

namespace duo_yolo {

struct Detection {
  float x1 = 0.0f;  // 原始帧像素坐标
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct DetectConfig {
  std::string tensorrt_dll;  // nvinfer_10.dll 的完整路径
  std::string engine_path;   // .engine；或 .onnx（首次加载时现编，结果缓存）
  std::string input_name = "images";
  std::string output_name;   // 空 = 自动取第一个输出
  int num_classes = 0;       // 0 = 从输出维度推断（仅 chw 布局需要）
  int output_stride = 6;     // rows 布局：每行多少个 float
  int device_index = 0;      // CUDA 设备序号
  float conf_threshold = 0.45f;
  float iou_threshold = 0.45f;
  std::vector<int> keep_classes;  // 空 = 全部保留

  // ---- ONNX 现编（engine_path 以 .onnx 结尾时才用）---------------------
  std::string onnx_parser_dll;  // 空 = nvinfer 同目录下的 nvonnxparser_10.dll
  std::string trtexec_path;     // 空 = nvinfer 同目录下的 trtexec.exe（回退用）
  bool build_fp16 = true;       // 构建时允许 FP16 层
  std::size_t workspace_mb = 2048;
  std::string engine_cache;     // 空 = <onnx 路径>.engine
  bool force_rebuild = false;   // 忽略缓存，强制重编
};

// TensorRT 自己的日志（反序列化、执行期警告）。插件在 create() 里绑定，
// 级别与主程序一致：0 trace / 1 debug / 2 info / 3 warn / 4 error。
using LogFn = void (*)(int level, const char* message);

class YoloDetector {
 public:
  YoloDetector();
  ~YoloDetector();

  YoloDetector(const YoloDetector&) = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  bool init(const DetectConfig& config, LogFn log, std::string& error);

  // input_nchw 是预处理后的 RGB（NCHW，值域 0..1）。
  // geom 把模型像素坐标映射回原始帧像素。
  bool infer(const std::vector<float>& input_nchw, const FrameGeometry& geom,
             std::vector<Detection>& out, std::string& error);

  [[nodiscard]] int input_width() const noexcept;
  [[nodiscard]] int input_height() const noexcept;
  [[nodiscard]] const std::string& io_summary() const noexcept;

 private:
  struct Impl;
  DetectConfig config_;
  Impl* impl_ = nullptr;
};

}  // namespace duo_yolo
