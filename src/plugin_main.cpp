// DuoCapture 分析插件：用 TensorRT 跑 YOLO 视觉识别。
//
// 这个插件只做「看」——识别画面里有哪些目标、在哪、置信度多少，
// 结果写进日志，不做任何输入注入或对外决策。主程序的合规边界
// （docs/compliance.md）在这里保持不变：只看不控。
//
// 数据流：
//   on_frame(分析线程) → NV12 纹理 → Map 回读 → center_crop + RGB
//   → TensorRT（CUDA）→ 解析 rows / chw 输出 → NMS → 原始帧坐标
//
// 输出格式有两种，按引擎的输出形状自动判定：
//   rows  [1, N, 6]         —— YOLOv8/10/26 导出时已做过 NMS
//   chw   [1, 4+nc, N]      —— 原始输出，插件自己挑类 + NMS
//
// 配置（--plugin-config，逗号分隔的 key=value）：
//   engine=<path>            .engine 文件（必填）
//   trt=<path>               nvinfer_10.dll 完整路径（必填）
//   input=images             输入张量名
//   output=output0           输出张量名，空 = 自动
//   conf=0.45                置信度阈值
//   iou=0.45                 NMS IoU 阈值（仅 chw 布局用）
//   stride=6                 rows 布局每行 float 数
//   classes=0|1|5|6          只保留这些类（`|` 分隔），空 = 全部
//   names=body|head|...      类名，按下标对应；只影响日志
//   device=0                 CUDA 设备
//   every=1                  每 N 帧推理一次
//   log=120                  每 N 帧打一次统计日志，0 = 不打

#include "duocapture/plugin/plugin_api.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "detector.h"
#include "frame_reader.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {

const DuoPluginHostServices* g_host = nullptr;

void host_log(int level, const char* message) {
  if (g_host != nullptr && g_host->log != nullptr && message != nullptr) {
    g_host->log(level, message);
  }
}

void log_info(const std::string& text) { host_log(DUO_LOG_INFO, text.c_str()); }
void log_warn(const std::string& text) { host_log(DUO_LOG_WARN, text.c_str()); }
void log_error(const std::string& text) { host_log(DUO_LOG_ERROR, text.c_str()); }

// 极简 key=value 解析。配置串是逗号分隔，值里允许出现空格。
std::map<std::string, std::string> parse_config(const char* config) {
  std::map<std::string, std::string> out;
  if (config == nullptr) {
    return out;
  }
  std::string text(config);
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string item =
        text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    const std::size_t eq = item.find('=');
    if (eq != std::string::npos) {
      std::string key = item.substr(0, eq);
      std::string value = item.substr(eq + 1);
      auto trim = [](std::string& s) {
        const char* ws = " \t\r\n";
        const std::size_t b = s.find_first_not_of(ws);
        const std::size_t e = s.find_last_not_of(ws);
        s = b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
      };
      trim(key);
      trim(value);
      if (!key.empty()) {
        out[key] = value;
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

// 列表值用 '|' 分隔，不能用 ','——配置串本身就按逗号切 key=value。
// 例：classes=0|1|5
std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t bar = text.find('|', start);
    const std::string item =
        text.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
    if (!item.empty()) {
      out.push_back(std::atoi(item.c_str()));
    }
    if (bar == std::string::npos) {
      break;
    }
    start = bar + 1;
  }
  return out;
}

std::vector<std::string> parse_str_list(const std::string& text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t bar = text.find('|', start);
    std::string item =
        text.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
    auto trim = [](std::string& s) {
      const char* ws = " \t\r\n";
      const std::size_t b = s.find_first_not_of(ws);
      const std::size_t e = s.find_last_not_of(ws);
      s = b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
    };
    trim(item);
    if (!item.empty()) {
      out.push_back(item);
    }
    if (bar == std::string::npos) {
      break;
    }
    start = bar + 1;
  }
  return out;
}

struct YoloPlugin {
  duo_yolo::YoloDetector detector;
  duo_yolo::Nv12FrameReader reader;

  std::vector<float> input;
  std::vector<duo_yolo::Detection> detections;
  std::vector<std::string> class_names;

  int every = 1;
  int log_interval = 120;
  std::string dump_path;
  int dump_written = 0;
  std::atomic<unsigned long long> frames{0};
  std::atomic<unsigned long long> inferred{0};
  std::atomic<unsigned long long> failed{0};
  std::atomic<unsigned long long> total_detections{0};
  std::atomic<int> last_detections{0};
  std::atomic<double> last_infer_ms{0.0};

  // 给宿主呈现线程取用的检测框快照。分析线程写、呈现线程读。
  std::mutex overlay_mutex;
  std::vector<DuoOverlayBox> overlay_boxes;

  bool ready = false;
  bool warned_no_device = false;
};

// TensorRT 日志回调。签名要和 detector.h 里的 LogFn 对上。
void trt_log(int level, const char* message) {
  host_log(level, message != nullptr ? message : "");
}

int on_frame(void* instance, const DuoFrameView* frame) {
  if (instance == nullptr || frame == nullptr || frame->texture == nullptr) {
    return -1;
  }
  auto* self = static_cast<YoloPlugin*>(instance);

  const unsigned long long n = self->frames.fetch_add(1) + 1;
  if (self->every > 1 && (n % static_cast<unsigned long long>(self->every)) != 0) {
    return 0;
  }

  if (!self->ready) {
    return -1;
  }

  // 设备在插件加载时可能还不存在（呈现窗口还没建好），这里每次查。
  auto* device = static_cast<ID3D11Device*>(g_host != nullptr ? g_host->d3d_device : nullptr);
  auto* context = static_cast<ID3D11DeviceContext*>(g_host != nullptr ? g_host->d3d_context : nullptr);
  if (device == nullptr || context == nullptr) {
    if (!self->warned_no_device) {
      self->warned_no_device = true;
      log_warn("yolo_det: D3D11 设备尚未就绪，暂时无法推理");
    }
    return -1;
  }

  const int w = self->detector.input_width();
  const int h = self->detector.input_height();
  auto* texture = static_cast<ID3D11Texture2D*>(frame->texture);

  duo_yolo::FrameGeometry geom;
  std::string error;
  if (!self->reader.read(device, context, texture, w, h, self->input, geom, error)) {
    self->failed.fetch_add(1);
    log_warn("yolo_det: 读帧失败: " + error);
    return -1;
  }

  // 调试用：把真正喂给模型的图导出成 PPM，用来对比预处理是否符合预期。
  if (!self->dump_path.empty() && self->dump_written < 1) {
    self->dump_written = 1;
    if (FILE* fp = nullptr; ::fopen_s(&fp, self->dump_path.c_str(), "wb") == 0 && fp != nullptr) {
      std::fprintf(fp, "P6\n%d %d\n255\n", w, h);
      for (int i = 0; i < w * h; ++i) {
        const unsigned char px[3] = {
            static_cast<unsigned char>(self->input[i] * 255.0f),
            static_cast<unsigned char>(self->input[static_cast<std::size_t>(w) * h + i] * 255.0f),
            static_cast<unsigned char>(self->input[static_cast<std::size_t>(2) * w * h + i] * 255.0f)};
        std::fwrite(px, 1, 3, fp);
      }
      std::fclose(fp);
      log_info("yolo_det: 已导出预处理结果到 " + self->dump_path);
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  if (!self->detector.infer(self->input, geom, self->detections, error)) {
    self->failed.fetch_add(1);
    log_warn("yolo_det: 推理失败: " + error);
    return -1;
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  self->last_infer_ms.store(ms);

  const unsigned long long done = self->inferred.fetch_add(1) + 1;
  self->last_detections.store(static_cast<int>(self->detections.size()));
  self->total_detections.fetch_add(self->detections.size());

  // 交给宿主画框。坐标已经是原始帧像素，宿主自己缩放到窗口。
  {
    std::lock_guard<std::mutex> lock(self->overlay_mutex);
    self->overlay_boxes.clear();
    self->overlay_boxes.reserve(self->detections.size());
    for (const duo_yolo::Detection& d : self->detections) {
      DuoOverlayBox box{};
      box.x1 = d.x1;
      box.y1 = d.y1;
      box.x2 = d.x2;
      box.y2 = d.y2;
      box.score = d.score;
      box.class_id = d.class_id;
      box.color = 0;  // 让宿主按类别配色
      self->overlay_boxes.push_back(box);
    }
  }

  if (self->log_interval > 0 && done % static_cast<unsigned long long>(self->log_interval) == 0) {
    char buf[320];
    std::snprintf(buf, sizeof(buf), "yolo_det: 第 %llu 帧，%zu 个目标，推理 %.2f ms（累计 %llu 个）",
                  done, self->detections.size(), ms, self->total_detections.load());
    log_info(buf);

    // 顺带把这一帧每个目标的类名/置信度/中心点打出来，方便核对模型输出对不对。
    for (const duo_yolo::Detection& d : self->detections) {
      const char* name = "?";
      if (d.class_id >= 0 && static_cast<std::size_t>(d.class_id) < self->class_names.size()) {
        name = self->class_names[static_cast<std::size_t>(d.class_id)].c_str();
      }
      char line[256];
      std::snprintf(line, sizeof(line), "yolo_det:   %s(%d) %.3f @ [%.0f,%.0f]-[%.0f,%.0f]",
                    name, d.class_id, static_cast<double>(d.score), static_cast<double>(d.x1),
                    static_cast<double>(d.y1), static_cast<double>(d.x2),
                    static_cast<double>(d.y2));
      log_info(line);
    }
  }

  return static_cast<int>(self->detections.size());
}

void on_reset(void* instance) {
  if (instance == nullptr) {
    return;
  }
  auto* self = static_cast<YoloPlugin*>(instance);
  self->reader.reset();
  self->frames.store(0);
  // 旧流的检测框必须一起丢掉：坐标是相对上一段流的，重置后继续画会
  // 停在错误的位置（尤其是断流后不再推理，宿主仍会每帧画这些旧框）。
  {
    std::lock_guard<std::mutex> lock(self->overlay_mutex);
    self->overlay_boxes.clear();
  }
  log_info("yolo_det: 流已重置");
}

void destroy_instance(void* instance) { delete static_cast<YoloPlugin*>(instance); }

int create(const DuoPluginHostServices* host, const char* config, void** out_instance) {
  if (out_instance == nullptr) {
    return -1;
  }
  *out_instance = nullptr;
  g_host = host;

  const auto cfg = parse_config(config);

  auto* self = new YoloPlugin();

  duo_yolo::DetectConfig dc;
  auto get = [&cfg](const char* key, const std::string& fallback) {
    const auto it = cfg.find(key);
    return it == cfg.end() ? fallback : it->second;
  };
  dc.engine_path = get("engine", "");
  dc.tensorrt_dll = get("trt", get("tensorrt", ""));
  dc.input_name = get("input", "images");
  dc.output_name = get("output", "");
  dc.conf_threshold = static_cast<float>(std::atof(get("conf", "0.45").c_str()));
  dc.iou_threshold = static_cast<float>(std::atof(get("iou", "0.45").c_str()));
  dc.output_stride = std::atoi(get("stride", "6").c_str());
  dc.num_classes = std::atoi(get("num_classes", "0").c_str());
  dc.device_index = std::atoi(get("device", "0").c_str());
  dc.keep_classes = parse_int_list(get("classes", ""));

  self->class_names = parse_str_list(get("names", ""));
  self->every = std::max(1, std::atoi(get("every", "1").c_str()));
  self->log_interval = std::max(0, std::atoi(get("log", "120").c_str()));
  self->dump_path = get("dump", "");

  std::string error;
  const auto init_begin = std::chrono::steady_clock::now();
  if (!self->detector.init(dc, &trt_log, error)) {
    // 初始化失败仍然返回实例：这样配置错了能在日志里看到原因，
    // 而不是静默地「插件没加载」。
    log_error("yolo_det: 初始化失败: " + error);
    *out_instance = self;
    return 0;
  }
  const double init_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - init_begin)
          .count();
  self->ready = true;

  char buf[320];
  std::snprintf(buf, sizeof(buf),
                "yolo_det: 就绪（引擎加载 + 预热 %.0f ms），输入 %dx%d（center_crop），"
                "conf %.2f，每 %d 帧推理一次",
                init_ms, self->detector.input_width(), self->detector.input_height(),
                static_cast<double>(dc.conf_threshold), self->every);
  log_info(buf);
  log_info("yolo_det: IO = " + self->detector.io_summary());

  *out_instance = self;
  return 0;
}

// 呈现线程调用：拷贝一份框给宿主。
int get_overlay_boxes(void* instance, DuoOverlayBox* out_boxes, int max_boxes) {
  if (instance == nullptr || out_boxes == nullptr || max_boxes <= 0) {
    return 0;
  }
  auto* self = static_cast<YoloPlugin*>(instance);
  std::lock_guard<std::mutex> lock(self->overlay_mutex);
  const int count = static_cast<int>(self->overlay_boxes.size()) > max_boxes
                        ? max_boxes
                        : static_cast<int>(self->overlay_boxes.size());
  for (int i = 0; i < count; ++i) {
    out_boxes[i] = self->overlay_boxes[static_cast<std::size_t>(i)];
  }
  return count;
}

const DuoAnalysisPlugin* as_analysis(void* instance) {
  if (instance == nullptr) {
    return nullptr;
  }
  static const DuoAnalysisPlugin api = {
      DUO_PLUGIN_ABI_VERSION,
      sizeof(DuoAnalysisPlugin),
      nullptr,  // instance 由宿主持有
      &on_frame,
      &on_reset,
      &destroy_instance,
      &get_overlay_boxes,  // ABI 1 之后追加；老宿主按 struct_size 忽略它
  };
  return &api;
}

const DuoPlugin g_plugin = {
    DUO_PLUGIN_ABI_VERSION,
    sizeof(DuoPlugin),
    DUO_PLUGIN_KIND_ANALYSIS,
    "duo.yolo_det",
    "1.0.0",
    "TensorRT YOLO 视觉识别：逐帧检测目标，只识别不注入",
    &create,
    &as_analysis,
    &destroy_instance,
};

}  // namespace

extern "C" DUO_PLUGIN_EXPORT const DuoPlugin* duo_plugin_entry(unsigned int host_abi_version) {
  if (host_abi_version != DUO_PLUGIN_ABI_VERSION) {
    return nullptr;
  }
  return &g_plugin;
}
