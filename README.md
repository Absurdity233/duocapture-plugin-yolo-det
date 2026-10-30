# yolo_det — TensorRT 视觉识别插件

DuoCapture 的分析插件：把 A 端解码出来的每一帧喂给 TensorRT 上的 YOLO 模型，
输出目标类别、置信度和框坐标。

**只做识别，不做任何输入注入。** 这是 `docs/compliance.md` 里划的那条线：
画面交给程序「看」没问题，让程序回写进第三方系统的决策就不行。

## 数据流

```
on_frame（分析线程）
  → NV12 纹理 → 暂存纹理 CopyResource → Map
  → BT.709 limited → RGB → center_crop / letterbox 到模型输入
  → cudaMemcpy → TensorRT enqueueV3
  → 解析 rows / chw 输出 → NMS（chw 才需要）→ 映射回原始帧像素
```

预处理在 CPU 上做。分析 tap 本来就是「忙则丢帧」，慢的代价是少分析几帧，
不会拖慢呈现——所以这里优先选简单可靠的路径，而不是 GPU 上的零拷贝。

## 构建

```bat
plugins\yolo_det\build.cmd
```

脚本会自己找 Visual Studio、`%TensorRT_DIR%`、`%CUDA_PATH%`，也可以显式传：

```bat
build.cmd <TensorRT 根目录> "<CUDA 根目录>" <duocapture 仓库根目录>
```

产物：

```
build\plugins\duo_plugin_yolo_det.dll
build\yolo_selftest.exe          ← 端到端自检工具
```

**不链接 `nvinfer_10.lib`。** TensorRT 是 400 MB 的 DLL，装在哪不确定，
所以运行时用 `LoadLibrary` 按配置里的路径加载，只解析
`createInferRuntime_INTERNAL` 一个导出符号，其余全走头文件里的虚表。
只有 `cudart` 是链接进去的（`cudaMalloc` / `cudaMemcpy` 这类普通导出函数）。

没有 TensorRT 头文件时也能编译，推理退化成 `init` 直接失败并说明原因——
CI 和没装 TensorRT 的机器不会因此挂掉。

## 使用

```bat
duocapv.exe --play --peer 192.168.10.1 ^
  --plugin-dir plugins\yolo_det\build\plugins ^
  --plugin duo.yolo_det ^
  --plugin-config "engine=<模型>.engine,trt=<TensorRT>\bin\nvinfer_10.dll,conf=0.45,names=body|head,log=60"
```

## 配置项

配置串是逗号分隔的 `key=value`。**列表值用 `|` 分隔**，因为 `,` 已经是分隔符了。

| 键 | 默认 | 说明 |
|---|---|---|
| `engine` | — | `.engine` 文件路径（**必填**） |
| `trt` | — | `nvinfer_10.dll` 的完整路径（**必填**） |
| `input` | `images` | 输入张量名 |
| `output` | 自动 | 输出张量名，空 = 取第一个输出 |
| `crop` | `center_crop` | 预处理：`center_crop` 或 `letterbox` |
| `conf` | `0.45` | 置信度阈值 |
| `iou` | `0.45` | NMS IoU 阈值（只有 chw 布局用得上） |
| `stride` | `6` | rows 布局每行的 float 数 |
| `classes` | 全部 | 只保留这些类，如 `0\|1\|5` |
| `names` | — | 类名，按下标对应，只影响日志 |
| `num_classes` | 0 | chw 布局的类数，0 = 从形状推断 |
| `device` | `0` | CUDA 设备序号 |
| `every` | `1` | 每 N 帧推理一次（降频省 GPU） |
| `log` | `120` | 每 N 次推理打一条统计日志，`0` = 不打 |
| `dump` | — | 把第一次推理的输入导出成 PPM，用于核对预处理 |

## 两种预处理模式

| | center_crop（默认） | letterbox |
|---|---|---|
| 做法 | 裁掉中心正方形以外的部分再缩放 | 整帧等比缩放 + 居中补 114 灰边 |
| 视野 | 只有中心区域 | 完整画面 |
| 目标大小 | 更大，更易检出 | 更小 |
| 边缘目标 | **看不见** | 能看见 |
| 坐标反算 | `(model - 0) / scale + crop_x` | `(model - pad) / scale` |

1920×1080 送 640×640 时，center_crop 只保留中心 **1080×1080** 的正方形
（占原宽 59%），左右各丢 420 像素。实测 dump 出来的模型输入里，
内容区位于 x/y 130–509；letterbox 则是满宽 0–639。

两者输出都换算回**原始帧像素**，调用方无需区分。

选型建议：只关心画面中心（比如看准星附近）用 `center_crop`，目标更大、更容易检出；
要让整个画面都在监控范围内用 `letterbox`。

## 输出布局自动判定

| 形状 | 布局 | 处理 |
|---|---|---|
| `[1, N, 5..8]` | rows | 模型已做完 NMS，直接取 `[x1,y1,x2,y2,score,cls]` |
| `[1, 4+nc, N]` | chw | 逐候选挑最大类 + 自己做 NMS |

`rows` 是 YOLOv8/10/26 的默认导出格式。

## 模型文件

仓库**不包含**任何模型或引擎——每个引擎约 43 MB，且与本机的 GPU 和
TensorRT 版本绑定，换台机器就得重转。用你自己的 ONNX 导出即可：

```bat
trtexec.exe --onnx=<model>.onnx --saveEngine=<model>.engine --fp16
```

下文的对拍数据来自一个 7 类（`body` / `head` / `teammate` 等）的
YOLO26m，只作为「插件输出与 onnxruntime 一致」的证据，不是必需的模型。

## 与 onnxruntime 对拍

**预处理模式会影响分数。** 同一张 1920×1080 图上，center_crop 给同一目标的
分数高于 letterbox——因为前者把目标放大了。对比不同来源的分数前先确认模式一致。

**对拍要用同一份输入像素。** 拿原图分别跑 onnxruntime 和插件，
两边各自 `resize` 的插值算法不同（OpenCV INTER_LINEAR vs 插件双线性），
分数会差 0.3 以上——那是预处理差异，不是推理错误。先让插件
`dump=xxx.ppm` 导出它实际喂给模型的图，再用 onnxruntime 跑这张图，
两边应该逐位吻合。

## 自检

不需要两台机器，也不需要起接收端：

```bat
build\yolo_selftest.exe build\plugins\duo_plugin_yolo_det.dll ^
  <图片路径> ^
  "engine=<模型>.engine,trt=<TensorRT>\bin\nvinfer_10.dll,conf=0.35,log=1,names=body|head"
```

它做的事和主程序一致：建 D3D11 设备 → 图片转 NV12 纹理 → 加载 DLL →
`create` → `as_analysis` → `on_frame`，日志打到控制台。

实测（640×640 输入，RTX 5070 Laptop）每帧推理 3.6–4.8 ms，
与同模型的 onnxruntime 结果逐个对齐（类别一致，分数差 < 0.03）。

## 踩过的坑

**NV12 在 D3D11 里是单个子资源，不是两个。** UV 平面不是
`Map(1)` 能拿到的第二个子资源——它就在同一个子资源里，紧跟在
Y 平面之后，起点是 `RowPitch × height`。按「两个子资源」写会导致
UV 全零，画面偏绿，模型输出一堆低分垃圾框。`Map(1)` 在暂存纹理上
返回 `E_INVALIDARG (0x80070057)` 就是这个信号。

上传侧同理：要用一次 `UpdateSubresource` + `D3D11_BOX`（`top=height`，
`bottom=height*1.5`）写 UV，不能靠 `CreateTexture2D` 的 `pInitialData[1]`。
仓库里 `src/decode/nvdec_cuvid.cpp` 的 CPU 兜底路径就是这么写的。

**TensorRT 在默认流上会自己插一次 `cudaStreamSynchronize`。** 会打警告，
也白掉一次同步。用 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`。

**`createInferRuntime` 是头文件里的 inline 函数**，内部调
`createInferRuntime_INTERNAL(void* logger, int32_t version)`。只解析这一个符号
就够了，不需要 `.lib`，也不需要处理 TensorRT 的版本化命名空间。

## 边界

- 插件不碰输入、不写进程内存、不参与任何游戏内决策
- 结果只写主程序日志；要接下游（比如 RTMP 叠加框）应该在
  `plugin_api.h` 加新的 kind，而不是从插件里直接控制别的系统
