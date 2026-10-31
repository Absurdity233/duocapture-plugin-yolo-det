# yolo_det — TensorRT 视觉识别插件

DuoCapture 的分析插件：把 A 端解码出来的每一帧喂给 TensorRT 上的 YOLO 模型，
输出目标类别、置信度和框坐标。

**只做识别，不做任何输入注入。** 这是 `docs/compliance.md` 里划的那条线：
画面交给程序「看」没问题，让程序回写进第三方系统的决策就不行。

## 数据流

```
on_frame（分析线程）
  → NV12 纹理 → 暂存纹理 CopyResource → Map
  → BT.709 limited → RGB → center_crop 到模型输入
  → cudaMemcpy → TensorRT enqueueV3
  → 解析 rows / chw 输出 → NMS（chw 才需要）→ 映射回原始帧像素
  → 交给宿主合成检测框（插件不直接写画面）
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

## 预处理：center_crop

取与模型输入同长宽比的中心区域，等比缩放到模型尺寸。不补边，
模型输入的每一格都来自真实像素。

```
src = model / scale + crop
```

**注意视野损失。** 1920×1080 送 640×640 时只保留中心 **1080×1080**
的正方形（占原宽 59%），左右各丢 420 像素。落在裁剪区外的目标
永远不会被检出。

输出的框坐标已换算回**原始帧像素**，调用方直接在上面画即可。
想确认裁掉了哪一块，用 `dump=` 导出实际喂给模型的图。

## 检测框绘制

识别出来的框会画在呈现画面上（绿色边框），由**宿主**合成，不是插件直接写。

原因是线程模型：分析线程和呈现线程共用同一张解码纹理
（`receiver` 把同一个 `DecodedFrame` 同时交给两边），插件往纹理上写会和
`VideoProcessorBlt` 抢同一块显存，画面会闪。所以插件只通过
`get_overlay_boxes` 上报坐标，宿主在 `Present` 前把框叠到后备缓冲上——
和遥测文字用的是同一条路径。

这个字段是 ABI 1 之后追加的，靠 `struct_size` 做单向校验，
所以老插件（没有该字段）仍能加载，不需要升 ABI 版本。

## 支持哪些模型

**输入尺寸不写死，从引擎的张量形状里读。** 换模型只需要重新导引擎，
插件代码不用改。已实测 640×640 和 416×416 两个引擎；非正方形
（如 640×384）在几何换算上也是对的，但还没有真实引擎验证过。

能用的模型必须同时满足：

| 条件 | 原因 |
|---|---|
| 固定输入形状 | 动态形状的 `dims_volume()` 为 0，插件直接拒绝 |
| NCHW `[1, 3, H, W]` | 预处理按 channel-planar 写 |
| FP32 输入/输出张量 | 整条链路按 4 字节/元素处理，FP16/INT8 会错位 |
| 输出为 `[1,N,5..8]` 或 `[1,4+nc,N]` | 见下节 |
| `conf` 输出已是 0..1 | 模型自带 sigmoid；如果是 logits 需要先 sigmoid |

引擎的 FP16 权重（`trtexec --fp16`）没问题——那是内部计算精度，
和 IO 张量类型是两回事。被拒绝的是 `--inputIOFormats=fp16` 这种
改了 IO 类型的引擎。

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

**裁剪会放大目标。** 同一目标在 center_crop 输入里占的像素比整帧缩放时多，
所以分数会偏高。拿本插件和别的工具对比分数前，先确认对方也是 center_crop。

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

`box_overlay_test.exe` 单独验证宿主的绘制：造一张全黑纹理，用已知坐标
调 `draw()`，回读像素检查边框有颜色、框内没被涂满、框外保持透明。

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
- 插件只算坐标，不直接改画面；绘制由宿主在自己的呈现线程完成
- 要接更多下游（比如 RTMP 叠加、录制烧字幕）应该在
  `plugin_api.h` 加新的 kind，而不是从插件里直接控制别的系统
