cmake_minimum_required(VERSION 3.20)

# 独立插件工程：只依赖 duocapture 的 plugin_api.h，不链接主程序。
#
#   cmake -S . -B build -DDUOCAPTURE_DIR=F:/duocapture -DTENSORRT_DIR=F:/TensorRT-10.16.1.11
#   cmake --build build
#
# 产物在 build/duo_plugin_yolo_det.dll，直接
#   duocapv --play --plugin-dir build --plugin duo.yolo_det --plugin-config "..."
# 使用。

project(duo_plugin_yolo_det VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认 Release 而不是 Debug：这个插件是拿来跑实时推理的，
# Debug（/Od /RTC1）下预处理能慢到 30 ms/帧，完全没法用。
# 注意必须在这里 FORCE 掉——本机的 CMake 会在 project() 之前就把
# CMAKE_BUILD_TYPE 设成 Debug（最小工程也一样能复现），
# 所以不能靠 "if(NOT CMAKE_BUILD_TYPE)" 判断。
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- duocapture 头文件 -------------------------------------------------------
# 只需要 include/duocapture/plugin/plugin_api.h 这一个纯 C 头文件。
if(NOT DUOCAPTURE_DIR)
  set(DUOCAPTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()
# 统一成正斜杠：CMake 在 add_executable 里会把反斜杠当转义符（\d 之类直接报错）。
file(TO_CMAKE_PATH "${DUOCAPTURE_DIR}" DUOCAPTURE_DIR)
set(DUO_PLUGIN_INCLUDE "${DUOCAPTURE_DIR}/include")
if(NOT EXISTS "${DUO_PLUGIN_INCLUDE}/duocapture/plugin/plugin_api.h")
  message(FATAL_ERROR
    "找不到 duocapture 的插件 ABI 头文件：${DUO_PLUGIN_INCLUDE}/duocapture/plugin/plugin_api.h\n"
    "用 -DDUOCAPTURE_DIR=<duocapture 仓库根目录> 指定。")
endif()

# ---- TensorRT（可选）--------------------------------------------------------
# 有头文件就开推理，没有就编一个「init 直接失败」的空壳——
# 这样 CI 和没装 TensorRT 的机器也能编译通过。
set(TENSORRT_DIR "$ENV{TensorRT_DIR}" CACHE PATH "TensorRT root (contains include/ and lib/)")
set(DUO_YOLO_HAVE_TENSORRT OFF)
if(TENSORRT_DIR AND EXISTS "${TENSORRT_DIR}/include/NvInfer.h")
  set(DUO_YOLO_HAVE_TENSORRT ON)
  message(STATUS "TensorRT: ${TENSORRT_DIR}")
else()
  message(WARNING
    "未找到 TensorRT（-DTENSORRT_DIR=... 或环境变量 TensorRT_DIR）。\n"
    "插件仍会编译，但推理不可用。")
endif()

# ---- CUDA（TensorRT 运行时依赖 cudart）--------------------------------------
set(DUO_CUDA_DIR "$ENV{CUDA_PATH}" CACHE PATH "CUDA Toolkit root")
if(DUO_YOLO_HAVE_TENSORRT AND NOT EXISTS "${DUO_CUDA_DIR}/include/cuda_runtime_api.h")
  message(FATAL_ERROR
    "找到了 TensorRT 但没找到 CUDA（需要 cuda_runtime_api.h）。\n"
    "用 -DDUO_CUDA_DIR=<CUDA 安装目录> 指定。")
endif()

set(DUO_YOLO_SOURCES
  src/plugin_main.cpp
  src/detector.cpp
  src/frame_reader.cpp
)

add_library(duo_plugin_yolo_det MODULE ${DUO_YOLO_SOURCES})
target_include_directories(duo_plugin_yolo_det PRIVATE
  "${DUO_PLUGIN_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(duo_plugin_yolo_det PRIVATE
  WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
target_link_libraries(duo_plugin_yolo_det PRIVATE d3d11 dxgi)

if(DUO_YOLO_HAVE_TENSORRT)
  # 用 SYSTEM 前缀：TensorRT 头文件里有一堆 C4996（自家 API 标弃用）和
  # C4100（未使用参数），不屏蔽的话 /W4 下会刷 60+ 条和我们无关的警告。
  target_include_directories(duo_plugin_yolo_det SYSTEM PRIVATE
    "${TENSORRT_DIR}/include"
    "${DUO_CUDA_DIR}/include"
  )
  # 注意：不链接 nvinfer_10.lib，TensorRT 全部走 LoadLibrary + 虚表。
  # 只链接 cudart，因为 cudaMalloc / cudaMemcpy 是普通导出函数，
  # 而且 nvcc 运行时本来就要求系统里装了 CUDA。
  find_library(DUO_CUDART_LIB NAMES cudart cudart_static
    PATHS "${DUO_CUDA_DIR}/lib/x64" NO_DEFAULT_PATH)
  if(NOT DUO_CUDART_LIB)
    message(FATAL_ERROR "找不到 cudart.lib，检查 -DDUO_CUDA_DIR 是否指向 CUDA Toolkit 根目录")
  endif()
  target_link_libraries(duo_plugin_yolo_det PRIVATE "${DUO_CUDART_LIB}")
  target_compile_definitions(duo_plugin_yolo_det PRIVATE DUO_YOLO_ENABLE_TENSORRT=1)
endif()

set_target_properties(duo_plugin_yolo_det PROPERTIES
  PREFIX ""
  OUTPUT_NAME "duo_plugin_yolo_det"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)

# ---- 自检工具（可选）--------------------------------------------------------
# 把一张图片做成 NV12 纹理，按宿主的方式调用插件，用来验证 ABI、
# 预处理和推理链路，不需要真的开两台机器。
option(DUO_YOLO_BUILD_SELFTEST "Build the plugin selftest harness" ON)
if(DUO_YOLO_BUILD_SELFTEST)
  add_executable(yolo_selftest tools/selftest.cpp)
  target_include_directories(yolo_selftest PRIVATE "${DUO_PLUGIN_INCLUDE}")
  target_compile_definitions(yolo_selftest PRIVATE
    WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
  target_link_libraries(yolo_selftest PRIVATE d3d11 dxgi windowscodecs ole32 uuid)
  set_target_properties(yolo_selftest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # 验证宿主的 BoxOverlay 真的把框画上去了（不是只编译通过）。
  # 需要 duocapture 的源码树，因为它要用到 box_overlay.cpp。
  if(EXISTS "${DUOCAPTURE_DIR}/src/present/box_overlay.cpp")
    add_executable(box_overlay_test tools/box_overlay_test.cpp
      "${DUOCAPTURE_DIR}/src/present/box_overlay.cpp")
    target_include_directories(box_overlay_test PRIVATE "${DUO_PLUGIN_INCLUDE}")
    target_compile_definitions(box_overlay_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
    target_link_libraries(box_overlay_test PRIVATE d3d11 dxgi d3dcompiler)
    set_target_properties(box_overlay_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  else()
    message(STATUS "跳过 box_overlay_test：找不到 duocapture 源码树")
  endif()
endif()

