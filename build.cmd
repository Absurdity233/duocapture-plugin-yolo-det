@echo off
rem 用 MSVC + Ninja 构建 yolo_det 插件。
rem   build.cmd            使用环境变量 TensorRT_DIR / CUDA_PATH
rem   build.cmd <TRT> <CUDA> <DUOCAPTURE>
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH ( echo VS C++ tools not found & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set "NINJA=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "ROOT=%~dp0"
set "TRT=%1"
if "%TRT%"=="" set "TRT=%TensorRT_DIR%"
set "CUDA=%2"
if "%CUDA%"=="" set "CUDA=%CUDA_PATH%"
set "DUOSRC=%3"
if "%DUOSRC%"=="" set "DUOSRC=%ROOT%..\.."

cmake -S "%ROOT%." -B "%ROOT%build" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DDUOCAPTURE_DIR="%DUOSRC%" ^
  -DTENSORRT_DIR="%TRT%" ^
  -DDUO_CUDA_DIR="%CUDA%" || exit /b 1
cmake --build "%ROOT%build" || exit /b 1
echo.
echo Built: %ROOT%build\plugins\duo_plugin_yolo_det.dll
