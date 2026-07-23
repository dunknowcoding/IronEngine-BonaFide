@echo off
REM ====================================================================
REM  build_native_win.bat - build bonafide_native on Windows
REM
REM  Activates the VS 2022 Build Tools x64 environment, puts a known-good
REM  CMake + Ninja first on PATH (so a stray MinGW gcc / STM32 toolchain
REM  doesn't get auto-picked), forces the MSVC host compiler, and builds
REM  with the Ninja generator (no VS .props integration required).
REM
REM  Usage:   scripts\build_native_win.bat [Release|Debug]
REM ====================================================================
setlocal EnableDelayedExpansion

set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"

set "REPO=%~dp0.."
set "NATIVE=%REPO%\native"
set "BUILD=%NATIVE%\build"
set "PYEXE=G:\Anaconda\envs\IronEngineWorld\python.exe"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "CMAKE_BIN=C:\ST\STM32CubeCLT_1.19.0\CMake\bin"
set "NINJA_BIN=C:\ST\STM32CubeCLT_1.19.0\Ninja\bin"

echo === activating VS 2022 x64 toolchain ===
call "%VCVARS%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed.
    exit /b 1
)

REM Put cmake + ninja FIRST so stray toolchains lose the lookup race.
set "PATH=%CMAKE_BIN%;%NINJA_BIN%;%PATH%"

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe still not on PATH after vcvars.
    exit /b 1
)
echo cl.exe:    & where cl.exe
echo cmake:     & where cmake
echo ninja:     & where ninja

REM Bridge the CUDA 11.7 <-> MSVC 14.44 version gap. Two independent
REM version gates have to be silenced:
REM   1. nvcc's own host_config.h check  -> -allow-unsupported-compiler
REM   2. the MSVC STL's STL1002 assert   -> -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
REM NVCC_PREPEND_FLAGS is honoured by EVERY nvcc invocation, including
REM CMake's compiler-ID probe (which ignores -DCMAKE_CUDA_FLAGS).
REM Safe for the small, STL-light kernels we author. The clean long-term
REM fix is CUDA >= 12.4 (matches this MSVC) - see native/README.md.
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

echo.
echo === configure (Ninja, %CFG%) ===
cmake -S "%NATIVE%" -B "%BUILD%" -G Ninja ^
      -DCMAKE_BUILD_TYPE=%CFG% ^
      -DCMAKE_C_COMPILER=cl ^
      -DCMAKE_CXX_COMPILER=cl ^
      -DPython_EXECUTABLE="%PYEXE%"
if errorlevel 1 (
    echo ERROR: CMake configure failed.
    exit /b 1
)

echo.
echo === build ===
cmake --build "%BUILD%" --config %CFG% -j
if errorlevel 1 (
    echo ERROR: build failed.
    exit /b 1
)

echo.
echo === install into site-packages ===
"%PYEXE%" "%REPO%\scripts\_install_native.py" "%BUILD%"
if errorlevel 1 (
    echo ERROR: install step failed.
    exit /b 1
)

echo.
echo === DONE ===
endlocal
