@echo off
REM ====================================================================
REM  build_native_win.bat - build bonafide_native on Windows
REM
REM  Activates the VS 2022 Build Tools x64 environment, puts a known-good
REM  CMake + Ninja first on PATH (so a stray MinGW gcc or other toolchain
REM  doesn't get auto-picked), forces the MSVC host compiler, and builds
REM  with the Ninja generator (no VS .props integration required).
REM
REM  Usage:   scripts\build_native_win.bat [Release|Debug]
REM  Python:  set BONAFIDE_PYTHON=C:\path\to\python.exe to choose the
REM           interpreter (default: first python.exe on PATH).
REM ====================================================================
setlocal EnableDelayedExpansion

set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"

set "REPO=%~dp0.."
set "NATIVE=%REPO%\native"
set "BUILD=%NATIVE%\build"
REM Python to build against: set BONAFIDE_PYTHON to override, otherwise the
REM first python.exe on PATH is used.
if "%BONAFIDE_PYTHON%"=="" (
    for /f "delims=" %%i in ('where python.exe 2^>nul') do (
        if not defined BONAFIDE_PYTHON set "BONAFIDE_PYTHON=%%i"
    )
)
if "%BONAFIDE_PYTHON%"=="" (
    echo ERROR: no python.exe on PATH; set BONAFIDE_PYTHON to your interpreter.
    exit /b 1
)
set "PYEXE=%BONAFIDE_PYTHON%"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "CMAKE_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"

echo === activating VS 2022 x64 toolchain ===
call "%VCVARS%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed.
    exit /b 1
)

REM Put system CMake and the CUDA toolkit first; vcvars supplies Ninja + cl.
if "%CUDA_PATH%"=="" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CMAKE_BIN%;%CUDA_PATH%\bin;%PATH%"

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe still not on PATH after vcvars.
    exit /b 1
)
echo cl.exe:    & where cl.exe
echo cmake:     & where cmake
echo ninja:     & where ninja

REM Only CUDA < 12.4 needs the MSVC version-gap workarounds; CUDA 13.x
REM officially supports this MSVC, so no NVCC_PREPEND_FLAGS are set here.
REM (For CUDA 11.7 use:
REM   set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
REM )

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
