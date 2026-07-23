# bonafide_native — C++ / CUDA acceleration extension
#
# Recommended build entry points:
#   Windows : scripts\build_native_win.bat        (handles vcvars + version gaps)
#   any OS  : python scripts/build_native.py      (prereq doctor + build)
#
# Manual:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DPython_EXECUTABLE=$(which python)
#   cmake --build build --config Release -j
#
# Validated configurations:
#   * Windows : VS 2022 Build Tools (MSVC 19.44) + CUDA 11.7, Ninja generator.
#               nvcc 11.7 predates this MSVC, so the build script exports
#               NVCC_PREPEND_FLAGS="-allow-unsupported-compiler
#               -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
#               to silence the two independent version gates. The clean
#               long-term fix is CUDA >= 12.4 (matches modern MSVC).
#   * Linux   : GCC 11 + CUDA 12.x, Ninja or Make.
cmake_minimum_required(VERSION 3.24)
project(bonafide_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pick reasonable architectures based on the detected CUDA toolkit.
# CUDA 11.x maxes out at SM 8.7 (Ada/Hopper land in 12.x).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_VERSION VERSION_LESS 12.0)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    endif()
    message(STATUS "Auto-selected CUDA arches for ${CUDAToolkit_VERSION}: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Find Python (the one driving the build) — needed for nanobind.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind shipped with the active Python.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import nanobind, sys; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_RES
)
if(NOT NB_RES EQUAL 0)
    message(FATAL_ERROR
        "nanobind not found in the active Python (${Python_EXECUTABLE}). "
        "Install it: pip install nanobind"
    )
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------- extension target
nanobind_add_module(
    bonafide_native
    NB_STATIC
    src/bindings.cpp
    src/octree.cu
    src/surfel.cu
    src/splat.cu
    src/upload.cpp
)

target_include_directories(bonafide_native PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(bonafide_native PRIVATE
    CUDA::cudart
)

# Release-mode optimisation. Host-compiler flags differ between MSVC and
# GCC/Clang, so gate them on the compiler id rather than hard-coding -O3.
if(MSVC)
    target_compile_options(bonafide_native PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    )
else()
    target_compile_options(bonafide_native PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
endif()
target_compile_options(bonafide_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math
                              -lineinfo
                              -Xptxas=-v
                              $<$<CONFIG:Release>:-O3>>
)

# Drop the .pyd / .so where pip / conda will discover it.
install(TARGETS bonafide_native
        LIBRARY DESTINATION ${Python_SITELIB}
        RUNTIME DESTINATION ${Python_SITELIB})
