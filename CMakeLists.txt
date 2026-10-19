# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(triton_migraphx_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# 1. 查找 ROCm 与 HIP 环境
# ==============================================================================
if(NOT DEFINED ENV{DTK_PATH})
  set(DTK_PATH "/opt/dtk" CACHE PATH "Path to ROCm installation")
else()
  set(DTK_PATH $ENV{DTK_PATH} CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED)
message(STATUS "Found HIP: ${hip_VERSION} at ${ROCM_PATH}")

# ==============================================================================
# 2. 查找 MIGraphX (包含 migraphx::c)
# ==============================================================================
find_package(migraphx REQUIRED)
message(STATUS "Found MIGraphX at ${migraphx_DIR}")

# ==============================================================================
# 3. 引入 Triton Backend 辅助库 (triton-backend-utils)
# ==============================================================================
# 允许用户通过 -DTRITON_CORE_REPO_TAG=r24.05 指定适配的 Triton 发行版本
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Triton Server release branch/tag")

# ==============================================================================
# 4. 构建 MIGraphX 后端共享库
# ==============================================================================
# Triton 约定的后端共享库命名格式为 libtriton_<backend_name>.so
add_library(triton_migraphx SHARED
  src/migraphx_backend.cc
)

# 设置头文件包含目录
target_include_directories(triton_migraphx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${DTK_PATH}/include
  ${TRITON_BACKEND_PATH}/include
  ${TRITON_CORE_PATH}/include/
)

add_library(triton-backend-utils STATIC IMPORTED GLOBAL)
set_target_properties(triton-backend-utils PROPERTIES
	IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritonbackendutils.a)
add_library(triton-common-async-work-queue STATIC IMPORTED GLOBAL)
set_target_properties(triton-common-async-work-queue PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritonasyncworkqueue.a)
add_library(triton-common-logging STATIC IMPORTED GLOBAL)
set_target_properties(triton-common-logging PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libtritoncommonlogging.a)

add_library(kernel_library_new STATIC IMPORTED GLOBAL)
set_target_properties(kernel_library_new PROPERTIES
        IMPORTED_LOCATION ${TRITON_BACKEND_PATH}/libkernel_library_new.a)

# 链接依赖项：MIGraphX C API、ROCm HIP 运行时、Triton 后端辅助工具库
target_link_libraries(triton_migraphx PRIVATE
  hip::host
  migraphx::c
  -Wl,--whole-archive
  triton-backend-utils
  triton-common-async-work-queue
  kernel_library_new
  -Wl,--no-whole-archive
  triton-common-logging
)

# 去除 lib 前缀，保持输出为 libtriton_migraphx.so
set_target_properties(triton_migraphx PROPERTIES
  OUTPUT_NAME "triton_migraphx"
  PREFIX "lib"
  POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# 5. 安装规则 (符合 Triton 后端目录结构)
# ==============================================================================
# 安装目录结构: <install_prefix>/backends/migraphx/libtriton_migraphx.so
install(TARGETS triton_migraphx
  DESTINATION backends/migraphx
)
