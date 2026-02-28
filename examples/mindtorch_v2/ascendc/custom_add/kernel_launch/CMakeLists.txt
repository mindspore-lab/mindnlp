cmake_minimum_required(VERSION 3.16)
project(add_custom_kernel)

set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim/cpu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE STRING "CANN package path")
set(ASCEND_KERNEL_LAUNCH_ONLY ON CACHE BOOL "Kernel launch only (no op_host)")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found in CANN package")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../add_custom.cpp)

ascendc_library(add_custom_kernels SHARED
    ${KERNEL_FILES}
)
