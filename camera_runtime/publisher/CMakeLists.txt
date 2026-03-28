cmake_minimum_required(VERSION 3.16)
project(dabai_frame_publisher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORBBECSDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../vendor/OrbbecSDK")
set(ORBBECSDK_INCLUDE_DIR "${ORBBECSDK_ROOT}/include")
set(ORBBECSDK_LIB_DIR "${ORBBECSDK_ROOT}/lib/arm64")
set(ORBBECSDK_LIB_FILE "${ORBBECSDK_LIB_DIR}/libOrbbecSDK.so")

if(NOT EXISTS "${ORBBECSDK_INCLUDE_DIR}")
    message(FATAL_ERROR "Orbbec SDK headers not found: ${ORBBECSDK_INCLUDE_DIR}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

set(orbbec_target "OrbbecSDK::OrbbecSDK")
set(orbbec_include_dirs "${ORBBECSDK_INCLUDE_DIR}")
set(orbbec_lib_dir "${ORBBECSDK_LIB_DIR}")

if(EXISTS "${ORBBECSDK_ROOT}/OrbbecSDKConfig.cmake" AND EXISTS "${ORBBECSDK_ROOT}/cmake/DetermineHostPlatform.cmake")
    find_package(OrbbecSDK REQUIRED PATHS "${ORBBECSDK_ROOT}" NO_DEFAULT_PATH)
    set(orbbec_include_dirs "${OrbbecSDK_INCLUDE_DIRS}")
    if(DEFINED OrbbecSDK_LIBS_DIR)
        set(orbbec_lib_dir "${OrbbecSDK_LIBS_DIR}")
    endif()
else()
    if(NOT EXISTS "${ORBBECSDK_LIB_FILE}")
        message(FATAL_ERROR "Orbbec SDK shared library not found: ${ORBBECSDK_LIB_FILE}")
    endif()
    add_library(${orbbec_target} SHARED IMPORTED GLOBAL)
    set_target_properties(${orbbec_target} PROPERTIES
        IMPORTED_LOCATION "${ORBBECSDK_LIB_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORBBECSDK_INCLUDE_DIR}"
    )
endif()

add_executable(dabai_frame_publisher
    src/main.cpp
)

target_include_directories(dabai_frame_publisher PRIVATE
    ${orbbec_include_dirs}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(dabai_frame_publisher PRIVATE
    ${orbbec_target}
    PkgConfig::ZMQ
    ${OpenCV_LIBS}
)

if(orbbec_lib_dir)
    set_target_properties(dabai_frame_publisher PROPERTIES
        BUILD_RPATH "${orbbec_lib_dir}"
        INSTALL_RPATH "${orbbec_lib_dir}"
    )
endif()
