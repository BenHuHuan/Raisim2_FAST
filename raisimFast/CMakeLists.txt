cmake_minimum_required(VERSION 3.18)
project(raisim_fast VERSION 2.16.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(raisim CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND OR "${Eigen3_VERSION}" VERSION_LESS "3.4")
        include(FetchContent)
        FetchContent_Declare(
            eigen3_fast_headers
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_Populate(eigen3_fast_headers)
        set(EIGEN3_INCLUDE_DIRS "${eigen3_fast_headers_SOURCE_DIR}")
    endif()
endif()
if(NOT DEFINED EIGEN3_INCLUDE_DIRS AND TARGET Eigen3::Eigen)
    get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif()

add_library(raisim_fast_vec SHARED cpp/raisim_fast_vec.cpp)
target_include_directories(raisim_fast_vec PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(raisim_fast_vec PRIVATE raisim::raisim OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(raisim_fast_vec PRIVATE -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        target_compile_options(raisim_fast_vec PRIVATE -march=native)
    endif()
endif()

if(UNIX)
    if(APPLE)
        set(_rfast_origin "@loader_path")
    else()
        set(_rfast_origin "$ORIGIN")
    endif()
    set_target_properties(raisim_fast_vec PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:raisim::raisim>;$<TARGET_FILE_DIR:raisim::raisim>/openusd"
        INSTALL_RPATH "${_rfast_origin};$<TARGET_FILE_DIR:raisim::raisim>;$<TARGET_FILE_DIR:raisim::raisim>/openusd"
    )
endif()

install(TARGETS raisim_fast_vec LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
