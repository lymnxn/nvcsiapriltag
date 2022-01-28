cmake_minimum_required(VERSION 3.2)
project(nvcsiapriltag)

# -std=gnu++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
message("-- System version: ${CMAKE_SYSTEM_VERSION}")

configure_file(inc/projectConfig.h.in inc/projectConfig.h)

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda")
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}; 
    -O3 
    -gencode arch=compute_37,code=sm_37
    -gencode arch=compute_53,code=sm_53
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)
# add_subdirectory(utils)
if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; 
        -gencode arch=compute_72,code=sm_72
    )
endif()

find_package(OpenCV REQUIRED)

if( NOT OpenCV_FOUND )
    message("-- didn't find OpenCV on system, disabling OpenCV")
else()
    message("-- OpenCV version:  " ${OpenCV_VERSION})

    if( ${OpenCV_VERSION_MAJOR} LESS 3 )
        message("-- OpenCV version less than 3.0, disabling OpenCV")
    else()
        message("-- OpenCV version >= 3.0.0, enabling OpenCV")
        set(HAS_OPENCV 1)
        add_definitions(-DHAS_OPENCV)
    endif()	
endif()

# set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
# set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
# file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
# set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
# file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

find_package(catkin REQUIRED COMPONENTS roscpp cpp_common message_generation std_msgs sensor_msgs geometry_msgs)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(vpi REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(jetson-utils REQUIRED)

if(NOT jetson-utils_FOUND)
    message("do not find jetson-utils")
else()
    message("-- jetson-utils path: ${jetson-utils_INCLUDE_DIRS}")
    message("-- jetson-utils version: ${jetson-utils_VERSION}")
    message("-- jetson-utils library: ${jetson-utils_LIBRARY}")
    message("-- jetson-utils dir: ${jetson-utils_DIR}")
    include_directories(/usr/local/include/jetson-utils)
    include_directories(${jetson-utils_INCLUDE_DIRS})
endif()
# find_package(jetson-inference)



message("--boost include path: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})


add_message_files(
    FILES
    AprilTagDetection.msg
    AprilTagDetectionArray.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    )

catkin_package(
    # INCLUDE_DIRS include
    # LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp message_runtime std_msgs nodelet
    # DEPENDS system_lib
    )
#  INCLUDE_DIRS include
#  LIBRARIES test
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
# include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/nvapriltags)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )
message( STATUS "CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )

add_library(nvapriltags STATIC IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_x86_64/libapril_tagging.a)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a)
endif()

# add_library()

add_executable(nvcsiapriltag src/nvcsiapriltag_main.cpp)
add_dependencies(nvcsiapriltag ${PROJECT_NAME}_gencpp)
target_link_libraries(nvcsiapriltag ${catkin_LIBRARIES} ${LOG4CXX_LIBRARIES} nvapriltags ${CUDA_LIBRARIES} vpi jetson-utils jetson-inference ${OpenCV_LIBS})
