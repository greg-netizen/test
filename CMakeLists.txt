cmake_minimum_required(VERSION 3.28)
project(LeNET CUDA)

find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 20)

set(OpenCV_DIR /home/cristian/opencv/build)

include_directories(
        include
        media
        ${OpenCV_INCLUDE_DIRS}
        )

add_executable(LeNET src/main.cu
        include/utils.h
        src/utils.cu
        include/weights.h
)

target_link_libraries(LeNET ${OpenCV_LIBS})

set_target_properties(LeNET PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
