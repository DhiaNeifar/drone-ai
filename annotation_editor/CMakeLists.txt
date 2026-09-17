cmake_minimum_required(VERSION 3.16)
project(drone_annotation_editor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(drone_annotation_editor main.cpp)
target_link_libraries(drone_annotation_editor PRIVATE ${OpenCV_LIBS})
target_include_directories(drone_annotation_editor PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
  target_compile_options(drone_annotation_editor PRIVATE /W4)
else()
  target_compile_options(drone_annotation_editor PRIVATE -Wall -Wextra -Wpedantic)
endif()
