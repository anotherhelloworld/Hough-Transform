cmake_minimum_required(VERSION 3.6)
project(transform)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp)
add_executable(transform ${SOURCE_FILES})
add_executable(drawImage drawTestRect.cpp)

target_link_libraries(transform ${OpenCV_LIBS})
target_link_libraries(drawImage ${OpenCV_LIBS})