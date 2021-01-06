cmake_minimum_required(VERSION 3.17)
project(futures)

set(CMAKE_CXX_STANDARD 17)

add_library(futures src/futures.cpp include/futures/utilities.h)
target_include_directories(futures PUBLIC include)


add_executable(futures-test main.cpp)
target_compile_options(futures-test PRIVATE "-fsanitize=address")
target_link_libraries(futures-test PRIVATE asan)

target_link_libraries(futures-test PRIVATE futures)
