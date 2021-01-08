cmake_minimum_required(VERSION 3.17)
project(futures)

set(CMAKE_CXX_STANDARD 17)


add_library(futures src/futures.cpp include/futures/utilities.h include/futures/completion-queue.h include/futures/traits.h)
target_include_directories(futures PUBLIC include)


set(FUTURES_GTEST_DIRECTORY "gtest")
option(FUTURES_BUILD_TESTS "Build tests for futures using GTest" ON)
if (FUTURES_BUILD_TESTS)
    enable_testing()
    set(BUILD_GMOCK OFF)
    add_subdirectory(${FUTURES_GTEST_DIRECTORY})

    add_executable(futures-test main.cpp test/static-tests.cpp test/test-helper.cpp test/test-helper.h arangodb.h)
    target_compile_options(futures-test PRIVATE "-fsanitize=address")
    target_link_libraries(futures-test PRIVATE asan)
    target_link_libraries(futures-test PRIVATE gtest)

    target_link_libraries(futures-test PRIVATE futures)

    add_test(NAME future-test COMMAND futures-test)
endif()
