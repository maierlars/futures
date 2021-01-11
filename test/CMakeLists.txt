enable_testing()
set(BUILD_GMOCK OFF)


add_executable(futures-test main.cpp static-tests.cpp test-helper.cpp test-helper.h exception-tests.cpp inline-value-tests.cpp)
target_compile_options(futures-test PRIVATE "-fsanitize=address")
target_link_libraries(futures-test PRIVATE asan)
target_link_libraries(futures-test PRIVATE gtest)

target_link_libraries(futures-test PRIVATE futures)

add_test(NAME future-test COMMAND futures-test)
