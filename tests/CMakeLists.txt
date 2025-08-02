# Tests directory CMakeLists.txt

# Find GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(WARNING "Google Test not found. Tests will be disabled.")
    return()
endif()

# Enable testing
enable_testing()

# Unit tests
add_executable(scheduler_tests
    test_scheduler.cpp
    test_priority_queue.cpp
    test_telemetry.cpp
    test_ai_predictor.cpp
)

# Set properties
set_target_properties(scheduler_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link libraries
target_link_libraries(scheduler_tests
    cuda_scheduler
    GTest::gtest
    GTest::gtest_main
)

# Include directories
target_include_directories(scheduler_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Add test
add_test(NAME SchedulerTests COMMAND scheduler_tests)

# Performance tests
add_executable(performance_tests
    test_performance.cpp
)

set_target_properties(performance_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(performance_tests
    cuda_scheduler
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(performance_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME PerformanceTests COMMAND performance_tests) 