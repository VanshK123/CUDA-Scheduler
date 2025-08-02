# Tests directory CMakeLists.txt

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(WARNING "Google Test not found. Tests will be disabled.")
    return()
endif()

enable_testing()

# Main scheduler tests
add_executable(scheduler_tests 
    test_scheduler.cpp
)

set_target_properties(scheduler_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(scheduler_tests 
    cuda_scheduler 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(scheduler_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME SchedulerTests COMMAND scheduler_tests)

# AI Predictor tests
add_executable(ai_predictor_tests 
    test_ai_predictor.cpp
)

set_target_properties(ai_predictor_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(ai_predictor_tests 
    cuda_scheduler 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(ai_predictor_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME AIPredictorTests COMMAND ai_predictor_tests)

# Multi-GPU Scheduler tests
add_executable(multi_gpu_scheduler_tests 
    test_multi_gpu_scheduler.cpp
)

set_target_properties(multi_gpu_scheduler_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(multi_gpu_scheduler_tests 
    cuda_scheduler 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(multi_gpu_scheduler_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME MultiGPUSchedulerTests COMMAND multi_gpu_scheduler_tests)

# Preemption Manager tests
add_executable(preemption_manager_tests 
    test_preemption_manager.cpp
)

set_target_properties(preemption_manager_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(preemption_manager_tests 
    cuda_scheduler 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(preemption_manager_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME PreemptionManagerTests COMMAND preemption_manager_tests)

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

# Integration tests
add_executable(integration_tests 
    test_integration.cpp
)

set_target_properties(integration_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(integration_tests 
    cuda_scheduler 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME IntegrationTests COMMAND integration_tests)

# Install test executables
install(TARGETS 
    scheduler_tests
    ai_predictor_tests
    multi_gpu_scheduler_tests
    preemption_manager_tests
    performance_tests
    integration_tests
    RUNTIME DESTINATION bin/tests
) 