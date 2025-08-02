# Benchmarks directory CMakeLists.txt

# Mixed workload benchmark
add_executable(mixed_workload_benchmark mixed_workload_benchmark.cpp)

set_target_properties(mixed_workload_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(mixed_workload_benchmark cuda_scheduler)

target_include_directories(mixed_workload_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Latency benchmark
add_executable(latency_benchmark latency_benchmark.cpp)

set_target_properties(latency_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(latency_benchmark cuda_scheduler)

target_include_directories(latency_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Throughput benchmark
add_executable(throughput_benchmark throughput_benchmark.cpp)

set_target_properties(throughput_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(throughput_benchmark cuda_scheduler)

target_include_directories(throughput_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install benchmarks
install(TARGETS 
    mixed_workload_benchmark
    latency_benchmark
    throughput_benchmark
    RUNTIME DESTINATION bin/benchmarks
) 