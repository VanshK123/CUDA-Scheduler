# Examples directory CMakeLists.txt

# Basic usage example
add_executable(basic_usage basic_usage.cpp)

# Set CUDA properties for the example
set_target_properties(basic_usage PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link with the scheduler library
target_link_libraries(basic_usage cuda_scheduler)

# Include directories
target_include_directories(basic_usage PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install the example
install(TARGETS basic_usage
    RUNTIME DESTINATION bin/examples
)

# Advanced features example
add_executable(advanced_features advanced_features.cpp)

# Set CUDA properties for the advanced example
set_target_properties(advanced_features PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link with the scheduler library
target_link_libraries(advanced_features cuda_scheduler)

# Include directories
target_include_directories(advanced_features PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install the advanced example
install(TARGETS advanced_features
    RUNTIME DESTINATION bin/examples
) 