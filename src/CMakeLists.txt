# Source directory CMakeLists.txt

# Find CUDA
find_package(CUDA REQUIRED)

# Find ONNX Runtime
find_package(ONNXRuntime QUIET)
if(NOT ONNXRuntime_FOUND)
    message(WARNING "ONNX Runtime not found. AI features will be disabled.")
    add_definitions(-DDISABLE_AI_FEATURES)
endif()

# Find NVML for GPU monitoring
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NVML nvidia-ml)
    if(NVML_FOUND)
        add_definitions(-DHAVE_NVML)
    endif()
endif()

# Create the main library
add_library(cuda_scheduler SHARED
    scheduler.cpp
    telemetry.cpp
    ai_predictor.cpp
    priority_queue.cpp
    performance_monitor.cpp
    multi_gpu_scheduler.cpp
    preemption_manager.cpp
    onnx_model.cpp
    utils.cpp
)

# Set target properties
set_target_properties(cuda_scheduler PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(cuda_scheduler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_scheduler
    ${CUDA_LIBRARIES}
    Threads::Threads
    cupti
)

# Add ONNX Runtime if available
if(ONNXRuntime_FOUND)
    target_link_libraries(cuda_scheduler ${ONNXRuntime_LIBRARIES})
    target_include_directories(cuda_scheduler PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
endif()

# Add NVML if available
if(NVML_FOUND)
    target_link_libraries(cuda_scheduler ${NVML_LIBRARIES})
    target_include_directories(cuda_scheduler PRIVATE ${NVML_INCLUDE_DIRS})
endif()

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda_scheduler OpenMP::OpenMP_CXX)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cuda_scheduler PRIVATE -Wall -Wextra -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cuda_scheduler PRIVATE -Wall -Wextra -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(cuda_scheduler PRIVATE /W3 /O2)
endif()

# Install rules
install(TARGETS cuda_scheduler
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
) 