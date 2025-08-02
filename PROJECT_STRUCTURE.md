# CUDA Scheduler Project Structure

This document provides a comprehensive overview of the project structure and the purpose of each component.

## Directory Structure

```
CUDA-Scheduler/
├── README.md                    # Main project documentation
├── CMakeLists.txt              # Main CMake configuration
├── requirements.txt             # Python dependencies for ML components
├── build.sh                    # Automated build script
├── PROJECT_STRUCTURE.md        # This file
│
├── include/                    # Public header files
│   └── cuda_scheduler/
│       ├── scheduler.hpp       # Main scheduler interface
│       ├── telemetry.hpp       # Telemetry collection interface
│       ├── ai_predictor.hpp    # AI prediction interface
│       ├── priority_queue.hpp  # Priority queue interface
│       ├── performance_monitor.hpp # Performance monitoring interface
│       └── onnx_model.hpp     # ONNX model wrapper interface
│
├── src/                        # Source code implementation
│   ├── CMakeLists.txt         # Source build configuration
│   ├── scheduler.cpp          # Main scheduler implementation
│   ├── telemetry.cpp          # Telemetry collection implementation
│   ├── ai_predictor.cpp       # AI prediction implementation
│   ├── priority_queue.cpp     # Priority queue implementation
│   ├── performance_monitor.cpp # Performance monitoring implementation
│   ├── onnx_model.cpp         # ONNX model wrapper implementation
│   └── utils.cpp              # Utility functions
│
├── examples/                   # Usage examples
│   ├── CMakeLists.txt         # Examples build configuration
│   └── basic_usage.cpp        # Basic usage demonstration
│
├── tests/                      # Unit tests
│   ├── CMakeLists.txt         # Tests build configuration
│   ├── test_scheduler.cpp     # Scheduler unit tests
│   ├── test_priority_queue.cpp # Priority queue tests
│   ├── test_telemetry.cpp     # Telemetry tests
│   ├── test_ai_predictor.cpp  # AI predictor tests
│   └── test_performance.cpp   # Performance tests
│
├── benchmarks/                 # Performance benchmarks
│   ├── CMakeLists.txt         # Benchmarks build configuration
│   ├── mixed_workload_benchmark.cpp # Mixed workload benchmark
│   ├── latency_benchmark.cpp  # Latency-focused benchmark
│   └── throughput_benchmark.cpp # Throughput-focused benchmark
│
├── ml/                        # Machine learning components
│   └── train_model.py         # Model training script
│
├── models/                     # Trained ML models (generated)
│   ├── kernel_predictor.pkl   # XGBoost model
│   ├── kernel_predictor_scaler.pkl # Feature scaler
│   └── kernel_predictor.onnx  # ONNX model for C++ inference
│
└── cmake/                     # CMake configuration files
    └── CUDASchedulerConfig.cmake.in # Package configuration template
```

## Component Overview

### Core Components

#### 1. Scheduler (`include/cuda_scheduler/scheduler.hpp`, `src/scheduler.cpp`)
- **Purpose**: Main interface for AI-driven CUDA kernel scheduling
- **Key Features**:
  - Kernel launch interception and scheduling
  - Priority-based queue management
  - AI prediction integration
  - Performance monitoring
  - Configuration management

#### 2. Telemetry Collector (`include/cuda_scheduler/telemetry.hpp`, `src/telemetry.cpp`)
- **Purpose**: Collect kernel execution data and GPU metrics
- **Key Features**:
  - CUDA runtime call interception
  - Kernel execution profiling
  - GPU performance metrics collection
  - Real-time data processing
  - NVML integration for hardware monitoring

#### 3. AI Predictor (`include/cuda_scheduler/ai_predictor.hpp`, `src/ai_predictor.cpp`)
- **Purpose**: Predict kernel execution times using ML models
- **Key Features**:
  - Feature extraction from kernel profiles
  - ONNX model inference
  - Prediction caching for performance
  - Hardware-aware feature engineering
  - Model accuracy tracking

#### 4. Priority Queue (`include/cuda_scheduler/priority_queue.hpp`, `src/priority_queue.cpp`)
- **Purpose**: Thread-safe priority queue for kernel task management
- **Key Features**:
  - Priority-based task ordering
  - Wait time tracking
  - Queue statistics
  - Overflow handling
  - Thread synchronization

#### 5. Performance Monitor (`include/cuda_scheduler/performance_monitor.hpp`, `src/performance_monitor.cpp`)
- **Purpose**: Real-time GPU performance monitoring and alerting
- **Key Features**:
  - GPU utilization tracking
  - Performance alert generation
  - Historical metrics storage
  - Resource utilization analysis
  - Thermal monitoring

#### 6. ONNX Model Wrapper (`include/cuda_scheduler/onnx_model.hpp`, `src/onnx_model.cpp`)
- **Purpose**: Simplified interface for ONNX Runtime model inference
- **Key Features**:
  - Model loading and validation
  - Input/output tensor management
  - Cross-platform model inference
  - Error handling and fallbacks

### Supporting Components

#### 1. Utilities (`src/utils.cpp`)
- **Purpose**: Common utility functions
- **Key Features**:
  - Time formatting and conversion
  - Memory size formatting
  - Priority and log level conversion
  - Kernel profile printing
  - Performance metrics display

#### 2. Machine Learning (`ml/train_model.py`)
- **Purpose**: Train ML models for kernel execution time prediction
- **Key Features**:
  - Synthetic data generation
  - XGBoost model training
  - Feature importance analysis
  - Model evaluation and validation
  - ONNX model export

### Build System

#### 1. Main CMakeLists.txt
- **Purpose**: Project-wide build configuration
- **Key Features**:
  - CUDA toolkit detection
  - Dependency management
  - Compiler flags and standards
  - Installation rules
  - Package configuration

#### 2. Component CMakeLists.txt Files
- **Purpose**: Component-specific build configuration
- **Key Features**:
  - Target definitions
  - Library linking
  - Include directories
  - Test integration

### Examples and Tests

#### 1. Examples (`examples/`)
- **Purpose**: Demonstrate scheduler usage
- **Key Features**:
  - Basic usage patterns
  - Configuration examples
  - Performance monitoring
  - Error handling

#### 2. Tests (`tests/`)
- **Purpose**: Unit and integration testing
- **Key Features**:
  - Component isolation testing
  - Performance regression testing
  - Error condition testing
  - Memory leak detection

#### 3. Benchmarks (`benchmarks/`)
- **Purpose**: Performance evaluation
- **Key Features**:
  - Mixed workload testing
  - Latency measurement
  - Throughput analysis
  - Resource utilization tracking

## Build Process

### Prerequisites
1. **CUDA Toolkit 11.0+**: For CUDA development
2. **CMake 3.16+**: For build system
3. **Python 3.8+**: For ML components
4. **C++17 Compiler**: For modern C++ features

### Build Steps
1. **Configure**: `cmake -B build -DCMAKE_BUILD_TYPE=Release`
2. **Build**: `make -C build -j$(nproc)`
3. **Test**: `make -C build test`
4. **Install**: `make -C build install`

### Automated Build
```bash
./build.sh  # Runs complete build process
```

## Integration Points

### 1. CUDA Runtime Integration
- Intercepts `cudaLaunchKernel` calls
- Provides priority-based scheduling
- Maintains backward compatibility

### 2. ML Model Integration
- ONNX Runtime for cross-platform inference
- Feature extraction from kernel parameters
- Prediction caching for performance

### 3. Performance Monitoring
- NVML integration for GPU metrics
- Real-time alert generation
- Historical data analysis

### 4. Thread Safety
- Lock-free data structures where possible
- Thread-safe priority queue
- Atomic operations for statistics

## Performance Targets

| Component | Target | Measurement |
|-----------|--------|-------------|
| Scheduler Overhead | < 1μs per kernel | Kernel launch timing |
| Prediction Latency | < 20μs | Model inference timing |
| Queue Wait Time | 40-50% reduction | End-to-end latency |
| Resource Utilization | 15-20% improvement | GPU utilization metrics |
| Memory Usage | < 100MB | Process memory footprint |

## Future Enhancements

### Phase 2 Features
- Multi-GPU scheduling coordination
- Online learning and model adaptation
- CUDA Graph optimization integration
- Advanced preemption strategies
