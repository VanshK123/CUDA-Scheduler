# Dynamic AI-Driven CUDA Kernel Scheduler

An intelligent CUDA kernel scheduler that uses machine learning to optimize GPU resource allocation and improve performance across mixed workloads.

## Project Overview

This project implements an AI-driven CUDA kernel scheduler that dynamically optimizes GPU resource allocation based on:
- Kernel execution time predictions using ML models
- Workload priority classification
- Real-time performance monitoring
- Adaptive scheduling algorithms

## Architecture

### Core Components

1. **Telemetry Collection Module** (`src/telemetry/`)
   - CUDA runtime call interception
   - Kernel execution profiling
   - Performance metrics collection

2. **AI Prediction Engine** (`src/ai/`)
   - Feature extraction from kernel launches
   - ML model inference (XGBoost/Transformer)
   - Prediction caching and optimization

3. **Adaptive Scheduler** (`src/scheduler/`)
   - Priority-based kernel queuing
   - Dynamic resource allocation
   - SLA-aware scheduling

4. **Performance Monitoring** (`src/monitoring/`)
   - Real-time metrics collection
   - Performance regression detection
   - Resource utilization tracking

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
- [x] Project structure setup
- [ ] Basic telemetry collection
- [ ] CUDA runtime hook implementation
- [ ] Performance monitoring framework

### Phase 2: AI Model Development (Weeks 5-8)
- [ ] Dataset generation and preprocessing
- [ ] Feature engineering pipeline
- [ ] Model training and validation
- [ ] C++ model integration

### Phase 3: Scheduler Implementation (Weeks 9-12)
- [ ] Core scheduling algorithms
- [ ] Priority queue implementation
- [ ] CUDA stream management
- [ ] Integration testing

### Phase 4: Testing and Optimization (Weeks 13-16)
- [ ] Benchmark suite development
- [ ] Performance validation
- [ ] Production readiness
- [ ] Documentation

## Performance Targets

| Metric | Baseline | Target Improvement |
|--------|----------|-------------------|
| Inference Latency (P95) | Current P95 | 20-30% reduction |
| Training Throughput | Current samples/sec | 10-15% increase |
| Queue Wait Time | Current average | 40-50% reduction |
| Resource Utilization | Current SM usage | 15-20% improvement |

## Building the Project

### Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.16+
- Python 3.8+ (for ML model training)
- XGBoost, NumPy, scikit-learn
- ONNX Runtime (for C++ model inference)

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd CUDA-Scheduler

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)

# Install dependencies for ML components
pip install -r requirements.txt
```

### Running Tests

```bash
# Run unit tests
make test

# Run performance benchmarks
./benchmarks/run_benchmarks.sh

# Run integration tests
./tests/run_integration_tests.sh
```

## Usage

### Basic Integration

```cpp
#include "cuda_scheduler/scheduler.hpp"

// Initialize the scheduler
auto scheduler = CUDAScheduler::create();

// Enable AI-driven scheduling
scheduler.enableAIScheduling(true);

// Launch kernels normally - scheduler intercepts automatically
cudaLaunchKernel(kernel_func, grid, block, shared_mem, stream, args...);
```

### Advanced Configuration

```cpp
// Configure scheduler for specific workload types
SchedulerConfig config;
config.latency_critical_threshold = 10.0f;  // ms
config.throughput_optimization = true;
config.enable_preemption = true;

scheduler.configure(config);
```

## Monitoring and Debugging

### Performance Metrics

The scheduler provides real-time metrics through the monitoring interface:

```cpp
auto metrics = scheduler.getMetrics();
std::cout << "Average queue wait time: " << metrics.avg_queue_wait_ms << "ms\n";
std::cout << "SM utilization: " << metrics.sm_utilization_percent << "%\n";
std::cout << "Prediction accuracy: " << metrics.prediction_accuracy << "%\n";
```

### Logging

Enable detailed logging for debugging:

```cpp
scheduler.setLogLevel(LogLevel::DEBUG);
scheduler.enableProfiling(true);
```