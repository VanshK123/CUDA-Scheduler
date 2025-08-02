#include "cuda_scheduler/scheduler.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Simple CUDA kernel for demonstration
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMultiply(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    std::cout << "CUDA Scheduler Basic Usage Example" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Initialize CUDA
    cudaError_t cuda_error = cudaSetDevice(0);
    if (cuda_error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cuda_error) << std::endl;
        return -1;
    }
    
    // Create and initialize the scheduler
    auto scheduler = cuda_scheduler::CUDAScheduler::create();
    
    cuda_scheduler::SchedulerConfig config;
    config.enable_ai_scheduling = true;
    config.latency_critical_threshold_ms = 5.0f;
    config.throughput_optimization = true;
    config.max_queue_size = 1000;
    config.model_path = "models/kernel_predictor.onnx";  // Will use stub if not found
    
    if (!scheduler->initialize(config)) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return -1;
    }
    
    // Enable detailed logging
    scheduler->setLogLevel(cuda_scheduler::LogLevel::INFO);
    scheduler->enableProfiling(true);
    
    std::cout << "\nScheduler initialized successfully!" << std::endl;
    
    // Allocate device memory
    const int vector_size = 1000000;
    const int matrix_size = 512;
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vector_size * sizeof(float));
    cudaMalloc(&d_b, vector_size * sizeof(float));
    cudaMalloc(&d_c, vector_size * sizeof(float));
    
    // Launch kernels with different characteristics
    std::cout << "\nLaunching kernels with AI-driven scheduling..." << std::endl;
    
    // Kernel 1: Vector addition (latency-critical)
    dim3 grid1((vector_size + 255) / 256, 1, 1);
    dim3 block1(256, 1, 1);
    
    cuda_scheduler::KernelLaunchParams params1;
    params1.func = (void*)vectorAdd;
    params1.grid_dim = grid1;
    params1.block_dim = block1;
    params1.shared_mem_size = 0;
    params1.stream = 0;
    params1.args = (void**)&d_a;
    params1.kernel_id = cuda_scheduler::utils::generateKernelId();
    params1.launch_time = cuda_scheduler::utils::getCurrentTime();
    
    scheduler->scheduleKernel(params1);
    
    // Kernel 2: Matrix multiplication (throughput-optimized)
    dim3 grid2((matrix_size + 15) / 16, (matrix_size + 15) / 16, 1);
    dim3 block2(16, 16, 1);
    
    cuda_scheduler::KernelLaunchParams params2;
    params2.func = (void*)matrixMultiply;
    params2.grid_dim = grid2;
    params2.block_dim = block2;
    params2.shared_mem_size = 0;
    params2.stream = 0;
    params2.args = (void**)&d_a;
    params2.kernel_id = cuda_scheduler::utils::generateKernelId();
    params2.launch_time = cuda_scheduler::utils::getCurrentTime();
    
    scheduler->scheduleKernel(params2);
    
    // Wait for kernels to complete
    cudaDeviceSynchronize();
    
    // Get performance metrics
    auto metrics = scheduler->getMetrics();
    
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "====================" << std::endl;
    cuda_scheduler::utils::printPerformanceMetrics(metrics);
    
    // Simulate more kernel launches
    std::cout << "\nSimulating mixed workload..." << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        // Alternate between latency-critical and throughput kernels
        if (i % 2 == 0) {
            // Latency-critical kernel
            cuda_scheduler::KernelLaunchParams params;
            params.func = (void*)vectorAdd;
            params.grid_dim = grid1;
            params.block_dim = block1;
            params.shared_mem_size = 0;
            params.stream = 0;
            params.args = (void**)&d_a;
            params.kernel_id = cuda_scheduler::utils::generateKernelId();
            params.launch_time = cuda_scheduler::utils::getCurrentTime();
            
            scheduler->scheduleKernel(params);
        } else {
            // Throughput kernel
            cuda_scheduler::KernelLaunchParams params;
            params.func = (void*)matrixMultiply;
            params.grid_dim = grid2;
            params.block_dim = block2;
            params.shared_mem_size = 0;
            params.stream = 0;
            params.args = (void**)&d_a;
            params.kernel_id = cuda_scheduler::utils::generateKernelId();
            params.launch_time = cuda_scheduler::utils::getCurrentTime();
            
            scheduler->scheduleKernel(params);
        }
        
        // Small delay to simulate real workload
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Get final metrics
    auto final_metrics = scheduler->getMetrics();
    
    std::cout << "\nFinal Performance Results:" << std::endl;
    std::cout << "=========================" << std::endl;
    cuda_scheduler::utils::printPerformanceMetrics(final_metrics);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    scheduler->shutdown();
    
    std::cout << "\nExample completed successfully!" << std::endl;
    
    return 0;
} 