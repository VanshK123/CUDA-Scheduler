#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace cuda_scheduler {

// Forward declarations
class TelemetryCollector;
class AIPredictor;
class PriorityQueue;
class PerformanceMonitor;
class MultiGPUScheduler;
class PreemptionManager;

/**
 * @brief Kernel launch parameters structure
 */
struct KernelLaunchParams {
    void* func;                    // Kernel function pointer
    dim3 grid_dim;                 // Grid dimensions
    dim3 block_dim;                // Block dimensions
    size_t shared_mem_size;        // Shared memory size in bytes
    cudaStream_t stream;           // CUDA stream
    void** args;                   // Kernel arguments
    uint64_t kernel_id;            // Unique kernel identifier
    std::chrono::high_resolution_clock::time_point launch_time;
    
    KernelLaunchParams() : func(nullptr), shared_mem_size(0), stream(0), args(nullptr), kernel_id(0) {}
};

/**
 * @brief Kernel execution profile
 */
struct KernelProfile {
    uint64_t kernel_id;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem_size;
    std::vector<size_t> input_shapes;
    std::string operation_type;
    uint64_t execution_time_ns;
    float sm_utilization;
    float memory_bandwidth_gb_s;
    std::chrono::high_resolution_clock::time_point completion_time;
    
    KernelProfile() : kernel_id(0), shared_mem_size(0), execution_time_ns(0), 
                     sm_utilization(0.0f), memory_bandwidth_gb_s(0.0f) {}
};

/**
 * @brief Workload priority levels
 */
enum class Priority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Scheduler configuration
 */
struct SchedulerConfig {
    bool enable_ai_scheduling = true;
    bool enable_preemption = false;
    float latency_critical_threshold_ms = 10.0f;
    bool throughput_optimization = false;
    size_t max_queue_size = 10000;
    size_t prediction_cache_size = 10000;
    std::string model_path = "models/kernel_predictor.onnx";
    
    SchedulerConfig() = default;
};

/**
 * @brief Performance metrics
 */
struct PerformanceMetrics {
    float avg_queue_wait_ms;
    float avg_execution_time_ms;
    float sm_utilization_percent;
    float memory_utilization_percent;
    float prediction_accuracy;
    size_t total_kernels_launched;
    size_t total_kernels_completed;
    std::chrono::high_resolution_clock::time_point last_update;
    
    PerformanceMetrics() : avg_queue_wait_ms(0.0f), avg_execution_time_ms(0.0f),
                          sm_utilization_percent(0.0f), memory_utilization_percent(0.0f),
                          prediction_accuracy(0.0f), total_kernels_launched(0), total_kernels_completed(0) {}
};

/**
 * @brief Log levels for debugging
 */
enum class LogLevel {
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    DEBUG = 3
};

/**
 * @brief Main CUDA Scheduler class
 * 
 * This class provides AI-driven kernel scheduling capabilities by:
 * 1. Intercepting CUDA kernel launches
 * 2. Predicting execution times using ML models
 * 3. Prioritizing kernels based on workload characteristics
 * 4. Optimizing resource allocation
 */
class CUDAScheduler {
public:
    /**
     * @brief Create a new scheduler instance
     * @return Shared pointer to scheduler instance
     */
    static std::shared_ptr<CUDAScheduler> create();
    
    /**
     * @brief Destructor
     */
    ~CUDAScheduler();
    
    /**
     * @brief Initialize the scheduler
     * @param config Scheduler configuration
     * @return true if initialization successful
     */
    bool initialize(const SchedulerConfig& config = SchedulerConfig{});
    
    /**
     * @brief Schedule a kernel launch
     * @param params Kernel launch parameters
     * @return CUDA result code
     */
    CUresult scheduleKernel(const KernelLaunchParams& params);
    
    /**
     * @brief Enable or disable AI-driven scheduling
     * @param enabled Whether to enable AI scheduling
     */
    void enableAIScheduling(bool enabled);
    
    /**
     * @brief Configure the scheduler
     * @param config New configuration
     */
    void configure(const SchedulerConfig& config);
    
    /**
     * @brief Get current performance metrics
     * @return Performance metrics structure
     */
    PerformanceMetrics getMetrics() const;
    
    /**
     * @brief Set logging level
     * @param level Log level
     */
    void setLogLevel(LogLevel level);
    
    /**
     * @brief Enable or disable profiling
     * @param enabled Whether to enable profiling
     */
    void enableProfiling(bool enabled);
    
    /**
     * @brief Shutdown the scheduler
     */
    void shutdown();

private:
    CUDAScheduler();
    
    // Core components
    std::unique_ptr<TelemetryCollector> telemetry_;
    std::unique_ptr<AIPredictor> predictor_;
    std::unique_ptr<PriorityQueue> priority_queue_;
    std::unique_ptr<PerformanceMonitor> monitor_;
    std::unique_ptr<MultiGPUScheduler> multi_gpu_scheduler_;
    std::unique_ptr<PreemptionManager> preemption_manager_;
    
    // Configuration and state
    SchedulerConfig config_;
    bool ai_scheduling_enabled_;
    bool profiling_enabled_;
    LogLevel log_level_;
    
    // Statistics
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    
    // Private methods
    Priority calculatePriority(const KernelProfile& profile, float predicted_time);
    CUresult dispatchNextKernel();
    void updateMetrics(const KernelProfile& profile);
    void logMessage(LogLevel level, const std::string& message) const;
};

} // namespace cuda_scheduler 