#pragma once

#include "scheduler.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>

namespace cuda_scheduler {

/**
 * @brief Multi-GPU device information
 */
struct GPUDevice {
    int device_id;
    size_t total_memory_gb;
    int multiprocessor_count;
    int compute_capability_major;
    int compute_capability_minor;
    float current_utilization;
    size_t free_memory_mb;
    int temperature_celsius;
    bool is_available;
    
    GPUDevice() : device_id(-1), total_memory_gb(0), multiprocessor_count(0),
                  compute_capability_major(0), compute_capability_minor(0),
                  current_utilization(0.0f), free_memory_mb(0), temperature_celsius(0),
                  is_available(false) {}
};

/**
 * @brief Multi-GPU scheduling strategy
 */
enum class MultiGPUStrategy {
    ROUND_ROBIN = 0,        // Distribute kernels round-robin across GPUs
    LOAD_BALANCED = 1,      // Balance load based on GPU utilization
    MEMORY_AWARE = 2,       // Consider memory availability
    AFFINITY_BASED = 3,     // Keep related kernels on same GPU
    PERFORMANCE_OPTIMIZED = 4 // Use performance prediction for placement
};

/**
 * @brief Multi-GPU scheduler for coordinating across multiple devices
 * 
 * This class provides:
 * 1. Multi-GPU device management and monitoring
 * 2. Intelligent kernel placement strategies
 * 3. Load balancing across devices
 * 4. Memory-aware scheduling
 * 5. Performance optimization across GPU cluster
 */
class MultiGPUScheduler {
public:
    /**
     * @brief Constructor
     */
    MultiGPUScheduler();
    
    /**
     * @brief Destructor
     */
    ~MultiGPUScheduler();
    
    /**
     * @brief Initialize multi-GPU scheduler
     * @param strategy Scheduling strategy to use
     * @return true if initialization successful
     */
    bool initialize(MultiGPUStrategy strategy = MultiGPUStrategy::LOAD_BALANCED);
    
    /**
     * @brief Get available GPU devices
     * @return Vector of GPU device information
     */
    std::vector<GPUDevice> getAvailableDevices() const;
    
    /**
     * @brief Select optimal GPU for kernel execution
     * @param profile Kernel profile
     * @param predicted_time Predicted execution time
     * @return Selected GPU device ID
     */
    int selectOptimalGPU(const KernelProfile& profile, float predicted_time);
    
    /**
     * @brief Schedule kernel across multiple GPUs
     * @param params Kernel launch parameters
     * @return CUDA result code
     */
    CUresult scheduleKernelMultiGPU(const KernelLaunchParams& params);
    
    /**
     * @brief Update GPU device information
     * @param device_id GPU device ID
     * @param device_info Updated device information
     */
    void updateDeviceInfo(int device_id, const GPUDevice& device_info);
    
    /**
     * @brief Get GPU load balancing statistics
     * @return Load balancing statistics
     */
    struct LoadBalancingStats {
        std::vector<float> gpu_utilizations;
        std::vector<size_t> kernels_per_gpu;
        float load_balance_score;  // 0.0 = perfectly balanced, 1.0 = completely unbalanced
        size_t total_kernels_scheduled;
    };
    
    LoadBalancingStats getLoadBalancingStats() const;
    
    /**
     * @brief Set scheduling strategy
     * @param strategy New scheduling strategy
     */
    void setStrategy(MultiGPUStrategy strategy);
    
    /**
     * @brief Enable or disable GPU device
     * @param device_id GPU device ID
     * @param enabled Whether to enable the device
     */
    void setDeviceEnabled(int device_id, bool enabled);
    
    /**
     * @brief Get performance metrics for all GPUs
     * @return Performance metrics for each GPU
     */
    std::vector<PerformanceMetrics> getMultiGPUMetrics() const;
    
    /**
     * @brief Shutdown multi-GPU scheduler
     */
    void shutdown();

private:
    // Multi-GPU state
    std::vector<GPUDevice> gpu_devices_;
    std::vector<std::unique_ptr<CUDAScheduler>> gpu_schedulers_;
    MultiGPUStrategy strategy_;
    
    // Load balancing state
    mutable std::mutex device_mutex_;
    std::atomic<size_t> total_kernels_scheduled_{0};
    std::vector<std::atomic<size_t>> kernels_per_gpu_;
    std::vector<std::atomic<float>> gpu_utilizations_;
    
    // Performance tracking
    std::unordered_map<int, std::chrono::high_resolution_clock::time_point> last_kernel_time_;
    std::unordered_map<int, std::vector<float>> gpu_performance_history_;
    
    // Private methods
    bool initializeGPUs();
    void updateDeviceMetrics();
    int selectGPU_RoundRobin();
    int selectGPU_LoadBalanced(const KernelProfile& profile, float predicted_time);
    int selectGPU_MemoryAware(const KernelProfile& profile);
    int selectGPU_AffinityBased(const KernelProfile& profile);
    int selectGPU_PerformanceOptimized(const KernelProfile& profile, float predicted_time);
    float calculateLoadBalanceScore() const;
    void updatePerformanceHistory(int device_id, float execution_time);
    bool isDeviceOverloaded(int device_id) const;
    bool hasSufficientMemory(int device_id, const KernelProfile& profile) const;
};

} // namespace cuda_scheduler 