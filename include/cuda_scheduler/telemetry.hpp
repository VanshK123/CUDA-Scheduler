#pragma once

#include "scheduler.hpp"
#include <atomic>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace cuda_scheduler {

/**
 * @brief Telemetry collector for kernel execution data
 * 
 * This class is responsible for:
 * 1. Intercepting CUDA runtime calls
 * 2. Collecting kernel execution profiles
 * 3. Monitoring GPU performance metrics
 * 4. Providing real-time telemetry data
 */
class TelemetryCollector {
public:
    /**
     * @brief Constructor
     */
    TelemetryCollector();
    
    /**
     * @brief Destructor
     */
    ~TelemetryCollector();
    
    /**
     * @brief Initialize the telemetry collector
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Record a kernel launch
     * @param params Kernel launch parameters
     */
    void recordKernelLaunch(const KernelLaunchParams& params);
    
    /**
     * @brief Record kernel completion
     * @param kernel_id Kernel identifier
     * @param execution_time_ns Execution time in nanoseconds
     */
    void recordKernelCompletion(uint64_t kernel_id, uint64_t execution_time_ns);
    
    /**
     * @brief Get collected profiles
     * @return Vector of kernel profiles
     */
    std::vector<KernelProfile> exportProfiles();
    
    /**
     * @brief Get current GPU metrics
     * @return GPU metrics structure
     */
    struct GPUMetrics {
        float sm_utilization_percent;
        float memory_utilization_percent;
        float memory_bandwidth_gb_s;
        float power_usage_watts;
        int temperature_celsius;
        size_t free_memory_mb;
        size_t total_memory_mb;
    };
    
    GPUMetrics getGPUMetrics() const;
    
    /**
     * @brief Enable or disable collection
     * @param enabled Whether to enable collection
     */
    void setCollectionEnabled(bool enabled);
    
    /**
     * @brief Get collection status
     * @return true if collection is enabled
     */
    bool isCollectionEnabled() const;
    
    /**
     * @brief Clear collected data
     */
    void clearData();
    
    /**
     * @brief Shutdown the telemetry collector
     */
    void shutdown();

private:
    // Collection state
    std::atomic<bool> collection_enabled_{true};
    std::queue<KernelProfile> profile_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Background processing
    std::thread processing_thread_;
    std::atomic<bool> shutdown_requested_{false};
    
    // GPU metrics
    mutable std::mutex metrics_mutex_;
    GPUMetrics current_metrics_;
    
    // Statistics
    std::atomic<uint64_t> total_kernels_launched_{0};
    std::atomic<uint64_t> total_kernels_completed_{0};
    
    // Private methods
    void processingLoop();
    void updateGPUMetrics();
    KernelProfile createProfile(const KernelLaunchParams& params);
    void processProfile(const KernelProfile& profile);
    
    // CUDA profiling helpers
    bool initializeCUPTI();
    void cleanupCUPTI();
    
    // Memory management
    static constexpr size_t MAX_QUEUE_SIZE = 100000;
    void trimQueue();
};

} // namespace cuda_scheduler 