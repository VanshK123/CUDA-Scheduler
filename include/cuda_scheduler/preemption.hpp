#pragma once

#include "scheduler.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <thread>

namespace cuda_scheduler {

/**
 * @brief Preemption strategy types
 */
enum class PreemptionStrategy {
    NONE = 0,              // No preemption
    COOPERATIVE = 1,       // Cooperative preemption (yield points)
    PREEMPTIVE = 2,        // Preemptive preemption (force stop)
    ADAPTIVE = 3,          // Adaptive based on workload
    TIME_SLICE = 4         // Time-slice based preemption
};

/**
 * @brief Preemption request structure
 */
struct PreemptionRequest {
    uint64_t kernel_id;
    Priority new_priority;
    std::chrono::high_resolution_clock::time_point request_time;
    float urgency_score;  // 0.0 to 1.0
    
    PreemptionRequest() : kernel_id(0), new_priority(Priority::NORMAL), urgency_score(0.0f) {}
    PreemptionRequest(uint64_t id, Priority priority, float urgency) 
        : kernel_id(id), new_priority(priority), urgency_score(urgency) {
        request_time = std::chrono::high_resolution_clock::now();
    }
};

/**
 * @brief Preemption context for tracking kernel state
 */
struct PreemptionContext {
    uint64_t kernel_id;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point last_checkpoint;
    Priority original_priority;
    Priority current_priority;
    bool is_preemptible;
    bool is_preempted;
    size_t checkpoint_count;
    float execution_progress;  // 0.0 to 1.0
    
    // CUDA context and kernel information
    CUcontext saved_context;
    void* kernel_function;
    dim3 current_grid_dim;
    dim3 current_block_dim;
    size_t current_shared_mem;
    std::vector<void*> current_kernel_args;
    cudaStream_t current_stream;
    
    // Saved kernel parameters
    dim3 saved_grid_dim;
    dim3 saved_block_dim;
    size_t saved_shared_mem;
    std::vector<void*> saved_kernel_args;
    cudaStream_t saved_stream;
    bool saved_kernel_params;
    
    // Memory regions
    struct MemoryRegion {
        void* device_ptr;
        size_t size;
        bool is_critical;
    };
    std::vector<MemoryRegion> memory_regions;
    
    struct SavedMemoryRegion {
        void* device_ptr;
        size_t size;
        bool needs_restore;
        std::vector<uint8_t> host_backup;
    };
    std::vector<SavedMemoryRegion> saved_memory_regions;
    
    // Progress tracking
    float current_progress;
    float saved_progress;
    std::chrono::steady_clock::duration current_execution_time;
    std::chrono::steady_clock::duration saved_execution_time;
    
    // Preemption timing
    std::chrono::steady_clock::time_point preemption_start_time;
    std::chrono::steady_clock::time_point preemption_end_time;
    std::chrono::steady_clock::duration total_preemption_time;
    
    // Statistics
    size_t save_count;
    size_t restore_count;
    std::chrono::steady_clock::time_point last_save_time;
    std::chrono::steady_clock::time_point last_restore_time;
    
    PreemptionContext() : kernel_id(0), original_priority(Priority::NORMAL), 
                         current_priority(Priority::NORMAL), is_preemptible(false),
                         is_preempted(false), checkpoint_count(0), execution_progress(0.0f),
                         saved_context(nullptr), kernel_function(nullptr), current_shared_mem(0),
                         current_stream(nullptr), saved_shared_mem(0), saved_stream(nullptr),
                         saved_kernel_params(false), current_progress(0.0f), saved_progress(0.0f),
                         total_preemption_time(std::chrono::steady_clock::duration::zero()),
                         save_count(0), restore_count(0) {}
};

/**
 * @brief Advanced preemption system for CUDA kernels
 * 
 * This class provides:
 * 1. Cooperative preemption with yield points
 * 2. Preemptive preemption for critical workloads
 * 3. Adaptive preemption based on workload characteristics
 * 4. Time-slice based preemption for fair scheduling
 * 5. Preemption context tracking and restoration
 */
class PreemptionManager {
public:
    /**
     * @brief Constructor
     */
    PreemptionManager();
    
    /**
     * @brief Destructor
     */
    ~PreemptionManager();
    
    /**
     * @brief Initialize preemption manager
     * @param strategy Preemption strategy to use
     * @param time_slice_ms Time slice duration for time-slice strategy
     * @return true if initialization successful
     */
    bool initialize(PreemptionStrategy strategy = PreemptionStrategy::COOPERATIVE, 
                   uint32_t time_slice_ms = 10);
    
    /**
     * @brief Register kernel for preemption tracking
     * @param kernel_id Kernel identifier
     * @param priority Kernel priority
     * @param is_preemptible Whether kernel supports preemption
     * @return true if registration successful
     */
    bool registerKernel(uint64_t kernel_id, Priority priority, bool is_preemptible = true);
    
    /**
     * @brief Unregister kernel from preemption tracking
     * @param kernel_id Kernel identifier
     */
    void unregisterKernel(uint64_t kernel_id);
    
    /**
     * @brief Request preemption of a kernel
     * @param request Preemption request
     * @return true if preemption request accepted
     */
    bool requestPreemption(const PreemptionRequest& request);
    
    /**
     * @brief Check if kernel should be preempted
     * @param kernel_id Kernel identifier
     * @return true if kernel should be preempted
     */
    bool shouldPreempt(uint64_t kernel_id);
    
    /**
     * @brief Execute preemption for a kernel
     * @param kernel_id Kernel identifier
     * @return CUDA result code
     */
    CUresult executePreemption(uint64_t kernel_id);
    
    /**
     * @brief Resume a preempted kernel
     * @param kernel_id Kernel identifier
     * @return CUDA result code
     */
    CUresult resumeKernel(uint64_t kernel_id);
    
    /**
     * @brief Update kernel execution progress
     * @param kernel_id Kernel identifier
     * @param progress Execution progress (0.0 to 1.0)
     */
    void updateProgress(uint64_t kernel_id, float progress);
    
    /**
     * @brief Set preemption strategy
     * @param strategy New preemption strategy
     */
    void setStrategy(PreemptionStrategy strategy);
    
    /**
     * @brief Get preemption statistics
     * @return Preemption statistics
     */
    struct PreemptionStats {
        size_t total_preemptions;
        size_t successful_preemptions;
        size_t failed_preemptions;
        float avg_preemption_time_ms;
        size_t active_kernels;
        size_t preempted_kernels;
    };
    
    PreemptionStats getStats() const;
    
    /**
     * @brief Enable or disable preemption for a kernel
     * @param kernel_id Kernel identifier
     * @param enabled Whether to enable preemption
     */
    void setKernelPreemptible(uint64_t kernel_id, bool enabled);
    
    /**
     * @brief Get preemption context for a kernel
     * @param kernel_id Kernel identifier
     * @return Preemption context
     */
    PreemptionContext getKernelContext(uint64_t kernel_id) const;
    
    /**
     * @brief Shutdown preemption manager
     */
    void shutdown();

private:
    // Preemption state
    PreemptionStrategy strategy_;
    uint32_t time_slice_ms_;
    std::atomic<bool> shutdown_requested_{false};
    
    // Kernel tracking
    mutable std::mutex context_mutex_;
    std::unordered_map<uint64_t, PreemptionContext> kernel_contexts_;
    std::queue<PreemptionRequest> preemption_queue_;
    
    // Time-slice tracking
    std::unordered_map<uint64_t, std::chrono::high_resolution_clock::time_point> kernel_start_times_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PreemptionStats stats_;
    
    // Background monitoring thread
    std::thread monitoring_thread_;
    
    // Private methods
    void monitoringLoop();
    bool canPreempt(uint64_t kernel_id) const;
    bool shouldPreempt_Cooperative(uint64_t kernel_id);
    bool shouldPreempt_Preemptive(uint64_t kernel_id);
    bool shouldPreempt_Adaptive(uint64_t kernel_id);
    bool shouldPreempt_TimeSlice(uint64_t kernel_id);
    void updatePreemptionStats(uint64_t kernel_id, bool successful);
    float calculateUrgencyScore(const PreemptionRequest& request) const;
    void cleanupExpiredContexts();
    bool isKernelExpired(uint64_t kernel_id) const;
    void restoreKernelContext(uint64_t kernel_id);
    void saveKernelContext(uint64_t kernel_id);
    std::string getCUDAErrorString(CUresult result);
};

} // namespace cuda_scheduler 