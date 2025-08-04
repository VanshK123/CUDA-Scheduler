#include "cuda_scheduler/preemption.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace cuda_scheduler {

PreemptionManager::PreemptionManager() 
    : strategy_(PreemptionStrategy::COOPERATIVE)
    , time_slice_ms_(10) {
}

PreemptionManager::~PreemptionManager() {
    shutdown();
}

bool PreemptionManager::initialize(PreemptionStrategy strategy, uint32_t time_slice_ms) {
    try {
        strategy_ = strategy;
        time_slice_ms_ = time_slice_ms;
        
        // Initialize statistics
        stats_.total_preemptions = 0;
        stats_.successful_preemptions = 0;
        stats_.failed_preemptions = 0;
        stats_.avg_preemption_time_ms = 0.0f;
        stats_.active_kernels = 0;
        stats_.preempted_kernels = 0;
        
        // Start monitoring thread
        shutdown_requested_ = false;
        monitoring_thread_ = std::thread(&PreemptionManager::monitoringLoop, this);
        
        std::cout << "Preemption Manager initialized with strategy " << static_cast<int>(strategy) << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Preemption initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool PreemptionManager::registerKernel(uint64_t kernel_id, Priority priority, bool is_preemptible) {
    try {
        std::lock_guard<std::mutex> lock(context_mutex_);
        
        PreemptionContext context;
        context.kernel_id = kernel_id;
        context.start_time = std::chrono::high_resolution_clock::now();
        context.last_checkpoint = context.start_time;
        context.original_priority = priority;
        context.current_priority = priority;
        context.is_preemptible = is_preemptible;
        context.is_preempted = false;
        context.checkpoint_count = 0;
        context.execution_progress = 0.0f;
        
        kernel_contexts_[kernel_id] = context;
        kernel_start_times_[kernel_id] = context.start_time;
        
        stats_.active_kernels++;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to register kernel " << kernel_id << ": " << e.what() << std::endl;
        return false;
    }
}

void PreemptionManager::unregisterKernel(uint64_t kernel_id) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it != kernel_contexts_.end()) {
        if (context_it->second.is_preempted) {
            stats_.preempted_kernels--;
        }
        stats_.active_kernels--;
        kernel_contexts_.erase(context_it);
    }
    
    kernel_start_times_.erase(kernel_id);
    gpu_performance_history_.erase(kernel_id);
}

bool PreemptionManager::requestPreemption(const PreemptionRequest& request) {
    try {
        std::lock_guard<std::mutex> lock(context_mutex_);
        
        auto context_it = kernel_contexts_.find(request.kernel_id);
        if (context_it == kernel_contexts_.end()) {
            return false;  // Kernel not registered
        }
        
        if (!context_it->second.is_preemptible) {
            return false;  // Kernel not preemptible
        }
        
        // Add to preemption queue
        preemption_queue_.push(request);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to request preemption: " << e.what() << std::endl;
        return false;
    }
}

bool PreemptionManager::shouldPreempt(uint64_t kernel_id) {
    if (strategy_ == PreemptionStrategy::NONE) {
        return false;
    }
    
    switch (strategy_) {
        case PreemptionStrategy::COOPERATIVE:
            return shouldPreempt_Cooperative(kernel_id);
        case PreemptionStrategy::PREEMPTIVE:
            return shouldPreempt_Preemptive(kernel_id);
        case PreemptionStrategy::ADAPTIVE:
            return shouldPreempt_Adaptive(kernel_id);
        case PreemptionStrategy::TIME_SLICE:
            return shouldPreempt_TimeSlice(kernel_id);
        default:
            return false;
    }
}

CUresult PreemptionManager::executePreemption(uint64_t kernel_id) {
    try {
        std::lock_guard<std::mutex> lock(context_mutex_);
        
        auto context_it = kernel_contexts_.find(kernel_id);
        if (context_it == kernel_contexts_.end()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        PreemptionContext& context = context_it->second;
        
        if (!context.is_preemptible || context.is_preempted) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        // Save kernel context
        saveKernelContext(kernel_id);
        
        // Mark as preempted
        context.is_preempted = true;
        context.last_checkpoint = std::chrono::high_resolution_clock::now();
        
        // Update statistics
        stats_.total_preemptions++;
        stats_.preempted_kernels++;
        
        std::cout << "Preempted kernel " << kernel_id << " at progress " 
                  << context.execution_progress * 100.0f << "%" << std::endl;
        
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "Preemption execution failed: " << e.what() << std::endl;
        return CUDA_ERROR_LAUNCH_FAILED;
    }
}

CUresult PreemptionManager::resumeKernel(uint64_t kernel_id) {
    try {
        std::lock_guard<std::mutex> lock(context_mutex_);
        
        auto context_it = kernel_contexts_.find(kernel_id);
        if (context_it == kernel_contexts_.end()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        PreemptionContext& context = context_it->second;
        
        if (!context.is_preempted) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        // Restore kernel context
        restoreKernelContext(kernel_id);
        
        // Mark as resumed
        context.is_preempted = false;
        context.checkpoint_count++;
        
        // Update statistics
        stats_.successful_preemptions++;
        stats_.preempted_kernels--;
        
        std::cout << "Resumed kernel " << kernel_id << " from checkpoint " 
                  << context.checkpoint_count << std::endl;
        
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "Kernel resume failed: " << e.what() << std::endl;
        return CUDA_ERROR_LAUNCH_FAILED;
    }
}

void PreemptionManager::updateProgress(uint64_t kernel_id, float progress) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it != kernel_contexts_.end()) {
        context_it->second.execution_progress = std::max(0.0f, std::min(1.0f, progress));
    }
}

void PreemptionManager::setStrategy(PreemptionStrategy strategy) {
    strategy_ = strategy;
    std::cout << "Preemption strategy changed to " << static_cast<int>(strategy) << std::endl;
}

PreemptionManager::PreemptionStats PreemptionManager::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void PreemptionManager::setKernelPreemptible(uint64_t kernel_id, bool enabled) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it != kernel_contexts_.end()) {
        context_it->second.is_preemptible = enabled;
    }
}

PreemptionContext PreemptionManager::getKernelContext(uint64_t kernel_id) const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it != kernel_contexts_.end()) {
        return context_it->second;
    }
    
    return PreemptionContext{};
}

void PreemptionManager::shutdown() {
    try {
        shutdown_requested_ = true;
        
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
        
        // Clean up contexts
        kernel_contexts_.clear();
        kernel_start_times_.clear();
        gpu_performance_history_.clear();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during preemption shutdown: " << e.what() << std::endl;
    }
}

void PreemptionManager::monitoringLoop() {
    while (!shutdown_requested_) {
        try {
            // Check for preemption requests
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                
                while (!preemption_queue_.empty()) {
                    PreemptionRequest request = preemption_queue_.front();
                    preemption_queue_.pop();
                    
                    if (shouldPreempt(request.kernel_id)) {
                        executePreemption(request.kernel_id);
                    }
                }
            }
            
            // Clean up expired contexts
            cleanupExpiredContexts();
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in preemption monitoring loop: " << e.what() << std::endl;
        }
    }
}

bool PreemptionManager::canPreempt(uint64_t kernel_id) const {
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it == kernel_contexts_.end()) {
        return false;
    }
    
    const PreemptionContext& context = context_it->second;
    return context.is_preemptible && !context.is_preempted;
}

bool PreemptionManager::shouldPreempt_Cooperative(uint64_t kernel_id) {
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it == kernel_contexts_.end()) {
        return false;
    }
    
    const PreemptionContext& context = context_it->second;
    
    if (context.execution_progress > 0.5f && context.checkpoint_count < 3) {
        return true;  // Allow preemption after 50% progress
    }
    
    return false;
}

bool PreemptionManager::shouldPreempt_Preemptive(uint64_t kernel_id) {
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it == kernel_contexts_.end()) {
        return false;
    }
    
    const PreemptionContext& context = context_it->second;
    
    // Preemptive preemption: check for high-priority requests
    auto now = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - context.start_time).count();
    
    // Preempt if kernel has been running too long
    if (execution_time > 1000) {  // 1 second threshold
        return true;
    }
    
    return false;
}

bool PreemptionManager::shouldPreempt_Adaptive(uint64_t kernel_id) {
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it == kernel_contexts_.end()) {
        return false;
    }
    
    const PreemptionContext& context = context_it->second;
    
    // Adaptive preemption: consider multiple factors
    auto now = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - context.start_time).count();
    
    // Factor 1: Execution time
    float time_factor = std::min(1.0f, execution_time / 5000.0f);  // 5 second max
    
    // Factor 2: Progress
    float progress_factor = context.execution_progress;
    
    // Factor 3: Priority (lower priority = more likely to be preempted)
    float priority_factor = 1.0f - (static_cast<int>(context.current_priority) / 3.0f);
    
    // Combined score
    float preemption_score = (time_factor * 0.4f + progress_factor * 0.3f + priority_factor * 0.3f);
    
    return preemption_score > 0.7f;  // 70% threshold
}

bool PreemptionManager::shouldPreempt_TimeSlice(uint64_t kernel_id) {
    auto start_time_it = kernel_start_times_.find(kernel_id);
    if (start_time_it == kernel_start_times_.end()) {
        return false;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - start_time_it->second).count();
    
    // Time-slice preemption: preempt after time slice expires
    return execution_time > time_slice_ms_;
}

void PreemptionManager::updatePreemptionStats(uint64_t kernel_id, bool successful) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (successful) {
        stats_.successful_preemptions++;
    } else {
        stats_.failed_preemptions++;
    }
    
    // Update average preemption time
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it != kernel_contexts_.end()) {
        auto preemption_time = std::chrono::duration_cast<std::chrono::microseconds>(
            context_it->second.last_checkpoint - context_it->second.start_time).count();
        
        stats_.avg_preemption_time_ms = 
            (stats_.avg_preemption_time_ms * (stats_.total_preemptions - 1) + preemption_time / 1000.0f) 
            / stats_.total_preemptions;
    }
}

float PreemptionManager::calculateUrgencyScore(const PreemptionRequest& request) const {
    // Calculate urgency based on priority and request time
    float priority_score = static_cast<float>(request.new_priority) / 3.0f;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - request.request_time).count();
    
    float age_factor = std::min(1.0f, age / 1000.0f);  // Age penalty
    
    return (priority_score * 0.7f + age_factor * 0.3f);
}

void PreemptionManager::cleanupExpiredContexts() {
    auto now = std::chrono::high_resolution_clock::now();
    auto expiry_threshold = now - std::chrono::minutes(5);  // 5 minute expiry
    
    std::vector<uint64_t> expired_kernels;
    
    for (const auto& pair : kernel_contexts_) {
        if (pair.second.start_time < expiry_threshold) {
            expired_kernels.push_back(pair.first);
        }
    }
    
    for (uint64_t kernel_id : expired_kernels) {
        unregisterKernel(kernel_id);
    }
}

bool PreemptionManager::isKernelExpired(uint64_t kernel_id) const {
    auto context_it = kernel_contexts_.find(kernel_id);
    if (context_it == kernel_contexts_.end()) {
        return true;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::minutes>(
        now - context_it->second.start_time).count();
    
    return age > 5;  // 5 minute expiry
}

void PreemptionManager::restoreKernelContext(uint64_t kernel_id) {
    auto it = kernel_contexts_.find(kernel_id);
    if (it == kernel_contexts_.end()) {
        std::cerr << "Warning: No context found for kernel " << kernel_id << std::endl;
        return;
    }

    PreemptionContext& context = it->second;
    
    // Restore CUDA context
    CUresult result = cuCtxSetCurrent(context.saved_context);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to restore CUDA context for kernel " << kernel_id 
                  << ": " << getCUDAErrorString(result) << std::endl;
        return;
    }

    // Restore kernel parameters
    if (context.kernel_function) {
        // Restore grid and block dimensions
        dim3 grid_dim = context.saved_grid_dim;
        dim3 block_dim = context.saved_block_dim;
        
        // Restore shared memory size
        size_t shared_mem = context.saved_shared_mem;
        
        // Restore kernel arguments
        void** args = context.saved_kernel_args.data();
        
        // Restore stream
        cudaStream_t stream = context.saved_stream;
        
        // Launch the kernel with restored parameters
        cudaError_t launch_result = cudaLaunchKernel(
            context.kernel_function,
            grid_dim, block_dim,
            args, shared_mem, stream
        );
        
        if (launch_result != cudaSuccess) {
            std::cerr << "Failed to relaunch kernel " << kernel_id 
                      << ": " << cudaGetErrorString(launch_result) << std::endl;
        } else {
            std::cout << "Successfully restored and relaunched kernel " << kernel_id << std::endl;
        }
    }

    // Restore memory state if needed
    if (!context.saved_memory_regions.empty()) {
        for (const auto& mem_region : context.saved_memory_regions) {
            // Restore memory content if it was modified
            if (mem_region.needs_restore) {
                cudaError_t mem_result = cudaMemcpy(
                    mem_region.device_ptr,
                    mem_region.host_backup.data(),
                    mem_region.size,
                    cudaMemcpyHostToDevice
                );
                
                if (mem_result != cudaSuccess) {
                    std::cerr << "Failed to restore memory region for kernel " << kernel_id 
                              << ": " << cudaGetErrorString(mem_result) << std::endl;
                }
            }
        }
    }

    // Update statistics
    context.restore_count++;
    context.last_restore_time = std::chrono::steady_clock::now();
    
    // Mark as no longer preempted
    context.is_preempted = false;
    context.preemption_end_time = std::chrono::steady_clock::now();
    
    // Calculate preemption duration
    auto preemption_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        context.preemption_end_time - context.preemption_start_time
    );
    context.total_preemption_time += preemption_duration;
    
    std::cout << "Restored context for kernel " << kernel_id 
              << " (preemption duration: " << preemption_duration.count() << "μs)" << std::endl;
}

void PreemptionManager::saveKernelContext(uint64_t kernel_id) {
    auto it = kernel_contexts_.find(kernel_id);
    if (it == kernel_contexts_.end()) {
        std::cerr << "Warning: No context found for kernel " << kernel_id << std::endl;
        return;
    }

    PreemptionContext& context = it->second;
    
    // Save current CUDA context
    CUcontext current_context;
    CUresult result = cuCtxGetCurrent(&current_context);
    if (result == CUDA_SUCCESS) {
        context.saved_context = current_context;
    } else {
        std::cerr << "Failed to get current CUDA context for kernel " << kernel_id 
                  << ": " << getCUDAErrorString(result) << std::endl;
    }

    // Save kernel execution state
    if (context.kernel_function) {
        // Save grid and block dimensions
        context.saved_grid_dim = context.current_grid_dim;
        context.saved_block_dim = context.current_block_dim;
        
        // Save shared memory size
        context.saved_shared_mem = context.current_shared_mem;
        
        // Save kernel arguments (deep copy)
        context.saved_kernel_args = context.current_kernel_args;
        
        // Save stream
        context.saved_stream = context.current_stream;
        
        // Mark that we have saved kernel parameters
        context.saved_kernel_params = true;
    }

    // Save critical memory regions
    if (!context.memory_regions.empty()) {
        context.saved_memory_regions.clear();
        
        for (const auto& mem_region : context.memory_regions) {
            SavedMemoryRegion saved_region;
            saved_region.device_ptr = mem_region.device_ptr;
            saved_region.size = mem_region.size;
            saved_region.needs_restore = mem_region.is_critical;
            
            // Create host backup for critical memory regions
            if (mem_region.is_critical) {
                saved_region.host_backup.resize(mem_region.size);
                
                cudaError_t mem_result = cudaMemcpy(
                    saved_region.host_backup.data(),
                    mem_region.device_ptr,
                    mem_region.size,
                    cudaMemcpyDeviceToHost
                );
                
                if (mem_result != cudaSuccess) {
                    std::cerr << "Failed to backup memory region for kernel " << kernel_id 
                              << ": " << cudaGetErrorString(mem_result) << std::endl;
                    saved_region.needs_restore = false;
                }
            }
            
            context.saved_memory_regions.push_back(saved_region);
        }
    }

    // Save execution progress
    context.saved_progress = context.current_progress;
    context.saved_execution_time = context.current_execution_time;
    
    // Mark as preempted
    context.is_preempted = true;
    context.preemption_start_time = std::chrono::steady_clock::now();
    
    // Update statistics
    context.save_count++;
    context.last_save_time = std::chrono::steady_clock::now();
    
    std::cout << "Saved context for kernel " << kernel_id 
              << " (progress: " << context.saved_progress << "%)" << std::endl;
}

std::string PreemptionManager::getCUDAErrorString(CUresult result) {
    const char* error_string = nullptr;
    cuGetErrorString(result, &error_string);
    
    if (error_string) {
        return std::string(error_string);
    } else {
        return "Unknown CUDA error: " + std::to_string(static_cast<int>(result));
    }
}

} // namespace cuda_scheduler 