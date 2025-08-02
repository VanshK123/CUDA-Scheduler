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
    // In a real implementation, this would restore CUDA context state
    // For now, we'll just log the restoration
    std::cout << "Restoring context for kernel " << kernel_id << std::endl;
}

void PreemptionManager::saveKernelContext(uint64_t kernel_id) {
    // In a real implementation, this would save CUDA context state
    // For now, we'll just log the save
    std::cout << "Saving context for kernel " << kernel_id << std::endl;
}

} // namespace cuda_scheduler 