#include "cuda_scheduler/telemetry.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

#ifdef HAVE_NVML
#include <nvml.h>
#endif

namespace cuda_scheduler {

TelemetryCollector::TelemetryCollector() 
    : total_kernels_launched_(0)
    , total_kernels_completed_(0) {
}

TelemetryCollector::~TelemetryCollector() {
    shutdown();
}

bool TelemetryCollector::initialize() {
    try {
        // Initialize CUPTI for detailed profiling
        if (!initializeCUPTI()) {
            std::cerr << "Warning: CUPTI initialization failed, using basic telemetry" << std::endl;
        }
        
        // Start background processing thread
        shutdown_requested_ = false;
        processing_thread_ = std::thread(&TelemetryCollector::processingLoop, this);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Telemetry initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void TelemetryCollector::recordKernelLaunch(const KernelLaunchParams& params) {
    if (!collection_enabled_) {
        return;
    }
    
    try {
        // Create kernel profile
        KernelProfile profile = createProfile(params);
        
        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (profile_queue_.size() < MAX_QUEUE_SIZE) {
                profile_queue_.push(profile);
                queue_cv_.notify_one();
            }
        }
        
        total_kernels_launched_++;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to record kernel launch: " << e.what() << std::endl;
    }
}

void TelemetryCollector::recordKernelCompletion(uint64_t kernel_id, uint64_t execution_time_ns) {
    if (!collection_enabled_) {
        return;
    }
    
    try {
        // Update GPU metrics
        updateGPUMetrics();
        
        total_kernels_completed_++;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to record kernel completion: " << e.what() << std::endl;
    }
}

std::vector<KernelProfile> TelemetryCollector::exportProfiles() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    std::vector<KernelProfile> profiles;
    profiles.reserve(profile_queue_.size());
    
    while (!profile_queue_.empty()) {
        profiles.push_back(profile_queue_.front());
        profile_queue_.pop();
    }
    
    return profiles;
}

TelemetryCollector::GPUMetrics TelemetryCollector::getGPUMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void TelemetryCollector::setCollectionEnabled(bool enabled) {
    collection_enabled_ = enabled;
}

bool TelemetryCollector::isCollectionEnabled() const {
    return collection_enabled_;
}

void TelemetryCollector::clearData() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    while (!profile_queue_.empty()) {
        profile_queue_.pop();
    }
    
    total_kernels_launched_ = 0;
    total_kernels_completed_ = 0;
}

void TelemetryCollector::shutdown() {
    try {
        shutdown_requested_ = true;
        
        if (processing_thread_.joinable()) {
            queue_cv_.notify_all();
            processing_thread_.join();
        }
        
        cleanupCUPTI();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during telemetry shutdown: " << e.what() << std::endl;
    }
}

void TelemetryCollector::processingLoop() {
    while (!shutdown_requested_) {
        KernelProfile profile;
        bool has_profile = false;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                                  [this] { return !profile_queue_.empty() || shutdown_requested_; })) {
                if (!profile_queue_.empty()) {
                    profile = profile_queue_.front();
                    profile_queue_.pop();
                    has_profile = true;
                }
            }
        }
        
        if (has_profile) {
            processProfile(profile);
        }
        
        // Update GPU metrics periodically
        static int update_counter = 0;
        if (++update_counter % 10 == 0) {
            updateGPUMetrics();
        }
        
        // Clean up completed kernels periodically
        static int cleanup_counter = 0;
        if (++cleanup_counter % 50 == 0) {
            cleanupCompletedKernels();
            
            // Log kernel tracking statistics
            auto stats = getKernelTrackingStats();
            if (stats.total_kernels_tracked > 0) {
                std::cout << "Kernel Tracking Stats - Total: " << stats.total_kernels_tracked
                          << ", Completed: " << stats.completed_kernels
                          << ", Pending: " << stats.pending_kernels
                          << ", Active Streams: " << stats.active_streams
                          << ", Avg Time: " << stats.avg_execution_time_ms << "ms" << std::endl;
            }
        }
    }
}

void TelemetryCollector::updateGPUMetrics() {
#ifdef HAVE_NVML
    try {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result == NVML_SUCCESS) {
            // Get SM utilization
            unsigned int sm_utilization;
            result = nvmlDeviceGetUtilizationRates(device, &sm_utilization, nullptr);
            if (result == NVML_SUCCESS) {
                current_metrics_.sm_utilization_percent = static_cast<float>(sm_utilization);
            }
            
            // Get memory utilization
            nvmlMemory_t memory;
            result = nvmlDeviceGetMemoryInfo(device, &memory);
            if (result == NVML_SUCCESS) {
                current_metrics_.memory_utilization_percent = 
                    static_cast<float>(memory.used) / static_cast<float>(memory.total) * 100.0f;
                current_metrics_.free_memory_mb = memory.free / (1024 * 1024);
                current_metrics_.total_memory_mb = memory.total / (1024 * 1024);
            }
            
            // Get power usage
            unsigned int power_mw;
            result = nvmlDeviceGetPowerUsage(device, &power_mw);
            if (result == NVML_SUCCESS) {
                current_metrics_.power_usage_watts = power_mw / 1000.0f;
            }
            
            // Get temperature
            unsigned int temperature;
            result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
            if (result == NVML_SUCCESS) {
                current_metrics_.temperature_celsius = static_cast<int>(temperature);
            }
            
        }
    } catch (const std::exception& e) {
        // NVML failed, use default values
        current_metrics_.sm_utilization_percent = 0.0f;
        current_metrics_.memory_utilization_percent = 0.0f;
        current_metrics_.power_usage_watts = 0.0f;
        current_metrics_.temperature_celsius = 0;
    }
#else
    // No NVML available, use default values
    current_metrics_.sm_utilization_percent = 0.0f;
    current_metrics_.memory_utilization_percent = 0.0f;
    current_metrics_.power_usage_watts = 0.0f;
    current_metrics_.temperature_celsius = 0;
#endif
}

KernelProfile TelemetryCollector::createProfile(const KernelLaunchParams& params) {
    KernelProfile profile;
    
    profile.kernel_id = params.kernel_id;
    profile.grid_dim = params.grid_dim;
    profile.block_dim = params.block_dim;
    profile.shared_mem_size = params.shared_mem_size;
    profile.launch_time = params.launch_time;
    
    // Estimate operation type based on kernel function pointer
    profile.operation_type = "unknown";
    
    // Calculate input tensor volume (simplified)
    profile.input_shapes.clear();
    if (params.args) {
        // This is a placeholder - real implementation would analyze kernel arguments
        profile.input_shapes.push_back(profile.grid_dim.x * profile.grid_dim.y * profile.grid_dim.z);
    }
    
    // Initialize performance metrics
    profile.execution_time_ns = 0;
    profile.sm_utilization = 0.0f;
    profile.memory_bandwidth_gb_s = 0.0f;
    
    return profile;
}

void TelemetryCollector::processProfile(const KernelProfile& profile) {
    // Update GPU metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        current_metrics_.sm_utilization_percent = profile.sm_utilization;
        current_metrics_.memory_bandwidth_gb_s = profile.memory_bandwidth_gb_s;
    }
    
}

bool TelemetryCollector::initializeCUPTI() {
    try {
        // CUPTI (CUDA Profiling Tools Interface) initialization
        // This provides detailed kernel execution profiling
        
        // Initialize CUPTI callbacks
        CUptiResult result = cuptiSubscribe(&cupti_subscriber_, 
                                          (CUpti_CallbackFunc)cuptiCallback, 
                                          this);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to subscribe to CUPTI callbacks: " << result << std::endl;
            return false;
        }
        
        // Enable CUPTI callbacks for kernel launches
        result = cuptiEnableCallback(1, cupti_subscriber_, 
                                   CUPTI_CB_DOMAIN_RUNTIME_API, 
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to enable CUPTI kernel launch callback: " << result << std::endl;
            return false;
        }
        
        // Enable callbacks for kernel completion
        result = cuptiEnableCallback(1, cupti_subscriber_, 
                                   CUPTI_CB_DOMAIN_RUNTIME_API, 
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to enable CUPTI kernel completion callback: " << result << std::endl;
            return false;
        }
        
        // Initialize CUPTI event groups for detailed metrics
        result = cuptiEventGroupCreate(cuda_context_, &cupti_event_group_, 0);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to create CUPTI event group: " << result << std::endl;
            return false;
        }
        
        // Add events for SM utilization, memory throughput, etc.
        result = cuptiEventGroupAddEvent(cupti_event_group_, CUPTI_EVENT_ACTIVE_WARPS);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to add CUPTI event: " << result << std::endl;
            return false;
        }
        
        result = cuptiEventGroupAddEvent(cupti_event_group_, CUPTI_EVENT_ACTIVE_CYCLES);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to add CUPTI event: " << result << std::endl;
            return false;
        }
        
        // Enable the event group
        result = cuptiEventGroupEnable(cupti_event_group_);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to enable CUPTI event group: " << result << std::endl;
            return false;
        }
        
        std::cout << "CUPTI initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CUPTI initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void TelemetryCollector::cleanupCUPTI() {
    try {
        if (cupti_event_group_) {
            cuptiEventGroupDisable(cupti_event_group_);
            cuptiEventGroupDestroy(cupti_event_group_);
            cupti_event_group_ = nullptr;
        }
        
        if (cupti_subscriber_) {
            cuptiUnsubscribe(cupti_subscriber_);
            cupti_subscriber_ = nullptr;
        }
        
        std::cout << "CUPTI cleaned up successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "CUPTI cleanup failed: " << e.what() << std::endl;
    }
}

void TelemetryCollector::trimQueue() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    while (profile_queue_.size() > MAX_QUEUE_SIZE / 2) {
        profile_queue_.pop();
    }
}

void TelemetryCollector::cleanupCompletedKernels() {
    std::lock_guard<std::mutex> lock(kernel_tracking_mutex_);
    
    // Clean up completed kernels from stream maps
    for (auto& stream_pair : stream_kernel_map_) {
        auto& kernel_list = stream_pair.second;
        
        // Remove completed kernels
        kernel_list.erase(
            std::remove_if(kernel_list.begin(), kernel_list.end(),
                [](const KernelTrackingInfo& info) { return info.completed; }),
            kernel_list.end()
        );
    }
    
    // Clean up completed kernels from tracking map (keep last 1000 for statistics)
    const size_t max_tracking_size = 1000;
    if (kernel_tracking_map_.size() > max_tracking_size) {
        std::vector<uint64_t> completed_kernels;
        
        for (const auto& pair : kernel_tracking_map_) {
            if (pair.second.completed) {
                completed_kernels.push_back(pair.first);
            }
        }
        
        // Remove oldest completed kernels
        size_t to_remove = kernel_tracking_map_.size() - max_tracking_size;
        for (size_t i = 0; i < std::min(to_remove, completed_kernels.size()); ++i) {
            kernel_tracking_map_.erase(completed_kernels[i]);
        }
    }
}

void TelemetryCollector::cuptiCallback(void* userdata, CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid, const void* cbdata) {
    TelemetryCollector* collector = static_cast<TelemetryCollector*>(userdata);
    
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        const CUpti_CallbackData* callback_data = static_cast<const CUpti_CallbackData*>(cbdata);
        
        switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020: {
                if (callback_data->callbackSite == CUPTI_API_ENTER) {
                    // Kernel launch started
                    const cudaLaunchKernel_params* params = 
                        static_cast<const cudaLaunchKernel_params*>(callback_data->functionParams);
                    
                    // Extract kernel information
                    KernelLaunchParams launch_params;
                    launch_params.func = params->func;
                    launch_params.grid_dim = params->gridDim;
                    launch_params.block_dim = params->blockDim;
                    launch_params.shared_mem_size = params->sharedMem;
                    launch_params.stream = params->stream;
                    launch_params.args = params->args;
                    launch_params.kernel_id = utils::generateKernelId();
                    launch_params.launch_time = utils::getCurrentTime();
                    
                    // Record kernel launch
                    collector->recordKernelLaunch(launch_params);
                    
                    // Track kernel for completion detection
                    std::lock_guard<std::mutex> lock(collector->kernel_tracking_mutex_);
                    
                    KernelTrackingInfo tracking_info;
                    tracking_info.kernel_id = launch_params.kernel_id;
                    tracking_info.launch_time = std::chrono::high_resolution_clock::now();
                    tracking_info.stream = launch_params.stream;
                    tracking_info.launch_params = launch_params;
                    tracking_info.completed = false;
                    tracking_info.execution_time_ns = 0;
                    
                    // Add to stream tracking
                    collector->stream_kernel_map_[launch_params.stream].push_back(tracking_info);
                    
                    // Add to kernel tracking map
                    collector->kernel_tracking_map_[launch_params.kernel_id] = tracking_info;
                    
                    std::cout << "Tracking kernel " << launch_params.kernel_id 
                              << " on stream " << launch_params.stream << std::endl;
                }
                break;
            }
            
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020: {
                if (callback_data->callbackSite == CUPTI_API_EXIT) {
                    // Kernel completion detected through stream synchronization
                    const cudaStreamSynchronize_params* params = 
                        static_cast<const cudaStreamSynchronize_params*>(callback_data->functionParams);
                    
                    cudaStream_t stream = params->stream;
                    
                    // Find completed kernels for this stream
                    std::lock_guard<std::mutex> lock(collector->kernel_tracking_mutex_);
                    
                    auto stream_it = collector->stream_kernel_map_.find(stream);
                    if (stream_it != collector->stream_kernel_map_.end()) {
                        auto& kernel_list = stream_it->second;
                        
                        for (auto& kernel_info : kernel_list) {
                            if (!kernel_info.completed) {
                                // Calculate execution time
                                auto completion_time = std::chrono::high_resolution_clock::now();
                                auto execution_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    completion_time - kernel_info.launch_time
                                );
                                
                                kernel_info.completed = true;
                                kernel_info.execution_time_ns = execution_duration.count();
                                
                                // Record kernel completion
                                collector->recordKernelCompletion(kernel_info.kernel_id, kernel_info.execution_time_ns);
                                
                                // Update tracking map
                                collector->kernel_tracking_map_[kernel_info.kernel_id] = kernel_info;
                                
                                std::cout << "Kernel " << kernel_info.kernel_id 
                                          << " completed in " << kernel_info.execution_time_ns / 1000000.0 
                                          << " ms" << std::endl;
                            }
                        }
                    }
                }
                break;
            }
            
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020: {
                if (callback_data->callbackSite == CUPTI_API_ENTER) {
                    // Event recorded - can be used for kernel completion tracking
                    const cudaEventRecord_params* params = 
                        static_cast<const cudaEventRecord_params*>(callback_data->functionParams);
                    
                    cudaStream_t stream = params->stream;
                    cudaEvent_t event = params->event;
                    
                    // Mark kernels on this stream as potentially completed
                    std::lock_guard<std::mutex> lock(collector->kernel_tracking_mutex_);
                    
                    auto stream_it = collector->stream_kernel_map_.find(stream);
                    if (stream_it != collector->stream_kernel_map_.end()) {
                        auto& kernel_list = stream_it->second;
                        
                        for (auto& kernel_info : kernel_list) {
                            if (!kernel_info.completed) {
                                // Estimate completion based on event recording
                                auto current_time = std::chrono::high_resolution_clock::now();
                                auto estimated_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    current_time - kernel_info.launch_time
                                );
                                
                                // If kernel has been running for a reasonable time, mark as completed
                                if (estimated_duration.count() > 1000000) { // > 1ms
                                    kernel_info.completed = true;
                                    kernel_info.execution_time_ns = estimated_duration.count();
                                    
                                    collector->recordKernelCompletion(kernel_info.kernel_id, kernel_info.execution_time_ns);
                                    collector->kernel_tracking_map_[kernel_info.kernel_id] = kernel_info;
                                    
                                    std::cout << "Kernel " << kernel_info.kernel_id 
                                              << " completed via event recording in " 
                                              << kernel_info.execution_time_ns / 1000000.0 << " ms" << std::endl;
                                }
                            }
                        }
                    }
                }
                break;
            }
            
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020: {
                if (callback_data->callbackSite == CUPTI_API_EXIT) {
                    // Event synchronization completed - all kernels up to this event are done
                    const cudaEventSynchronize_params* params = 
                        static_cast<const cudaEventSynchronize_params*>(callback_data->functionParams);
                    
                    cudaEvent_t event = params->event;
                    
                    // Mark all kernels as completed (simplified approach)
                    std::lock_guard<std::mutex> lock(collector->kernel_tracking_mutex_);
                    
                    for (auto& stream_pair : collector->stream_kernel_map_) {
                        auto& kernel_list = stream_pair.second;
                        
                        for (auto& kernel_info : kernel_list) {
                            if (!kernel_info.completed) {
                                auto completion_time = std::chrono::high_resolution_clock::now();
                                auto execution_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    completion_time - kernel_info.launch_time
                                );
                                
                                kernel_info.completed = true;
                                kernel_info.execution_time_ns = execution_duration.count();
                                
                                collector->recordKernelCompletion(kernel_info.kernel_id, kernel_info.execution_time_ns);
                                collector->kernel_tracking_map_[kernel_info.kernel_id] = kernel_info;
                                
                                std::cout << "Kernel " << kernel_info.kernel_id 
                                          << " completed via event synchronization in " 
                                          << kernel_info.execution_time_ns / 1000000.0 << " ms" << std::endl;
                            }
                        }
                    }
                }
                break;
            }
        }
    }
}

TelemetryCollector::KernelTrackingStats TelemetryCollector::getKernelTrackingStats() const {
    std::lock_guard<std::mutex> lock(kernel_tracking_mutex_);
    
    KernelTrackingStats stats;
    stats.total_kernels_tracked = kernel_tracking_map_.size();
    stats.completed_kernels = 0;
    stats.pending_kernels = 0;
    stats.active_streams = stream_kernel_map_.size();
    
    uint64_t total_execution_time = 0;
    uint64_t max_execution_time = 0;
    uint64_t min_execution_time = UINT64_MAX;
    size_t completed_count = 0;
    
    for (const auto& pair : kernel_tracking_map_) {
        const auto& kernel_info = pair.second;
        
        if (kernel_info.completed) {
            stats.completed_kernels++;
            completed_count++;
            
            total_execution_time += kernel_info.execution_time_ns;
            max_execution_time = std::max(max_execution_time, kernel_info.execution_time_ns);
            min_execution_time = std::min(min_execution_time, kernel_info.execution_time_ns);
        } else {
            stats.pending_kernels++;
        }
    }
    
    if (completed_count > 0) {
        stats.avg_execution_time_ms = (total_execution_time / completed_count) / 1000000.0f;
        stats.max_execution_time_ms = max_execution_time / 1000000.0f;
        stats.min_execution_time_ms = (min_execution_time == UINT64_MAX) ? 0.0f : min_execution_time / 1000000.0f;
    } else {
        stats.avg_execution_time_ms = 0.0f;
        stats.max_execution_time_ms = 0.0f;
        stats.min_execution_time_ms = 0.0f;
    }
    
    return stats;
}

} // namespace cuda_scheduler 