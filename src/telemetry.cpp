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
    // This is a simplified approach - in practice, you'd want more sophisticated detection
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
    
    // Additional processing could include:
    // - Feature extraction for ML models
    // - Performance trend analysis
    // - Alert generation for performance issues
}

bool TelemetryCollector::initializeCUPTI() {
    // CUPTI initialization would go here
    // For now, return true to indicate basic telemetry is available
    return true;
}

void TelemetryCollector::cleanupCUPTI() {
    // CUPTI cleanup would go here
}

void TelemetryCollector::trimQueue() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    while (profile_queue_.size() > MAX_QUEUE_SIZE / 2) {
        profile_queue_.pop();
    }
}

} // namespace cuda_scheduler 