#include "cuda_scheduler/multi_gpu_scheduler.hpp"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

namespace cuda_scheduler {

MultiGPUScheduler::MultiGPUScheduler() 
    : strategy_(MultiGPUStrategy::LOAD_BALANCED) {
}

MultiGPUScheduler::~MultiGPUScheduler() {
    shutdown();
}

bool MultiGPUScheduler::initialize(MultiGPUStrategy strategy) {
    try {
        strategy_ = strategy;
        
        // Initialize GPU devices
        if (!initializeGPUs()) {
            std::cerr << "Failed to initialize GPUs" << std::endl;
            return false;
        }
        
        // Initialize load balancing state
        kernels_per_gpu_.resize(gpu_devices_.size());
        gpu_utilizations_.resize(gpu_devices_.size());
        
        for (size_t i = 0; i < gpu_devices_.size(); ++i) {
            kernels_per_gpu_[i] = 0;
            gpu_utilizations_[i] = 0.0f;
        }
        
        std::cout << "Multi-GPU Scheduler initialized with " << gpu_devices_.size() << " GPUs" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Multi-GPU initialization failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<GPUDevice> MultiGPUScheduler::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    return gpu_devices_;
}

int MultiGPUScheduler::selectOptimalGPU(const KernelProfile& profile, float predicted_time) {
    switch (strategy_) {
        case MultiGPUStrategy::ROUND_ROBIN:
            return selectGPU_RoundRobin();
        case MultiGPUStrategy::LOAD_BALANCED:
            return selectGPU_LoadBalanced(profile, predicted_time);
        case MultiGPUStrategy::MEMORY_AWARE:
            return selectGPU_MemoryAware(profile);
        case MultiGPUStrategy::AFFINITY_BASED:
            return selectGPU_AffinityBased(profile);
        case MultiGPUStrategy::PERFORMANCE_OPTIMIZED:
            return selectGPU_PerformanceOptimized(profile, predicted_time);
        default:
            return selectGPU_LoadBalanced(profile, predicted_time);
    }
}

CUresult MultiGPUScheduler::scheduleKernelMultiGPU(const KernelLaunchParams& params) {
    try {
        // Create kernel profile
        KernelProfile profile;
        profile.kernel_id = params.kernel_id;
        profile.grid_dim = params.grid_dim;
        profile.block_dim = params.block_dim;
        profile.shared_mem_size = params.shared_mem_size;
        
        // Get AI prediction if available
        float predicted_time = 0.0f;
        if (gpu_schedulers_[0]) {  // Use first scheduler for prediction
            auto prediction = gpu_schedulers_[0]->getMetrics();
            predicted_time = prediction.avg_execution_time_ms;
        }
        
        // Select optimal GPU
        int selected_gpu = selectOptimalGPU(profile, predicted_time);
        
        if (selected_gpu < 0 || selected_gpu >= static_cast<int>(gpu_schedulers_.size())) {
            std::cerr << "Invalid GPU selection: " << selected_gpu << std::endl;
            return CUDA_ERROR_LAUNCH_FAILED;
        }
        
        // Set CUDA device
        cudaError_t cuda_error = cudaSetDevice(selected_gpu);
        if (cuda_error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device " << selected_gpu << ": " 
                      << cudaGetErrorString(cuda_error) << std::endl;
            return CUDA_ERROR_LAUNCH_FAILED;
        }
        
        // Schedule kernel on selected GPU
        CUresult result = gpu_schedulers_[selected_gpu]->scheduleKernel(params);
        
        if (result == CUDA_SUCCESS) {
            // Update statistics
            kernels_per_gpu_[selected_gpu]++;
            total_kernels_scheduled_++;
            
            // Update performance history
            updatePerformanceHistory(selected_gpu, predicted_time);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Multi-GPU scheduling failed: " << e.what() << std::endl;
        return CUDA_ERROR_LAUNCH_FAILED;
    }
}

void MultiGPUScheduler::updateDeviceInfo(int device_id, const GPUDevice& device_info) {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    if (device_id >= 0 && device_id < static_cast<int>(gpu_devices_.size())) {
        gpu_devices_[device_id] = device_info;
        gpu_utilizations_[device_id] = device_info.current_utilization;
    }
}

MultiGPUScheduler::LoadBalancingStats MultiGPUScheduler::getLoadBalancingStats() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    LoadBalancingStats stats;
    stats.gpu_utilizations.resize(gpu_devices_.size());
    stats.kernels_per_gpu.resize(gpu_devices_.size());
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        stats.gpu_utilizations[i] = gpu_utilizations_[i];
        stats.kernels_per_gpu[i] = kernels_per_gpu_[i];
    }
    
    stats.load_balance_score = calculateLoadBalanceScore();
    stats.total_kernels_scheduled = total_kernels_scheduled_;
    
    return stats;
}

void MultiGPUScheduler::setStrategy(MultiGPUStrategy strategy) {
    strategy_ = strategy;
    std::cout << "Multi-GPU strategy changed to " << static_cast<int>(strategy) << std::endl;
}

void MultiGPUScheduler::setDeviceEnabled(int device_id, bool enabled) {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    if (device_id >= 0 && device_id < static_cast<int>(gpu_devices_.size())) {
        gpu_devices_[device_id].is_available = enabled;
        std::cout << "GPU " << device_id << " " << (enabled ? "enabled" : "disabled") << std::endl;
    }
}

std::vector<PerformanceMetrics> MultiGPUScheduler::getMultiGPUMetrics() const {
    std::vector<PerformanceMetrics> metrics;
    metrics.reserve(gpu_schedulers_.size());
    
    for (const auto& scheduler : gpu_schedulers_) {
        if (scheduler) {
            metrics.push_back(scheduler->getMetrics());
        } else {
            metrics.push_back(PerformanceMetrics{});
        }
    }
    
    return metrics;
}

void MultiGPUScheduler::shutdown() {
    try {
        for (auto& scheduler : gpu_schedulers_) {
            if (scheduler) {
                scheduler->shutdown();
            }
        }
        gpu_schedulers_.clear();
        gpu_devices_.clear();
    } catch (const std::exception& e) {
        std::cerr << "Error during multi-GPU shutdown: " << e.what() << std::endl;
    }
}

bool MultiGPUScheduler::initializeGPUs() {
    try {
        // Get number of CUDA devices
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        std::cout << "Found " << device_count << " CUDA devices" << std::endl;
        
        // Initialize each GPU device
        gpu_devices_.resize(device_count);
        gpu_schedulers_.resize(device_count);
        
        for (int i = 0; i < device_count; ++i) {
            // Get device properties
            cudaDeviceProp prop;
            error = cudaGetDeviceProperties(&prop, i);
            if (error != cudaSuccess) {
                std::cerr << "Failed to get properties for device " << i << std::endl;
                continue;
            }
            
            // Initialize GPU device info
            gpu_devices_[i].device_id = i;
            gpu_devices_[i].total_memory_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
            gpu_devices_[i].multiprocessor_count = prop.multiProcessorCount;
            gpu_devices_[i].compute_capability_major = prop.major;
            gpu_devices_[i].compute_capability_minor = prop.minor;
            gpu_devices_[i].is_available = true;
            
            // Create scheduler for this GPU
            gpu_schedulers_[i] = CUDAScheduler::create();
            
            // Configure scheduler for this device
            SchedulerConfig config;
            config.enable_ai_scheduling = true;
            config.max_queue_size = 5000;  // Smaller queues per GPU
            
            if (!gpu_schedulers_[i]->initialize(config)) {
                std::cerr << "Failed to initialize scheduler for GPU " << i << std::endl;
                gpu_devices_[i].is_available = false;
            }
            
            std::cout << "GPU " << i << ": " << prop.name 
                      << " (" << gpu_devices_[i].total_memory_gb << "GB, "
                      << gpu_devices_[i].multiprocessor_count << " SMs)" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void MultiGPUScheduler::updateDeviceMetrics() {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (gpu_devices_[i].is_available) {
            // Update GPU metrics (simplified - in practice would use NVML)
            gpu_devices_[i].current_utilization = gpu_utilizations_[i];
            gpu_devices_[i].free_memory_mb = gpu_devices_[i].total_memory_gb * 1024 * 0.8f;  // Simplified
        }
    }
}

int MultiGPUScheduler::selectGPU_RoundRobin() {
    static std::atomic<size_t> current_gpu{0};
    
    for (size_t attempt = 0; attempt < gpu_devices_.size(); ++attempt) {
        size_t gpu = current_gpu.fetch_add(1) % gpu_devices_.size();
        
        if (gpu_devices_[gpu].is_available && !isDeviceOverloaded(gpu)) {
            return static_cast<int>(gpu);
        }
    }
    
    return 0;  // Fallback to first GPU
}

int MultiGPUScheduler::selectGPU_LoadBalanced(const KernelProfile& profile, float predicted_time) {
    int best_gpu = -1;
    float best_score = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (!gpu_devices_[i].is_available || isDeviceOverloaded(i)) {
            continue;
        }
        
        // Calculate load balancing score
        float utilization = gpu_utilizations_[i];
        float queue_length = static_cast<float>(kernels_per_gpu_[i]);
        float memory_pressure = 1.0f - (static_cast<float>(gpu_devices_[i].free_memory_mb) / 
                                       (gpu_devices_[i].total_memory_gb * 1024));
        
        // Weighted score (lower is better)
        float score = utilization * 0.5f + queue_length * 0.3f + memory_pressure * 0.2f;
        
        if (score < best_score) {
            best_score = score;
            best_gpu = static_cast<int>(i);
        }
    }
    
    return best_gpu >= 0 ? best_gpu : 0;
}

int MultiGPUScheduler::selectGPU_MemoryAware(const KernelProfile& profile) {
    int best_gpu = -1;
    size_t best_free_memory = 0;
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (!gpu_devices_[i].is_available) {
            continue;
        }
        
        if (hasSufficientMemory(i, profile) && gpu_devices_[i].free_memory_mb > best_free_memory) {
            best_free_memory = gpu_devices_[i].free_memory_mb;
            best_gpu = static_cast<int>(i);
        }
    }
    
    return best_gpu >= 0 ? best_gpu : 0;
}

int MultiGPUScheduler::selectGPU_AffinityBased(const KernelProfile& profile) {
    // Simple affinity: prefer GPU with most recent kernel of same type
    // In practice, this would track kernel relationships and data locality
    
    int best_gpu = -1;
    auto now = std::chrono::high_resolution_clock::now();
    auto best_time = now - std::chrono::hours(1);  // Default to old time
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (!gpu_devices_[i].is_available) {
            continue;
        }
        
        auto it = last_kernel_time_.find(static_cast<int>(i));
        if (it != last_kernel_time_.end() && it->second > best_time) {
            best_time = it->second;
            best_gpu = static_cast<int>(i);
        }
    }
    
    return best_gpu >= 0 ? best_gpu : 0;
}

int MultiGPUScheduler::selectGPU_PerformanceOptimized(const KernelProfile& profile, float predicted_time) {
    int best_gpu = -1;
    float best_performance = 0.0f;
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (!gpu_devices_[i].is_available) {
            continue;
        }
        
        // Calculate expected performance on this GPU
        float gpu_performance = static_cast<float>(gpu_devices_[i].multiprocessor_count) / 100.0f;
        float utilization_penalty = gpu_utilizations_[i] * 0.5f;
        float memory_penalty = (1.0f - static_cast<float>(gpu_devices_[i].free_memory_mb) / 
                               (gpu_devices_[i].total_memory_gb * 1024)) * 0.3f;
        
        float performance_score = gpu_performance - utilization_penalty - memory_penalty;
        
        if (performance_score > best_performance) {
            best_performance = performance_score;
            best_gpu = static_cast<int>(i);
        }
    }
    
    return best_gpu >= 0 ? best_gpu : 0;
}

float MultiGPUScheduler::calculateLoadBalanceScore() const {
    if (gpu_devices_.empty()) {
        return 0.0f;
    }
    
    // Calculate standard deviation of GPU utilizations
    float mean_utilization = 0.0f;
    for (float util : gpu_utilizations_) {
        mean_utilization += util;
    }
    mean_utilization /= gpu_utilizations_.size();
    
    float variance = 0.0f;
    for (float util : gpu_utilizations_) {
        variance += (util - mean_utilization) * (util - mean_utilization);
    }
    variance /= gpu_utilizations_.size();
    
    float std_dev = std::sqrt(variance);
    
    // Convert to balance score (0.0 = perfectly balanced, 1.0 = completely unbalanced)
    return std::min(1.0f, std_dev / 100.0f);
}

void MultiGPUScheduler::updatePerformanceHistory(int device_id, float execution_time) {
    if (device_id >= 0 && device_id < static_cast<int>(gpu_performance_history_.size())) {
        auto& history = gpu_performance_history_[device_id];
        history.push_back(execution_time);
        
        // Keep only recent history (last 100 entries)
        if (history.size() > 100) {
            history.erase(history.begin());
        }
    }
}

bool MultiGPUScheduler::isDeviceOverloaded(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(gpu_devices_.size())) {
        return true;
    }
    
    // Check utilization threshold
    if (gpu_utilizations_[device_id] > 90.0f) {
        return true;
    }
    
    // Check queue length threshold
    if (kernels_per_gpu_[device_id] > 100) {
        return true;
    }
    
    // Check memory pressure
    if (gpu_devices_[device_id].free_memory_mb < 1024) {  // Less than 1GB free
        return true;
    }
    
    return false;
}

bool MultiGPUScheduler::hasSufficientMemory(int device_id, const KernelProfile& profile) const {
    if (device_id < 0 || device_id >= static_cast<int>(gpu_devices_.size())) {
        return false;
    }
    
    // Estimate memory requirement (simplified)
    size_t estimated_memory = profile.grid_dim.x * profile.grid_dim.y * profile.grid_dim.z *
                             profile.block_dim.x * profile.block_dim.y * profile.block_dim.z * 4;  // 4 bytes per thread
    
    return gpu_devices_[device_id].free_memory_mb * 1024 * 1024 > estimated_memory;
}

} // namespace cuda_scheduler 