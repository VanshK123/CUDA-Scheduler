#include "cuda_scheduler/scheduler.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace cuda_scheduler {

namespace utils {

std::string formatDuration(std::chrono::nanoseconds duration) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(duration - hours);
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration - hours - mins);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration - hours - mins - secs);
    
    std::stringstream ss;
    if (hours.count() > 0) {
        ss << hours.count() << "h ";
    }
    if (mins.count() > 0 || hours.count() > 0) {
        ss << mins.count() << "m ";
    }
    ss << secs.count() << "." << std::setfill('0') << std::setw(3) << ms.count() << "s";
    
    return ss.str();
}

std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return ss.str();
}

std::string formatPercentage(float value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << value << "%";
    return ss.str();
}

std::string getPriorityString(Priority priority) {
    switch (priority) {
        case Priority::LOW: return "LOW";
        case Priority::NORMAL: return "NORMAL";
        case Priority::HIGH: return "HIGH";
        case Priority::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

std::string getLogLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::INFO: return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        default: return "UNKNOWN";
    }
}

void printKernelProfile(const KernelProfile& profile) {
    std::cout << "Kernel Profile:" << std::endl;
    std::cout << "  ID: " << profile.kernel_id << std::endl;
    std::cout << "  Grid: (" << profile.grid_dim.x << ", " << profile.grid_dim.y << ", " << profile.grid_dim.z << ")" << std::endl;
    std::cout << "  Block: (" << profile.block_dim.x << ", " << profile.block_dim.y << ", " << profile.block_dim.z << ")" << std::endl;
    std::cout << "  Shared Memory: " << formatBytes(profile.shared_mem_size) << std::endl;
    std::cout << "  Operation Type: " << profile.operation_type << std::endl;
    std::cout << "  Execution Time: " << formatDuration(std::chrono::nanoseconds(profile.execution_time_ns)) << std::endl;
    std::cout << "  SM Utilization: " << formatPercentage(profile.sm_utilization) << std::endl;
    std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) << profile.memory_bandwidth_gb_s << " GB/s" << std::endl;
}

void printPerformanceMetrics(const PerformanceMetrics& metrics) {
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "  Average Queue Wait Time: " << std::fixed << std::setprecision(2) << metrics.avg_queue_wait_ms << " ms" << std::endl;
    std::cout << "  Average Execution Time: " << std::fixed << std::setprecision(2) << metrics.avg_execution_time_ms << " ms" << std::endl;
    std::cout << "  SM Utilization: " << formatPercentage(metrics.sm_utilization_percent) << std::endl;
    std::cout << "  Memory Utilization: " << formatPercentage(metrics.memory_utilization_percent) << std::endl;
    std::cout << "  Prediction Accuracy: " << formatPercentage(metrics.prediction_accuracy) << std::endl;
    std::cout << "  Total Kernels Launched: " << metrics.total_kernels_launched << std::endl;
    std::cout << "  Total Kernels Completed: " << metrics.total_kernels_completed << std::endl;
}

void printSchedulerConfig(const SchedulerConfig& config) {
    std::cout << "Scheduler Configuration:" << std::endl;
    std::cout << "  AI Scheduling Enabled: " << (config.enable_ai_scheduling ? "Yes" : "No") << std::endl;
    std::cout << "  Preemption Enabled: " << (config.enable_preemption ? "Yes" : "No") << std::endl;
    std::cout << "  Latency Critical Threshold: " << std::fixed << std::setprecision(2) << config.latency_critical_threshold_ms << " ms" << std::endl;
    std::cout << "  Throughput Optimization: " << (config.throughput_optimization ? "Yes" : "No") << std::endl;
    std::cout << "  Max Queue Size: " << config.max_queue_size << std::endl;
    std::cout << "  Prediction Cache Size: " << config.prediction_cache_size << std::endl;
    std::cout << "  Model Path: " << config.model_path << std::endl;
}

bool isValidGridBlockConfig(const dim3& grid, const dim3& block) {
    // Check for valid grid dimensions
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return false;
    }
    
    // Check for valid block dimensions
    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return false;
    }
    
    // Check total threads per block (typically max 1024)
    size_t total_threads_per_block = block.x * block.y * block.z;
    if (total_threads_per_block > 1024) {
        return false;
    }
    
    return true;
}

uint64_t generateKernelId() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1);
}

std::chrono::high_resolution_clock::time_point getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

float calculateEfficiency(const KernelProfile& profile) {
    // Calculate efficiency based on resource utilization
    float sm_efficiency = profile.sm_utilization / 100.0f;
    float memory_efficiency = profile.memory_bandwidth_gb_s / 1000.0f;  // Normalize to 1.0
    
    // Weighted average
    return (sm_efficiency * 0.7f + memory_efficiency * 0.3f);
}

std::string getOperationTypeFromFunction(void* func) {
    // This is a simplified approach - in practice, you'd want more sophisticated detection
    // based on function signatures, debug symbols, or runtime analysis
    
    // For now, return a generic type
    return "unknown";
}

bool isLatencyCritical(const std::string& operation_type) {
    // Determine if an operation is latency-critical based on its type
    const std::vector<std::string> latency_critical_ops = {
        "inference", "forward", "prediction", "classification", "detection"
    };
    
    std::string lower_type = operation_type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    for (const auto& op : latency_critical_ops) {
        if (lower_type.find(op) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

bool isThroughputOptimized(const std::string& operation_type) {
    // Determine if an operation is throughput-optimized based on its type
    const std::vector<std::string> throughput_ops = {
        "training", "backward", "optimizer", "batch", "gradient"
    };
    
    std::string lower_type = operation_type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    for (const auto& op : throughput_ops) {
        if (lower_type.find(op) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

} // namespace utils

} // namespace cuda_scheduler 