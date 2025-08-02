#include "cuda_scheduler/performance_monitor.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

#ifdef HAVE_NVML
#include <nvml.h>
#endif

namespace cuda_scheduler {

PerformanceMonitor::PerformanceMonitor() {
    // Initialize summary
    summary_.avg_sm_utilization = 0.0f;
    summary_.avg_memory_utilization = 0.0f;
    summary_.avg_power_usage = 0.0f;
    summary_.total_energy_consumed_joules = 0.0f;
    summary_.total_kernels_executed = 0;
    summary_.avg_kernel_execution_time_ms = 0.0f;
    summary_.prediction_accuracy = 0.0f;
    summary_.alerts_generated = 0;
}

PerformanceMonitor::~PerformanceMonitor() {
    shutdown();
}

bool PerformanceMonitor::initialize() {
    try {
        // Initialize NVML for GPU monitoring
        if (!initializeNVML()) {
            std::cerr << "Warning: NVML initialization failed, using basic monitoring" << std::endl;
        }
        
        // Initialize metrics
        current_metrics_ = MonitoringMetrics{};
        historical_metrics_.clear();
        active_alerts_.clear();
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Performance monitor initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void PerformanceMonitor::startMonitoring() {
    if (monitoring_enabled_) {
        return;  // Already monitoring
    }
    
    monitoring_enabled_ = true;
    shutdown_requested_ = false;
    
    // Start monitoring thread
    monitoring_thread_ = std::thread(&PerformanceMonitor::monitoringLoop, this);
    
    std::cout << "Performance monitoring started" << std::endl;
}

void PerformanceMonitor::stopMonitoring() {
    if (!monitoring_enabled_) {
        return;  // Not monitoring
    }
    
    monitoring_enabled_ = false;
    shutdown_requested_ = true;
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    std::cout << "Performance monitoring stopped" << std::endl;
}

MonitoringMetrics PerformanceMonitor::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

std::vector<MonitoringMetrics> PerformanceMonitor::getHistoricalMetrics(uint32_t duration_ms) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::vector<MonitoringMetrics> result;
    auto cutoff_time = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds(duration_ms);
    
    for (const auto& metrics : historical_metrics_) {
        if (metrics.timestamp >= cutoff_time) {
            result.push_back(metrics);
        }
    }
    
    return result;
}

std::vector<PerformanceAlert> PerformanceMonitor::getAlerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    return active_alerts_;
}

void PerformanceMonitor::clearAlerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    active_alerts_.clear();
}

void PerformanceMonitor::setAlertThresholds(float high_utilization_threshold,
                                          float low_utilization_threshold,
                                          float memory_pressure_threshold,
                                          int thermal_threshold) {
    thresholds_.high_utilization = high_utilization_threshold;
    thresholds_.low_utilization = low_utilization_threshold;
    thresholds_.memory_pressure = memory_pressure_threshold;
    thresholds_.thermal_threshold = thermal_threshold;
}

void PerformanceMonitor::updateMetrics(const MonitoringMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_ = metrics;
    current_metrics_.timestamp = std::chrono::high_resolution_clock::now();
    
    // Add to historical data
    historical_metrics_.push_back(current_metrics_);
    trimHistoricalData();
    
    // Update summary
    updateSummary();
}

PerformanceMonitor::PerformanceSummary PerformanceMonitor::getPerformanceSummary() const {
    std::lock_guard<std::mutex> lock(summary_mutex_);
    return summary_;
}

void PerformanceMonitor::shutdown() {
    try {
        stopMonitoring();
        cleanupNVML();
    } catch (const std::exception& e) {
        std::cerr << "Error during performance monitor shutdown: " << e.what() << std::endl;
    }
}

void PerformanceMonitor::monitoringLoop() {
    while (!shutdown_requested_) {
        try {
            // Update GPU metrics
            updateGPUMetrics();
            
            // Check for alerts
            checkAlerts();
            
            // Remove expired alerts
            removeExpiredAlerts();
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 10Hz monitoring
            
        } catch (const std::exception& e) {
            std::cerr << "Error in monitoring loop: " << e.what() << std::endl;
        }
    }
}

void PerformanceMonitor::updateGPUMetrics() {
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
                current_metrics_.allocated_memory_mb = memory.used / (1024 * 1024);
                current_metrics_.free_memory_mb = memory.free / (1024 * 1024);
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
            
            // Update timestamp
            current_metrics_.timestamp = std::chrono::high_resolution_clock::now();
            
            // Add to historical data
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                historical_metrics_.push_back(current_metrics_);
                trimHistoricalData();
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

void PerformanceMonitor::checkAlerts() {
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
    
    // Check for high utilization
    if (current_metrics_.sm_utilization_percent > thresholds_.high_utilization * 100.0f) {
        std::string message = "High SM utilization: " + std::to_string(current_metrics_.sm_utilization_percent) + "%";
        float severity = (current_metrics_.sm_utilization_percent - thresholds_.high_utilization * 100.0f) / 100.0f;
        generateAlert(AlertType::HIGH_UTILIZATION, message, severity);
    }
    
    // Check for low utilization
    if (current_metrics_.sm_utilization_percent < thresholds_.low_utilization * 100.0f) {
        std::string message = "Low SM utilization: " + std::to_string(current_metrics_.sm_utilization_percent) + "%";
        float severity = (thresholds_.low_utilization * 100.0f - current_metrics_.sm_utilization_percent) / 100.0f;
        generateAlert(AlertType::LOW_UTILIZATION, message, severity);
    }
    
    // Check for memory pressure
    if (current_metrics_.memory_utilization_percent > thresholds_.memory_pressure * 100.0f) {
        std::string message = "High memory utilization: " + std::to_string(current_metrics_.memory_utilization_percent) + "%";
        float severity = (current_metrics_.memory_utilization_percent - thresholds_.memory_pressure * 100.0f) / 100.0f;
        generateAlert(AlertType::MEMORY_PRESSURE, message, severity);
    }
    
    // Check for thermal throttling
    if (current_metrics_.temperature_celsius > thresholds_.thermal_threshold) {
        std::string message = "High temperature: " + std::to_string(current_metrics_.temperature_celsius) + "°C";
        float severity = static_cast<float>(current_metrics_.temperature_celsius - thresholds_.thermal_threshold) / 20.0f;
        generateAlert(AlertType::THERMAL_THROTTLING, message, severity);
    }
}

void PerformanceMonitor::generateAlert(AlertType type, const std::string& message, float severity) {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    PerformanceAlert alert;
    alert.type = type;
    alert.message = message;
    alert.severity = std::min(1.0f, std::max(0.0f, severity));
    alert.timestamp = std::chrono::high_resolution_clock::now();
    
    active_alerts_.push_back(alert);
    summary_.alerts_generated++;
    
    // Log alert
    std::string severity_str = (severity > 0.7f) ? "HIGH" : (severity > 0.3f) ? "MEDIUM" : "LOW";
    std::cout << "[" << severity_str << "] Performance Alert: " << message << std::endl;
}

void PerformanceMonitor::updateSummary() {
    std::lock_guard<std::mutex> lock(summary_mutex_);
    
    if (historical_metrics_.empty()) {
        return;
    }
    
    // Calculate averages
    float total_sm_util = 0.0f;
    float total_memory_util = 0.0f;
    float total_power = 0.0f;
    float total_execution_time = 0.0f;
    uint64_t total_kernels = 0;
    
    for (const auto& metrics : historical_metrics_) {
        total_sm_util += metrics.sm_utilization_percent;
        total_memory_util += metrics.memory_utilization_percent;
        total_power += metrics.power_usage_watts;
        total_execution_time += metrics.avg_execution_time_ms;
        total_kernels += metrics.kernels_completed;
    }
    
    size_t count = historical_metrics_.size();
    summary_.avg_sm_utilization = total_sm_util / count;
    summary_.avg_memory_utilization = total_memory_util / count;
    summary_.avg_power_usage = total_power / count;
    summary_.avg_kernel_execution_time_ms = total_execution_time / count;
    summary_.total_kernels_executed = total_kernels;
    
    // Calculate energy consumption (simplified)
    if (count > 1) {
        auto duration = historical_metrics_.back().timestamp - historical_metrics_.front().timestamp;
        auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        summary_.total_energy_consumed_joules = summary_.avg_power_usage * duration_seconds;
    }
}

bool PerformanceMonitor::initializeNVML() {
#ifdef HAVE_NVML
    try {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "NVML initialization failed: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

void PerformanceMonitor::cleanupNVML() {
#ifdef HAVE_NVML
    try {
        nvmlShutdown();
    } catch (const std::exception& e) {
        std::cerr << "NVML cleanup failed: " << e.what() << std::endl;
    }
#endif
}

bool PerformanceMonitor::getGPUMetrics(MonitoringMetrics& metrics) {
#ifdef HAVE_NVML
    try {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result == NVML_SUCCESS) {
            // Get SM utilization
            unsigned int sm_utilization;
            result = nvmlDeviceGetUtilizationRates(device, &sm_utilization, nullptr);
            if (result == NVML_SUCCESS) {
                metrics.sm_utilization_percent = static_cast<float>(sm_utilization);
            }
            
            // Get memory utilization
            nvmlMemory_t memory;
            result = nvmlDeviceGetMemoryInfo(device, &memory);
            if (result == NVML_SUCCESS) {
                metrics.memory_utilization_percent = 
                    static_cast<float>(memory.used) / static_cast<float>(memory.total) * 100.0f;
            }
            
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to get GPU metrics: " << e.what() << std::endl;
    }
#endif
    return false;
}

void PerformanceMonitor::trimHistoricalData() {
    while (historical_metrics_.size() > MAX_HISTORICAL_METRICS) {
        historical_metrics_.pop_front();
    }
}

void PerformanceMonitor::removeExpiredAlerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    auto it = active_alerts_.begin();
    
    while (it != active_alerts_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->timestamp);
        if (age.count() > ALERT_EXPIRY_MS) {
            it = active_alerts_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace cuda_scheduler 