#pragma once

#include "scheduler.hpp"
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

namespace cuda_scheduler {

/**
 * @brief Performance monitoring metrics
 */
struct MonitoringMetrics {
    // GPU utilization metrics
    float sm_utilization_percent;
    float memory_utilization_percent;
    float memory_bandwidth_gb_s;
    float power_usage_watts;
    int temperature_celsius;
    
    // Kernel execution metrics
    uint64_t kernels_launched;
    uint64_t kernels_completed;
    float avg_execution_time_ms;
    float avg_queue_wait_time_ms;
    
    // Prediction accuracy metrics
    float prediction_accuracy;
    float prediction_error_ms;
    size_t total_predictions;
    
    // Resource allocation metrics
    size_t active_streams;
    size_t allocated_memory_mb;
    size_t free_memory_mb;
    
    // Timestamp
    std::chrono::high_resolution_clock::time_point timestamp;
    
    MonitoringMetrics() : sm_utilization_percent(0.0f), memory_utilization_percent(0.0f),
                         memory_bandwidth_gb_s(0.0f), power_usage_watts(0.0f), temperature_celsius(0),
                         kernels_launched(0), kernels_completed(0), avg_execution_time_ms(0.0f),
                         avg_queue_wait_time_ms(0.0f), prediction_accuracy(0.0f), prediction_error_ms(0.0f),
                         total_predictions(0), active_streams(0), allocated_memory_mb(0), free_memory_mb(0) {}
};

/**
 * @brief Performance alert types
 */
enum class AlertType {
    HIGH_UTILIZATION = 0,
    LOW_UTILIZATION = 1,
    MEMORY_PRESSURE = 2,
    THERMAL_THROTTLING = 3,
    PREDICTION_ACCURACY_DROP = 4,
    QUEUE_OVERFLOW = 5
};

/**
 * @brief Performance alert structure
 */
struct PerformanceAlert {
    AlertType type;
    std::string message;
    float severity;  // 0.0 to 1.0
    std::chrono::high_resolution_clock::time_point timestamp;
    
    PerformanceAlert() : type(AlertType::HIGH_UTILIZATION), severity(0.0f) {}
};

/**
 * @brief Performance monitor for GPU resource tracking
 * 
 * This class provides:
 * 1. Real-time GPU performance monitoring
 * 2. Historical metrics tracking
 * 3. Performance alert generation
 * 4. Resource utilization analysis
 */
class PerformanceMonitor {
public:
    /**
     * @brief Constructor
     */
    PerformanceMonitor();
    
    /**
     * @brief Destructor
     */
    ~PerformanceMonitor();
    
    /**
     * @brief Initialize the performance monitor
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Start monitoring
     */
    void startMonitoring();
    
    /**
     * @brief Stop monitoring
     */
    void stopMonitoring();
    
    /**
     * @brief Get current metrics
     * @return Current monitoring metrics
     */
    MonitoringMetrics getCurrentMetrics() const;
    
    /**
     * @brief Get historical metrics
     * @param duration_ms Duration to look back in milliseconds
     * @return Vector of historical metrics
     */
    std::vector<MonitoringMetrics> getHistoricalMetrics(uint32_t duration_ms) const;
    
    /**
     * @brief Get performance alerts
     * @return Vector of active alerts
     */
    std::vector<PerformanceAlert> getAlerts() const;
    
    /**
     * @brief Clear alerts
     */
    void clearAlerts();
    
    /**
     * @brief Set alert thresholds
     * @param high_utilization_threshold High utilization threshold (0.0-1.0)
     * @param low_utilization_threshold Low utilization threshold (0.0-1.0)
     * @param memory_pressure_threshold Memory pressure threshold (0.0-1.0)
     * @param thermal_threshold Thermal threshold in Celsius
     */
    void setAlertThresholds(float high_utilization_threshold = 0.9f,
                           float low_utilization_threshold = 0.1f,
                           float memory_pressure_threshold = 0.8f,
                           int thermal_threshold = 85);
    
    /**
     * @brief Update metrics from external sources
     * @param metrics New metrics to incorporate
     */
    void updateMetrics(const MonitoringMetrics& metrics);
    
    /**
     * @brief Get performance summary
     * @return Performance summary structure
     */
    struct PerformanceSummary {
        float avg_sm_utilization;
        float avg_memory_utilization;
        float avg_power_usage;
        float total_energy_consumed_joules;
        uint64_t total_kernels_executed;
        float avg_kernel_execution_time_ms;
        float prediction_accuracy;
        size_t alerts_generated;
    };
    
    PerformanceSummary getPerformanceSummary() const;
    
    /**
     * @brief Shutdown the performance monitor
     */
    void shutdown();

private:
    // Monitoring state
    std::atomic<bool> monitoring_enabled_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::thread monitoring_thread_;
    
    // Metrics storage
    mutable std::mutex metrics_mutex_;
    std::deque<MonitoringMetrics> historical_metrics_;
    MonitoringMetrics current_metrics_;
    
    // Alerts
    mutable std::mutex alerts_mutex_;
    std::vector<PerformanceAlert> active_alerts_;
    
    // Configuration
    struct AlertThresholds {
        float high_utilization = 0.9f;
        float low_utilization = 0.1f;
        float memory_pressure = 0.8f;
        int thermal_threshold = 85;
    } thresholds_;
    
    // Statistics
    PerformanceSummary summary_;
    mutable std::mutex summary_mutex_;
    
    // Private methods
    void monitoringLoop();
    void updateGPUMetrics();
    void checkAlerts();
    void generateAlert(AlertType type, const std::string& message, float severity);
    void updateSummary();
    
    // GPU monitoring helpers
    bool initializeNVML();
    void cleanupNVML();
    bool getGPUMetrics(MonitoringMetrics& metrics);
    
    // Historical data management
    static constexpr size_t MAX_HISTORICAL_METRICS = 10000;
    void trimHistoricalData();
    
    // Alert management
    void removeExpiredAlerts();
    static constexpr uint32_t ALERT_EXPIRY_MS = 60000;  // 1 minute
};

} // namespace cuda_scheduler 