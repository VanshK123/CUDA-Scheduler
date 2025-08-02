#include "cuda_scheduler/scheduler.hpp"
#include "cuda_scheduler/telemetry.hpp"
#include "cuda_scheduler/ai_predictor.hpp"
#include "cuda_scheduler/multi_gpu_scheduler.hpp"
#include "cuda_scheduler/preemption.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace cuda_scheduler {

// Global scheduler instance
static std::shared_ptr<CUDAScheduler> g_scheduler_instance = nullptr;

CUDAScheduler::CUDAScheduler() 
    : ai_scheduling_enabled_(true)
    , profiling_enabled_(false)
    , log_level_(LogLevel::INFO) {
}

CUDAScheduler::~CUDAScheduler() {
    shutdown();
}

std::shared_ptr<CUDAScheduler> CUDAScheduler::create() {
    if (!g_scheduler_instance) {
        g_scheduler_instance = std::shared_ptr<CUDAScheduler>(new CUDAScheduler());
    }
    return g_scheduler_instance;
}

bool CUDAScheduler::initialize(const SchedulerConfig& config) {
    try {
        config_ = config;
        
        // Initialize telemetry collector
        telemetry_ = std::make_unique<TelemetryCollector>();
        if (!telemetry_->initialize()) {
            logMessage(LogLevel::ERROR, "Failed to initialize telemetry collector");
            return false;
        }
        
        // Initialize AI predictor if enabled
        if (config.enable_ai_scheduling) {
            predictor_ = std::make_unique<AIPredictor>();
            if (!predictor_->initialize(config.model_path)) {
                logMessage(LogLevel::WARNING, "Failed to initialize AI predictor, falling back to basic scheduling");
                ai_scheduling_enabled_ = false;
            } else {
                ai_scheduling_enabled_ = true;
                logMessage(LogLevel::INFO, "AI-driven scheduling enabled");
            }
        }
        
        // Initialize priority queue
        priority_queue_ = std::make_unique<PriorityQueue>(config.max_queue_size);
        
        // Initialize performance monitor
        monitor_ = std::make_unique<PerformanceMonitor>();
        
        // Initialize multi-GPU scheduler
        multi_gpu_scheduler_ = std::make_unique<MultiGPUScheduler>();
        if (!multi_gpu_scheduler_->initialize()) {
            logMessage(LogLevel::WARNING, "Failed to initialize multi-GPU scheduler, using single GPU mode");
        } else {
            logMessage(LogLevel::INFO, "Multi-GPU scheduler initialized successfully");
        }
        
        // Initialize preemption manager
        preemption_manager_ = std::make_unique<PreemptionManager>();
        if (!preemption_manager_->initialize()) {
            logMessage(LogLevel::WARNING, "Failed to initialize preemption manager, preemption disabled");
        } else {
            logMessage(LogLevel::INFO, "Preemption manager initialized successfully");
        }
        
        logMessage(LogLevel::INFO, "CUDA Scheduler initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        logMessage(LogLevel::ERROR, "Initialization failed: " + std::string(e.what()));
        return false;
    }
}

CUresult CUDAScheduler::scheduleKernel(const KernelLaunchParams& params) {
    try {
        // Register kernel for preemption tracking
        if (preemption_manager_) {
            preemption_manager_->registerKernel(params.kernel_id, Priority::NORMAL, true);
        }
        
        // Record kernel launch in telemetry
        telemetry_->recordKernelLaunch(params);
        
        // Create kernel profile
        KernelProfile profile = telemetry_->createProfile(params);
        
        // Get AI prediction if enabled
        float predicted_time = 0.0f;
        if (ai_scheduling_enabled_ && predictor_) {
            auto prediction = predictor_->predict(profile);
            predicted_time = prediction.predicted_execution_time_ms;
            
            if (log_level_ >= LogLevel::DEBUG) {
                std::stringstream ss;
                ss << "Kernel " << params.kernel_id << " predicted time: " 
                   << std::fixed << std::setprecision(2) << predicted_time << "ms";
                logMessage(LogLevel::DEBUG, ss.str());
            }
        }
        
        // Calculate priority
        Priority priority = calculatePriority(profile, predicted_time);
        
        // Add to priority queue
        KernelTask task{params, priority, predicted_time};
        if (!priority_queue_->enqueue(task)) {
            logMessage(LogLevel::WARNING, "Priority queue full, dropping kernel " + std::to_string(params.kernel_id));
            return CUDA_ERROR_LAUNCH_FAILED;
        }
        
        // Update metrics
        metrics_.total_kernels_launched++;
        metrics_.last_update = std::chrono::high_resolution_clock::now();
        
        // Dispatch next kernel
        return dispatchNextKernel();
        
    } catch (const std::exception& e) {
        logMessage(LogLevel::ERROR, "Kernel scheduling failed: " + std::string(e.what()));
        return CUDA_ERROR_LAUNCH_FAILED;
    }
}

Priority CUDAScheduler::calculatePriority(const KernelProfile& profile, float predicted_time) {
    // Base priority calculation
    float priority_score = 0.0f;
    
    // Workload type priority
    if (profile.operation_type.find("inference") != std::string::npos ||
        profile.operation_type.find("forward") != std::string::npos) {
        priority_score += 10.0f;  // High priority for inference
    } else if (profile.operation_type.find("backward") != std::string::npos) {
        priority_score += 8.0f;   // Medium-high priority for backward pass
    } else if (profile.operation_type.find("optimizer") != std::string::npos) {
        priority_score += 5.0f;   // Medium priority for optimizer
    }
    
    // Latency critical threshold
    if (predicted_time > 0 && predicted_time < config_.latency_critical_threshold_ms) {
        priority_score += 15.0f;  // Critical priority for latency-sensitive kernels
    }
    
    // Resource utilization optimization
    if (config_.throughput_optimization) {
        // Prefer kernels that utilize resources efficiently
        float resource_efficiency = (profile.sm_utilization + profile.memory_bandwidth_gb_s / 1000.0f) / 2.0f;
        priority_score += resource_efficiency * 5.0f;
    }
    
    // Queue depth penalty
    size_t queue_depth = priority_queue_->size();
    priority_score -= queue_depth * 0.1f;
    
    // Convert to priority enum
    if (priority_score >= 20.0f) return Priority::CRITICAL;
    if (priority_score >= 15.0f) return Priority::HIGH;
    if (priority_score >= 8.0f) return Priority::NORMAL;
    return Priority::LOW;
}

CUresult CUDAScheduler::dispatchNextKernel() {
    try {
        KernelTask task;
        if (priority_queue_->dequeue(task)) {
            // Launch the kernel using CUDA runtime
            CUresult result = cudaLaunchKernel(
                task.params.func,
                task.params.grid_dim,
                task.params.block_dim,
                task.params.shared_mem_size,
                task.params.stream,
                task.params.args
            );
            
            if (result == CUDA_SUCCESS) {
                if (log_level_ >= LogLevel::DEBUG) {
                    std::stringstream ss;
                    ss << "Dispatched kernel " << task.params.kernel_id 
                       << " with priority " << static_cast<int>(task.priority);
                    logMessage(LogLevel::DEBUG, ss.str());
                }
            } else {
                logMessage(LogLevel::ERROR, "Failed to launch kernel " + std::to_string(task.params.kernel_id));
            }
            
            return result;
        }
        
        return CUDA_SUCCESS;  // No kernels to dispatch
        
    } catch (const std::exception& e) {
        logMessage(LogLevel::ERROR, "Kernel dispatch failed: " + std::string(e.what()));
        return CUDA_ERROR_LAUNCH_FAILED;
    }
}

void CUDAScheduler::enableAIScheduling(bool enabled) {
    ai_scheduling_enabled_ = enabled;
    logMessage(LogLevel::INFO, "AI scheduling " + std::string(enabled ? "enabled" : "disabled"));
}

void CUDAScheduler::configure(const SchedulerConfig& config) {
    config_ = config;
    
    // Update AI scheduling state
    if (predictor_ && config.enable_ai_scheduling) {
        ai_scheduling_enabled_ = true;
    } else {
        ai_scheduling_enabled_ = false;
    }
    
    // Update priority queue size
    if (priority_queue_) {
        priority_queue_->resize(config.max_queue_size);
    }
    
    logMessage(LogLevel::INFO, "Scheduler configuration updated");
}

PerformanceMetrics CUDAScheduler::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Update metrics from telemetry
    if (telemetry_) {
        auto gpu_metrics = telemetry_->getGPUMetrics();
        metrics_.sm_utilization_percent = gpu_metrics.sm_utilization_percent;
        metrics_.memory_utilization_percent = gpu_metrics.memory_utilization_percent;
    }
    
    // Update queue wait time
    if (priority_queue_) {
        metrics_.avg_queue_wait_ms = priority_queue_->getAverageWaitTime();
    }
    
    // Update prediction accuracy
    if (predictor_) {
        metrics_.prediction_accuracy = predictor_->getAccuracy();
    }
    
    return metrics_;
}

void CUDAScheduler::setLogLevel(LogLevel level) {
    log_level_ = level;
    logMessage(LogLevel::INFO, "Log level set to " + std::to_string(static_cast<int>(level)));
}

void CUDAScheduler::enableProfiling(bool enabled) {
    profiling_enabled_ = enabled;
    if (telemetry_) {
        telemetry_->setCollectionEnabled(enabled);
    }
    logMessage(LogLevel::INFO, "Profiling " + std::string(enabled ? "enabled" : "disabled"));
}

void CUDAScheduler::shutdown() {
    try {
        logMessage(LogLevel::INFO, "Shutting down CUDA Scheduler");
        
        if (telemetry_) {
            telemetry_->shutdown();
        }
        
        if (predictor_) {
            predictor_->shutdown();
        }
        
        if (monitor_) {
            monitor_->shutdown();
        }
        
        if (multi_gpu_scheduler_) {
            multi_gpu_scheduler_->shutdown();
        }
        
        if (preemption_manager_) {
            preemption_manager_->shutdown();
        }
        
        // Clear any remaining tasks
        if (priority_queue_) {
            priority_queue_->clear();
        }
        
        logMessage(LogLevel::INFO, "CUDA Scheduler shutdown complete");
        
    } catch (const std::exception& e) {
        std::cerr << "Error during shutdown: " << e.what() << std::endl;
    }
}

void CUDAScheduler::updateMetrics(const KernelProfile& profile) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_kernels_completed++;
    metrics_.avg_execution_time_ms = 
        (metrics_.avg_execution_time_ms * (metrics_.total_kernels_completed - 1) + 
         profile.execution_time_ns / 1000000.0f) / metrics_.total_kernels_completed;
    
    metrics_.last_update = std::chrono::high_resolution_clock::now();
}

void CUDAScheduler::logMessage(LogLevel level, const std::string& message) const {
    if (level <= log_level_) {
        std::string level_str;
        switch (level) {
            case LogLevel::ERROR: level_str = "ERROR"; break;
            case LogLevel::WARNING: level_str = "WARNING"; break;
            case LogLevel::INFO: level_str = "INFO"; break;
            case LogLevel::DEBUG: level_str = "DEBUG"; break;
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        std::cout << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << level_str << "] CUDA-Scheduler: " << message << std::endl;
    }
}

} // namespace cuda_scheduler 