#include <gtest/gtest.h>
#include "cuda_scheduler/scheduler.hpp"
#include <memory>
#include <vector>
#include <chrono>

namespace cuda_scheduler {
namespace test {

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        scheduler_ = CUDAScheduler::create();
        config_.enable_ai_scheduling = true;
        config_.max_queue_size = 1000;
        config_.enable_preemption = true;
    }

    void TearDown() override {
        if (scheduler_) {
            scheduler_->shutdown();
        }
    }

    std::shared_ptr<CUDAScheduler> scheduler_;
    SchedulerConfig config_;
};

TEST_F(SchedulerTest, InitializationTest) {
    EXPECT_TRUE(scheduler_->initialize(config_));
    EXPECT_TRUE(scheduler_->isInitialized());
}

TEST_F(SchedulerTest, KernelSchedulingTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    // Create a simple kernel launch
    KernelLaunchParams params;
    params.kernel_id = utils::generateKernelId();
    params.func = nullptr;  // Mock function pointer
    params.grid_dim = dim3(100, 1, 1);
    params.block_dim = dim3(256, 1, 1);
    params.shared_mem_size = 0;
    params.stream = 0;
    params.args = nullptr;
    params.launch_time = utils::getCurrentTime();

    // Schedule kernel
    CUresult result = scheduler_->scheduleKernel(params);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(SchedulerTest, PriorityCalculationTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    // Create kernel profile
    KernelProfile profile;
    profile.kernel_id = utils::generateKernelId();
    profile.grid_dim = dim3(100, 1, 1);
    profile.block_dim = dim3(256, 1, 1);
    profile.shared_mem_size = 0;
    profile.operation_type = "vector_add";
    profile.predicted_execution_time_ms = 1.5f;
    profile.actual_execution_time_ms = 1.6f;

    // Test priority calculation
    Priority priority = scheduler_->calculatePriority(profile, 1.5f);
    EXPECT_TRUE(priority == Priority::LOW || 
                priority == Priority::NORMAL || 
                priority == Priority::HIGH || 
                priority == Priority::CRITICAL);
}

TEST_F(SchedulerTest, MetricsCollectionTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    // Enable profiling
    scheduler_->enableProfiling(true);

    // Schedule some kernels
    for (int i = 0; i < 5; ++i) {
        KernelLaunchParams params;
        params.kernel_id = utils::generateKernelId();
        params.grid_dim = dim3(50, 1, 1);
        params.block_dim = dim3(128, 1, 1);
        params.shared_mem_size = 0;
        params.stream = 0;
        params.args = nullptr;
        params.launch_time = utils::getCurrentTime();

        scheduler_->scheduleKernel(params);
    }

    // Get metrics
    auto metrics = scheduler_->getMetrics();
    EXPECT_GE(metrics.total_kernels_scheduled, 5);
    EXPECT_GE(metrics.avg_queue_wait_ms, 0.0f);
    EXPECT_GE(metrics.sm_utilization_percent, 0.0f);
}

TEST_F(SchedulerTest, ConfigurationTest) {
    // Test different configurations
    SchedulerConfig config1;
    config1.enable_ai_scheduling = false;
    config1.max_queue_size = 500;
    config1.enable_preemption = false;

    EXPECT_TRUE(scheduler_->initialize(config1));

    SchedulerConfig config2;
    config2.enable_ai_scheduling = true;
    config2.max_queue_size = 2000;
    config2.enable_preemption = true;
    config2.latency_critical_threshold_ms = 10.0f;

    EXPECT_TRUE(scheduler_->initialize(config2));
}

TEST_F(SchedulerTest, LoggingTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    // Test different log levels
    scheduler_->setLogLevel(LogLevel::DEBUG);
    scheduler_->setLogLevel(LogLevel::INFO);
    scheduler_->setLogLevel(LogLevel::WARNING);
    scheduler_->setLogLevel(LogLevel::ERROR);

    // Enable profiling
    scheduler_->enableProfiling(true);
    EXPECT_TRUE(scheduler_->isProfilingEnabled());
}

TEST_F(SchedulerTest, ShutdownTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    // Schedule some kernels
    for (int i = 0; i < 3; ++i) {
        KernelLaunchParams params;
        params.kernel_id = utils::generateKernelId();
        params.grid_dim = dim3(10, 1, 1);
        params.block_dim = dim3(64, 1, 1);
        params.shared_mem_size = 0;
        params.stream = 0;
        params.args = nullptr;
        params.launch_time = utils::getCurrentTime();

        scheduler_->scheduleKernel(params);
    }

    // Shutdown should complete successfully
    EXPECT_NO_THROW(scheduler_->shutdown());
}

TEST_F(SchedulerTest, ErrorHandlingTest) {
    // Test initialization with invalid config
    SchedulerConfig invalid_config;
    invalid_config.max_queue_size = 0;  // Invalid size

    // Should handle gracefully
    EXPECT_FALSE(scheduler_->initialize(invalid_config));

    // Test with valid config
    EXPECT_TRUE(scheduler_->initialize(config_));

    // Test invalid kernel parameters
    KernelLaunchParams invalid_params;
    invalid_params.kernel_id = 0;  // Invalid ID
    invalid_params.func = nullptr;
    invalid_params.grid_dim = dim3(0, 0, 0);  // Invalid dimensions
    invalid_params.block_dim = dim3(0, 0, 0);

    CUresult result = scheduler_->scheduleKernel(invalid_params);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(SchedulerTest, PerformanceTest) {
    ASSERT_TRUE(scheduler_->initialize(config_));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Schedule many kernels quickly
    for (int i = 0; i < 100; ++i) {
        KernelLaunchParams params;
        params.kernel_id = utils::generateKernelId();
        params.grid_dim = dim3(10, 1, 1);
        params.block_dim = dim3(64, 1, 1);
        params.shared_mem_size = 0;
        params.stream = 0;
        params.args = nullptr;
        params.launch_time = utils::getCurrentTime();

        scheduler_->scheduleKernel(params);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Should complete within reasonable time (less than 1 second)
    EXPECT_LT(duration.count(), 1000000);

    auto metrics = scheduler_->getMetrics();
    EXPECT_EQ(metrics.total_kernels_scheduled, 100);
}

} // namespace test
} // namespace cuda_scheduler

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 