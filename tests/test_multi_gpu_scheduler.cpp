#include <gtest/gtest.h>
#include "cuda_scheduler/multi_gpu_scheduler.hpp"
#include <memory>
#include <vector>

namespace cuda_scheduler {
namespace test {

class MultiGPUSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        scheduler_ = std::make_unique<MultiGPUScheduler>();
    }

    void TearDown() override {
        if (scheduler_) {
            scheduler_->shutdown();
        }
    }

    KernelProfile createTestProfile() {
        KernelProfile profile;
        profile.kernel_id = utils::generateKernelId();
        profile.grid_dim = dim3(100, 1, 1);
        profile.block_dim = dim3(256, 1, 1);
        profile.shared_mem_size = 1024;
        profile.operation_type = "vector_add";
        profile.predicted_execution_time_ms = 1.5f;
        profile.actual_execution_time_ms = 1.6f;
        return profile;
    }

    std::unique_ptr<MultiGPUScheduler> scheduler_;
};

TEST_F(MultiGPUSchedulerTest, InitializationTest) {
    // Test initialization with different strategies
    EXPECT_TRUE(scheduler_->initialize(MultiGPUStrategy::ROUND_ROBIN));
    
    auto scheduler2 = std::make_unique<MultiGPUScheduler>();
    EXPECT_TRUE(scheduler2->initialize(MultiGPUStrategy::LOAD_BALANCED));
    
    auto scheduler3 = std::make_unique<MultiGPUScheduler>();
    EXPECT_TRUE(scheduler3->initialize(MultiGPUStrategy::MEMORY_AWARE));
    
    auto scheduler4 = std::make_unique<MultiGPUScheduler>();
    EXPECT_TRUE(scheduler4->initialize(MultiGPUStrategy::AFFINITY_BASED));
    
    auto scheduler5 = std::make_unique<MultiGPUScheduler>();
    EXPECT_TRUE(scheduler5->initialize(MultiGPUStrategy::PERFORMANCE_OPTIMIZED));
}

TEST_F(MultiGPUSchedulerTest, DeviceDiscoveryTest) {
    ASSERT_TRUE(scheduler_->initialize());

    auto devices = scheduler_->getAvailableDevices();
    
    // Should find at least one GPU
    EXPECT_GT(devices.size(), 0);
    
    // Test device properties
    for (const auto& device : devices) {
        EXPECT_GE(device.device_id, 0);
        EXPECT_GT(device.total_memory_gb, 0);
        EXPECT_GT(device.multiprocessor_count, 0);
        EXPECT_GT(device.compute_capability_major, 0);
        EXPECT_TRUE(device.is_available);
    }
}

TEST_F(MultiGPUSchedulerTest, GPUSelectionTest) {
    ASSERT_TRUE(scheduler_->initialize());

    KernelProfile profile = createTestProfile();
    float predicted_time = 1.5f;

    // Test GPU selection
    int selected_gpu = scheduler_->selectOptimalGPU(profile, predicted_time);
    EXPECT_GE(selected_gpu, 0);

    // Test with different strategies
    scheduler_->setStrategy(MultiGPUStrategy::ROUND_ROBIN);
    int gpu1 = scheduler_->selectOptimalGPU(profile, predicted_time);
    
    scheduler_->setStrategy(MultiGPUStrategy::LOAD_BALANCED);
    int gpu2 = scheduler_->selectOptimalGPU(profile, predicted_time);
    
    scheduler_->setStrategy(MultiGPUStrategy::MEMORY_AWARE);
    int gpu3 = scheduler_->selectOptimalGPU(profile, predicted_time);
    
    // All should be valid GPU indices
    EXPECT_GE(gpu1, 0);
    EXPECT_GE(gpu2, 0);
    EXPECT_GE(gpu3, 0);
}

TEST_F(MultiGPUSchedulerTest, KernelSchedulingTest) {
    ASSERT_TRUE(scheduler_->initialize());

    // Create kernel launch parameters
    KernelLaunchParams params;
    params.kernel_id = utils::generateKernelId();
    params.func = nullptr;  // Mock function pointer
    params.grid_dim = dim3(100, 1, 1);
    params.block_dim = dim3(256, 1, 1);
    params.shared_mem_size = 1024;
    params.stream = 0;
    params.args = nullptr;
    params.launch_time = utils::getCurrentTime();

    // Schedule kernel
    CUresult result = scheduler_->scheduleKernelMultiGPU(params);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(MultiGPUSchedulerTest, LoadBalancingTest) {
    ASSERT_TRUE(scheduler_->initialize(MultiGPUStrategy::LOAD_BALANCED));

    // Schedule multiple kernels to test load balancing
    for (int i = 0; i < 10; ++i) {
        KernelProfile profile = createTestProfile();
        profile.kernel_id = utils::generateKernelId();
        
        int selected_gpu = scheduler_->selectOptimalGPU(profile, 1.5f);
        EXPECT_GE(selected_gpu, 0);
    }

    // Get load balancing statistics
    auto stats = scheduler_->getLoadBalancingStats();
    EXPECT_GT(stats.total_kernels_scheduled, 0);
    EXPECT_GE(stats.load_balance_score, 0.0f);
    EXPECT_LE(stats.load_balance_score, 1.0f);
}

TEST_F(MultiGPUSchedulerTest, DeviceManagementTest) {
    ASSERT_TRUE(scheduler_->initialize());

    auto devices = scheduler_->getAvailableDevices();
    if (devices.size() > 1) {
        // Test disabling a device
        scheduler_->setDeviceEnabled(1, false);
        
        // Update device info
        GPUDevice device_info = devices[1];
        device_info.is_available = false;
        scheduler_->updateDeviceInfo(1, device_info);
        
        // Test enabling a device
        scheduler_->setDeviceEnabled(1, true);
        device_info.is_available = true;
        scheduler_->updateDeviceInfo(1, device_info);
    }
}

TEST_F(MultiGPUSchedulerTest, StrategySwitchingTest) {
    ASSERT_TRUE(scheduler_->initialize());

    KernelProfile profile = createTestProfile();
    float predicted_time = 1.5f;

    // Test all strategies
    std::vector<MultiGPUStrategy> strategies = {
        MultiGPUStrategy::ROUND_ROBIN,
        MultiGPUStrategy::LOAD_BALANCED,
        MultiGPUStrategy::MEMORY_AWARE,
        MultiGPUStrategy::AFFINITY_BASED,
        MultiGPUStrategy::PERFORMANCE_OPTIMIZED
    };

    for (auto strategy : strategies) {
        scheduler_->setStrategy(strategy);
        int selected_gpu = scheduler_->selectOptimalGPU(profile, predicted_time);
        EXPECT_GE(selected_gpu, 0);
    }
}

TEST_F(MultiGPUSchedulerTest, PerformanceMetricsTest) {
    ASSERT_TRUE(scheduler_->initialize());

    // Schedule some kernels
    for (int i = 0; i < 5; ++i) {
        KernelLaunchParams params;
        params.kernel_id = utils::generateKernelId();
        params.grid_dim = dim3(50, 1, 1);
        params.block_dim = dim3(128, 1, 1);
        params.shared_mem_size = 512;
        params.stream = 0;
        params.args = nullptr;
        params.launch_time = utils::getCurrentTime();

        scheduler_->scheduleKernelMultiGPU(params);
    }

    // Get performance metrics
    auto metrics = scheduler_->getMultiGPUMetrics();
    EXPECT_GT(metrics.size(), 0);
    
    for (const auto& metric : metrics) {
        EXPECT_GE(metric.total_kernels_scheduled, 0);
        EXPECT_GE(metric.avg_execution_time_ms, 0.0f);
        EXPECT_GE(metric.sm_utilization_percent, 0.0f);
        EXPECT_LE(metric.sm_utilization_percent, 100.0f);
    }
}

TEST_F(MultiGPUSchedulerTest, MemoryAwareSchedulingTest) {
    ASSERT_TRUE(scheduler_->initialize(MultiGPUStrategy::MEMORY_AWARE));

    // Create memory-intensive kernel profile
    KernelProfile profile = createTestProfile();
    profile.shared_mem_size = 16384;  // 16KB shared memory
    profile.grid_dim = dim3(1000, 1000, 1);  // Large grid
    profile.block_dim = dim3(256, 1, 1);

    int selected_gpu = scheduler_->selectOptimalGPU(profile, 5.0f);
    EXPECT_GE(selected_gpu, 0);
}

TEST_F(MultiGPUSchedulerTest, AffinityBasedSchedulingTest) {
    ASSERT_TRUE(scheduler_->initialize(MultiGPUStrategy::AFFINITY_BASED));

    // Schedule related kernels
    for (int i = 0; i < 3; ++i) {
        KernelProfile profile = createTestProfile();
        profile.kernel_id = utils::generateKernelId();
        profile.operation_type = "matrix_multiply";  // Same operation type
        
        int selected_gpu = scheduler_->selectOptimalGPU(profile, 2.0f);
        EXPECT_GE(selected_gpu, 0);
    }
}

TEST_F(MultiGPUSchedulerTest, PerformanceOptimizedSchedulingTest) {
    ASSERT_TRUE(scheduler_->initialize(MultiGPUStrategy::PERFORMANCE_OPTIMIZED));

    // Create compute-intensive kernel profile
    KernelProfile profile = createTestProfile();
    profile.operation_type = "convolution";
    profile.grid_dim = dim3(500, 500, 1);
    profile.block_dim = dim3(16, 16, 1);
    profile.shared_mem_size = 8192;  // 8KB shared memory

    int selected_gpu = scheduler_->selectOptimalGPU(profile, 10.0f);
    EXPECT_GE(selected_gpu, 0);
}

TEST_F(MultiGPUSchedulerTest, ErrorHandlingTest) {
    ASSERT_TRUE(scheduler_->initialize());

    // Test with invalid kernel parameters
    KernelLaunchParams invalid_params;
    invalid_params.kernel_id = 0;
    invalid_params.func = nullptr;
    invalid_params.grid_dim = dim3(0, 0, 0);
    invalid_params.block_dim = dim3(0, 0, 0);

    CUresult result = scheduler_->scheduleKernelMultiGPU(invalid_params);
    EXPECT_NE(result, CUDA_SUCCESS);
}

TEST_F(MultiGPUSchedulerTest, LoadBalanceScoreTest) {
    ASSERT_TRUE(scheduler_->initialize());

    // Get initial load balance score
    auto initial_stats = scheduler_->getLoadBalancingStats();
    float initial_score = initial_stats.load_balance_score;

    // Schedule kernels to create load imbalance
    for (int i = 0; i < 20; ++i) {
        KernelProfile profile = createTestProfile();
        profile.kernel_id = utils::generateKernelId();
        
        scheduler_->selectOptimalGPU(profile, 1.0f + (i * 0.1f));
    }

    // Get updated load balance score
    auto updated_stats = scheduler_->getLoadBalancingStats();
    float updated_score = updated_stats.load_balance_score;

    // Scores should be valid
    EXPECT_GE(initial_score, 0.0f);
    EXPECT_LE(initial_score, 1.0f);
    EXPECT_GE(updated_score, 0.0f);
    EXPECT_LE(updated_score, 1.0f);
}

TEST_F(MultiGPUSchedulerTest, ShutdownTest) {
    ASSERT_TRUE(scheduler_->initialize());

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

        scheduler_->scheduleKernelMultiGPU(params);
    }

    // Shutdown should complete successfully
    EXPECT_NO_THROW(scheduler_->shutdown());
}

} // namespace test
} // namespace cuda_scheduler

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 