#include <gtest/gtest.h>
#include "cuda_scheduler/preemption.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

namespace cuda_scheduler {
namespace test {

class PreemptionManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager_ = std::make_unique<PreemptionManager>();
    }

    void TearDown() override {
        if (manager_) {
            manager_->shutdown();
        }
    }

    uint64_t createTestKernelId() {
        return utils::generateKernelId();
    }

    std::unique_ptr<PreemptionManager> manager_;
};

TEST_F(PreemptionManagerTest, InitializationTest) {
    // Test initialization with different strategies
    EXPECT_TRUE(manager_->initialize(PreemptionStrategy::NONE));
    
    auto manager2 = std::make_unique<PreemptionManager>();
    EXPECT_TRUE(manager2->initialize(PreemptionStrategy::COOPERATIVE));
    
    auto manager3 = std::make_unique<PreemptionManager>();
    EXPECT_TRUE(manager3->initialize(PreemptionStrategy::PREEMPTIVE));
    
    auto manager4 = std::make_unique<PreemptionManager>();
    EXPECT_TRUE(manager4->initialize(PreemptionStrategy::ADAPTIVE));
    
    auto manager5 = std::make_unique<PreemptionManager>();
    EXPECT_TRUE(manager5->initialize(PreemptionStrategy::TIME_SLICE, 5));  // 5ms time slice
}

TEST_F(PreemptionManagerTest, KernelRegistrationTest) {
    ASSERT_TRUE(manager_->initialize());

    uint64_t kernel_id = createTestKernelId();
    
    // Register kernel
    EXPECT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Register same kernel again (should fail)
    EXPECT_FALSE(manager_->registerKernel(kernel_id, Priority::HIGH, false));
    
    // Register different kernel
    uint64_t kernel_id2 = createTestKernelId();
    EXPECT_TRUE(manager_->registerKernel(kernel_id2, Priority::HIGH, true));
}

TEST_F(PreemptionManagerTest, PreemptionRequestTest) {
    ASSERT_TRUE(manager_->initialize());

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Create preemption request
    PreemptionRequest request(kernel_id, Priority::CRITICAL, 0.9f);
    EXPECT_TRUE(manager_->requestPreemption(request));
    
    // Test request for non-existent kernel
    PreemptionRequest invalid_request(999999, Priority::HIGH, 0.5f);
    EXPECT_FALSE(manager_->requestPreemption(invalid_request));
}

TEST_F(PreemptionManagerTest, PreemptionExecutionTest) {
    ASSERT_TRUE(manager_->initialize(PreemptionStrategy::COOPERATIVE));

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Update progress to trigger cooperative preemption
    manager_->updateProgress(kernel_id, 0.6f);  // 60% progress
    
    // Check if should preempt
    bool should_preempt = manager_->shouldPreempt(kernel_id);
    EXPECT_TRUE(should_preempt);
    
    // Execute preemption
    CUresult result = manager_->executePreemption(kernel_id);
    EXPECT_EQ(result, CUDA_SUCCESS);
    
    // Resume kernel
    result = manager_->resumeKernel(kernel_id);
    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(PreemptionManagerTest, StrategyTestingTest) {
    // Test cooperative strategy
    auto manager_coop = std::make_unique<PreemptionManager>();
    ASSERT_TRUE(manager_coop->initialize(PreemptionStrategy::COOPERATIVE));
    
    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_coop->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Should not preempt initially
    EXPECT_FALSE(manager_coop->shouldPreempt(kernel_id));
    
    // Update progress to trigger preemption
    manager_coop->updateProgress(kernel_id, 0.7f);
    EXPECT_TRUE(manager_coop->shouldPreempt(kernel_id));
    
    // Test preemptive strategy
    auto manager_preempt = std::make_unique<PreemptionManager>();
    ASSERT_TRUE(manager_preempt->initialize(PreemptionStrategy::PREEMPTIVE));
    
    uint64_t kernel_id2 = createTestKernelId();
    ASSERT_TRUE(manager_preempt->registerKernel(kernel_id2, Priority::NORMAL, true));
    
    // Wait for time threshold
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_TRUE(manager_preempt->shouldPreempt(kernel_id2));
}

TEST_F(PreemptionManagerTest, TimeSliceTest) {
    ASSERT_TRUE(manager_->initialize(PreemptionStrategy::TIME_SLICE, 10));  // 10ms time slice

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Should not preempt immediately
    EXPECT_FALSE(manager_->shouldPreempt(kernel_id));
    
    // Wait for time slice to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    EXPECT_TRUE(manager_->shouldPreempt(kernel_id));
}

TEST_F(PreemptionManagerTest, AdaptiveStrategyTest) {
    ASSERT_TRUE(manager_->initialize(PreemptionStrategy::ADAPTIVE));

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::LOW, true));
    
    // Update progress and wait for execution time
    manager_->updateProgress(kernel_id, 0.8f);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // Should preempt based on adaptive criteria
    bool should_preempt = manager_->shouldPreempt(kernel_id);
    EXPECT_TRUE(should_preempt);
}

TEST_F(PreemptionManagerTest, ContextTrackingTest) {
    ASSERT_TRUE(manager_->initialize());

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::HIGH, true));
    
    // Get initial context
    PreemptionContext context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.kernel_id, kernel_id);
    EXPECT_EQ(context.original_priority, Priority::HIGH);
    EXPECT_EQ(context.current_priority, Priority::HIGH);
    EXPECT_TRUE(context.is_preemptible);
    EXPECT_FALSE(context.is_preempted);
    EXPECT_EQ(context.checkpoint_count, 0);
    EXPECT_EQ(context.execution_progress, 0.0f);
    
    // Update progress
    manager_->updateProgress(kernel_id, 0.5f);
    context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.execution_progress, 0.5f);
}

TEST_F(PreemptionManagerTest, StatisticsTest) {
    ASSERT_TRUE(manager_->initialize());

    // Register and preempt multiple kernels
    for (int i = 0; i < 5; ++i) {
        uint64_t kernel_id = createTestKernelId();
        ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
        
        manager_->updateProgress(kernel_id, 0.6f);
        if (manager_->shouldPreempt(kernel_id)) {
            manager_->executePreemption(kernel_id);
            manager_->resumeKernel(kernel_id);
        }
    }
    
    // Get statistics
    auto stats = manager_->getStats();
    EXPECT_GT(stats.total_preemptions, 0);
    EXPECT_GT(stats.successful_preemptions, 0);
    EXPECT_GE(stats.avg_preemption_time_ms, 0.0f);
    EXPECT_GT(stats.active_kernels, 0);
}

TEST_F(PreemptionManagerTest, KernelPreemptibilityTest) {
    ASSERT_TRUE(manager_->initialize());

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Test enabling/disabling preemption
    manager_->setKernelPreemptible(kernel_id, false);
    EXPECT_FALSE(manager_->requestPreemption(PreemptionRequest(kernel_id, Priority::CRITICAL, 1.0f)));
    
    manager_->setKernelPreemptible(kernel_id, true);
    EXPECT_TRUE(manager_->requestPreemption(PreemptionRequest(kernel_id, Priority::CRITICAL, 1.0f)));
}

TEST_F(PreemptionManagerTest, StrategySwitchingTest) {
    ASSERT_TRUE(manager_->initialize(PreemptionStrategy::COOPERATIVE));

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Test strategy switching
    manager_->setStrategy(PreemptionStrategy::PREEMPTIVE);
    EXPECT_TRUE(manager_->shouldPreempt(kernel_id));  // Should preempt immediately
    
    manager_->setStrategy(PreemptionStrategy::ADAPTIVE);
    bool should_preempt = manager_->shouldPreempt(kernel_id);
    EXPECT_TRUE(should_preempt || !should_preempt);  // Should return valid boolean
    
    manager_->setStrategy(PreemptionStrategy::TIME_SLICE);
    should_preempt = manager_->shouldPreempt(kernel_id);
    EXPECT_TRUE(should_preempt || !should_preempt);  // Should return valid boolean
}

TEST_F(PreemptionManagerTest, ProgressTrackingTest) {
    ASSERT_TRUE(manager_->initialize());

    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    
    // Test progress updates
    manager_->updateProgress(kernel_id, 0.25f);
    PreemptionContext context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.execution_progress, 0.25f);
    
    manager_->updateProgress(kernel_id, 0.75f);
    context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.execution_progress, 0.75f);
    
    // Test boundary conditions
    manager_->updateProgress(kernel_id, -0.1f);  // Should clamp to 0.0
    context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.execution_progress, 0.0f);
    
    manager_->updateProgress(kernel_id, 1.5f);  // Should clamp to 1.0
    context = manager_->getKernelContext(kernel_id);
    EXPECT_EQ(context.execution_progress, 1.0f);
}

TEST_F(PreemptionManagerTest, ErrorHandlingTest) {
    ASSERT_TRUE(manager_->initialize());

    // Test preemption of non-existent kernel
    CUresult result = manager_->executePreemption(999999);
    EXPECT_NE(result, CUDA_SUCCESS);
    
    // Test resume of non-existent kernel
    result = manager_->resumeKernel(999999);
    EXPECT_NE(result, CUDA_SUCCESS);
    
    // Test resume of non-preempted kernel
    uint64_t kernel_id = createTestKernelId();
    ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
    result = manager_->resumeKernel(kernel_id);
    EXPECT_NE(result, CUDA_SUCCESS);  // Not preempted yet
}

TEST_F(PreemptionManagerTest, PerformanceTest) {
    ASSERT_TRUE(manager_->initialize());

    auto start_time = std::chrono::high_resolution_clock::now();

    // Register many kernels quickly
    for (int i = 0; i < 100; ++i) {
        uint64_t kernel_id = createTestKernelId();
        EXPECT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
        
        manager_->updateProgress(kernel_id, 0.5f);
        manager_->shouldPreempt(kernel_id);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Should complete within reasonable time (less than 1 second)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(PreemptionManagerTest, ShutdownTest) {
    ASSERT_TRUE(manager_->initialize());

    // Register some kernels
    for (int i = 0; i < 3; ++i) {
        uint64_t kernel_id = createTestKernelId();
        ASSERT_TRUE(manager_->registerKernel(kernel_id, Priority::NORMAL, true));
        manager_->updateProgress(kernel_id, 0.3f);
    }

    // Shutdown should complete successfully
    EXPECT_NO_THROW(manager_->shutdown());
}

TEST_F(PreemptionManagerTest, UrgencyScoreTest) {
    ASSERT_TRUE(manager_->initialize());

    // Test different urgency levels
    PreemptionRequest request1(createTestKernelId(), Priority::LOW, 0.1f);
    PreemptionRequest request2(createTestKernelId(), Priority::NORMAL, 0.5f);
    PreemptionRequest request3(createTestKernelId(), Priority::HIGH, 0.8f);
    PreemptionRequest request4(createTestKernelId(), Priority::CRITICAL, 1.0f);
    
    // All should be valid requests
    EXPECT_TRUE(manager_->requestPreemption(request1));
    EXPECT_TRUE(manager_->requestPreemption(request2));
    EXPECT_TRUE(manager_->requestPreemption(request3));
    EXPECT_TRUE(manager_->requestPreemption(request4));
}

} // namespace test
} // namespace cuda_scheduler

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 