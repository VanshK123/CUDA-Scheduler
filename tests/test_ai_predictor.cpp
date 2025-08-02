#include <gtest/gtest.h>
#include "cuda_scheduler/ai_predictor.hpp"
#include <memory>
#include <vector>
#include <chrono>

namespace cuda_scheduler {
namespace test {

class AIPredictorTest : public ::testing::Test {
protected:
    void SetUp() override {
        predictor_ = std::make_unique<AIPredictor>();
        model_path_ = "test_model.onnx";
    }

    void TearDown() override {
        if (predictor_) {
            predictor_->shutdown();
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

    std::unique_ptr<AIPredictor> predictor_;
    std::string model_path_;
};

TEST_F(AIPredictorTest, InitializationTest) {
    // Test initialization with different model types
    EXPECT_TRUE(predictor_->initialize(model_path_, ModelType::XGBOOST));
    
    // Test with transformer model
    auto predictor2 = std::make_unique<AIPredictor>();
    EXPECT_TRUE(predictor2->initialize(model_path_, ModelType::TRANSFORMER));
    
    // Test with ensemble model
    auto predictor3 = std::make_unique<AIPredictor>();
    EXPECT_TRUE(predictor3->initialize(model_path_, ModelType::ENSEMBLE));
}

TEST_F(AIPredictorTest, FeatureExtractionTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    KernelProfile profile = createTestProfile();
    FeatureVector features = predictor_->extractFeatures(profile);

    // Test primary features
    EXPECT_EQ(features.grid_x, 100);
    EXPECT_EQ(features.grid_y, 1);
    EXPECT_EQ(features.grid_z, 1);
    EXPECT_EQ(features.block_x, 256);
    EXPECT_EQ(features.block_y, 1);
    EXPECT_EQ(features.block_z, 1);
    EXPECT_EQ(features.shared_mem_kb, 1);  // 1024 bytes = 1 KB

    // Test derived features
    EXPECT_GE(features.arithmetic_intensity, 0.0f);
    EXPECT_GE(features.parallelism_degree, 0.0f);
    EXPECT_GE(features.memory_access_pattern_score, 0.0f);

    // Test hardware context
    EXPECT_GT(features.compute_capability_major, 0);
    EXPECT_GT(features.total_global_memory_gb, 0);
    EXPECT_GT(features.multiprocessor_count, 0);
}

TEST_F(AIPredictorTest, PredictionTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    KernelProfile profile = createTestProfile();
    PredictionResult result = predictor_->predict(profile);

    // Test prediction result
    EXPECT_GE(result.predicted_execution_time_ms, 0.0f);
    EXPECT_GE(result.confidence_score, 0.0f);
    EXPECT_LE(result.confidence_score, 1.0f);
    EXPECT_FALSE(result.model_version.empty());
}

TEST_F(AIPredictorTest, CachingTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    KernelProfile profile = createTestProfile();
    
    // First prediction
    PredictionResult result1 = predictor_->predict(profile);
    
    // Second prediction (should be cached)
    PredictionResult result2 = predictor_->predict(profile);
    
    // Results should be similar (cached)
    EXPECT_NEAR(result1.predicted_execution_time_ms, 
                result2.predicted_execution_time_ms, 0.001f);

    // Test cache statistics
    auto cache_stats = predictor_->getCacheStats();
    EXPECT_GT(cache_stats.cache_size, 0);
    EXPECT_GT(cache_stats.hit_rate, 0.0f);
}

TEST_F(AIPredictorTest, CacheManagementTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Enable caching
    predictor_->enableCaching(true);
    EXPECT_TRUE(predictor_->getCacheStats().cache_size >= 0);

    // Disable caching
    predictor_->enableCaching(false);
    
    // Clear cache
    predictor_->clearCache();
    auto cache_stats = predictor_->getCacheStats();
    EXPECT_EQ(cache_stats.cache_size, 0);
}

TEST_F(AIPredictorTest, ModelUpdateTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Create training data
    std::vector<KernelProfile> profiles;
    std::vector<float> actual_times;
    
    for (int i = 0; i < 10; ++i) {
        KernelProfile profile = createTestProfile();
        profile.kernel_id = utils::generateKernelId();
        profile.actual_execution_time_ms = 1.0f + (i * 0.1f);
        
        profiles.push_back(profile);
        actual_times.push_back(profile.actual_execution_time_ms);
    }

    // Update model
    EXPECT_TRUE(predictor_->updateModel(profiles, actual_times));
    
    // Check accuracy improvement
    float accuracy = predictor_->getAccuracy();
    EXPECT_GE(accuracy, 0.0f);
    EXPECT_LE(accuracy, 100.0f);
}

TEST_F(AIPredictorTest, OnlineLearningTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Simulate online learning with multiple batches
    for (int batch = 0; batch < 3; ++batch) {
        std::vector<KernelProfile> profiles;
        std::vector<float> actual_times;
        
        for (int i = 0; i < 5; ++i) {
            KernelProfile profile = createTestProfile();
            profile.kernel_id = utils::generateKernelId();
            profile.actual_execution_time_ms = 1.0f + (batch * 0.5f) + (i * 0.1f);
            
            profiles.push_back(profile);
            actual_times.push_back(profile.actual_execution_time_ms);
        }

        EXPECT_TRUE(predictor_->updateModel(profiles, actual_times));
    }

    // Check that model has learned
    float final_accuracy = predictor_->getAccuracy();
    EXPECT_GE(final_accuracy, 0.0f);
}

TEST_F(AIPredictorTest, FeatureCalculationTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    KernelProfile profile = createTestProfile();
    FeatureVector features = predictor_->extractFeatures(profile);

    // Test arithmetic intensity calculation
    float arithmetic_intensity = features.arithmetic_intensity;
    EXPECT_GE(arithmetic_intensity, 0.0f);
    EXPECT_LE(arithmetic_intensity, 1000.0f);  // Reasonable upper bound

    // Test parallelism degree calculation
    float parallelism_degree = features.parallelism_degree;
    EXPECT_GE(parallelism_degree, 0.0f);
    EXPECT_LE(parallelism_degree, 1.0f);  // Normalized to [0,1]

    // Test memory access pattern score
    float memory_score = features.memory_access_pattern_score;
    EXPECT_GE(memory_score, 0.0f);
    EXPECT_LE(memory_score, 1.0f);  // Normalized to [0,1]
}

TEST_F(AIPredictorTest, ModelRetrainingTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Create training data for retraining
    std::vector<std::vector<float>> X_train;
    std::vector<float> y_train;
    
    for (int i = 0; i < 20; ++i) {
        std::vector<float> features(20, 0.0f);
        features[0] = static_cast<float>(i);  // grid_x
        features[1] = static_cast<float>(i * 2);  // grid_y
        features[2] = 1.0f;  // grid_z
        features[3] = 256.0f;  // block_x
        features[4] = 1.0f;  // block_y
        features[5] = 1.0f;  // block_z
        features[6] = 1.0f;  // shared_mem_kb
        
        X_train.push_back(features);
        y_train.push_back(1.0f + (i * 0.1f));  // Execution time
    }

    // Test retraining
    EXPECT_TRUE(predictor_->retrainModel(X_train, y_train));
}

TEST_F(AIPredictorTest, PerformanceTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Make many predictions quickly
    for (int i = 0; i < 100; ++i) {
        KernelProfile profile = createTestProfile();
        profile.kernel_id = utils::generateKernelId();
        
        PredictionResult result = predictor_->predict(profile);
        EXPECT_GE(result.predicted_execution_time_ms, 0.0f);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Should complete within reasonable time (less than 1 second)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(AIPredictorTest, ErrorHandlingTest) {
    // Test initialization with invalid model path
    EXPECT_FALSE(predictor_->initialize("nonexistent_model.onnx"));

    // Test with valid model
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Test prediction with invalid profile
    KernelProfile invalid_profile;
    invalid_profile.kernel_id = 0;
    invalid_profile.grid_dim = dim3(0, 0, 0);
    invalid_profile.block_dim = dim3(0, 0, 0);

    PredictionResult result = predictor_->predict(invalid_profile);
    EXPECT_GE(result.predicted_execution_time_ms, 0.0f);  // Should handle gracefully

    // Test model update with mismatched data
    std::vector<KernelProfile> profiles(5);
    std::vector<float> actual_times(3);  // Mismatched size

    EXPECT_FALSE(predictor_->updateModel(profiles, actual_times));
}

TEST_F(AIPredictorTest, ModelVersionTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Test model version tracking
    KernelProfile profile = createTestProfile();
    PredictionResult result = predictor_->predict(profile);
    
    EXPECT_FALSE(result.model_version.empty());
    EXPECT_TRUE(result.model_version.find("retrained_") == std::string::npos);  // Initial version
}

TEST_F(AIPredictorTest, HardwareContextTest) {
    ASSERT_TRUE(predictor_->initialize(model_path_));

    // Test hardware context initialization
    KernelProfile profile = createTestProfile();
    FeatureVector features = predictor_->extractFeatures(profile);

    // Hardware context should be properly initialized
    EXPECT_GT(features.compute_capability_major, 0);
    EXPECT_GT(features.compute_capability_minor, 0);
    EXPECT_GT(features.total_global_memory_gb, 0);
    EXPECT_GT(features.multiprocessor_count, 0);
}

} // namespace test
} // namespace cuda_scheduler

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 