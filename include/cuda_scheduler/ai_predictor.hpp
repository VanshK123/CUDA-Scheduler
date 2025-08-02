#pragma once

#include "scheduler.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace cuda_scheduler {

/**
 * @brief Feature vector for ML model input
 */
struct FeatureVector {
    // Primary features
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem_kb;
    uint64_t input_tensor_volume;
    float operation_complexity_score;
    
    // Derived features
    float arithmetic_intensity;
    float parallelism_degree;
    float memory_access_pattern_score;
    
    // Historical features
    float recent_avg_execution_time_ms;
    float queue_depth_penalty;
    
    // Hardware context
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_global_memory_gb;
    int multiprocessor_count;
    
    FeatureVector() : grid_x(0), grid_y(0), grid_z(0), block_x(0), block_y(0), block_z(0),
                     shared_mem_kb(0), input_tensor_volume(0), operation_complexity_score(0.0f),
                     arithmetic_intensity(0.0f), parallelism_degree(0.0f), memory_access_pattern_score(0.0f),
                     recent_avg_execution_time_ms(0.0f), queue_depth_penalty(0.0f),
                     compute_capability_major(0), compute_capability_minor(0), total_global_memory_gb(0), multiprocessor_count(0) {}
};

/**
 * @brief Prediction result structure
 */
struct PredictionResult {
    float predicted_execution_time_ms;
    float confidence_score;
    std::string model_version;
    std::chrono::high_resolution_clock::time_point prediction_time;
    
    PredictionResult() : predicted_execution_time_ms(0.0f), confidence_score(0.0f) {}
};

/**
 * @brief Model types supported by the predictor
 */
enum class ModelType {
    XGBOOST = 0,
    TRANSFORMER = 1,
    ENSEMBLE = 2
};

/**
 * @brief AI Predictor for kernel execution time prediction
 * 
 * This class provides:
 * 1. Feature extraction from kernel profiles
 * 2. ML model inference for execution time prediction
 * 3. Prediction caching for performance optimization
 * 4. Model versioning and updates
 */
class AIPredictor {
public:
    /**
     * @brief Constructor
     */
    AIPredictor();
    
    /**
     * @brief Destructor
     */
    ~AIPredictor();
    
    /**
     * @brief Initialize the predictor
     * @param model_path Path to the ONNX model file
     * @param model_type Type of model to use
     * @return true if initialization successful
     */
    bool initialize(const std::string& model_path, ModelType model_type = ModelType::XGBOOST);
    
    /**
     * @brief Predict execution time for a kernel
     * @param profile Kernel profile
     * @return Prediction result
     */
    PredictionResult predict(const KernelProfile& profile);
    
    /**
     * @brief Extract features from kernel profile
     * @param profile Kernel profile
     * @return Feature vector
     */
    FeatureVector extractFeatures(const KernelProfile& profile);
    
    /**
     * @brief Update the model with new training data
     * @param profiles Vector of kernel profiles
     * @param actual_times Vector of actual execution times
     * @return true if update successful
     */
    bool updateModel(const std::vector<KernelProfile>& profiles, 
                    const std::vector<float>& actual_times);
    
    /**
     * @brief Get prediction accuracy
     * @return Accuracy percentage
     */
    float getAccuracy() const;
    
    /**
     * @brief Enable or disable prediction caching
     * @param enabled Whether to enable caching
     */
    void enableCaching(bool enabled);
    
    /**
     * @brief Clear prediction cache
     */
    void clearCache();
    
    /**
     * @brief Get cache statistics
     * @return Cache hit rate and size
     */
    struct CacheStats {
        float hit_rate;
        size_t cache_size;
        size_t max_cache_size;
    };
    
    CacheStats getCacheStats() const;
    
    /**
     * @brief Shutdown the predictor
     */
    void shutdown();

private:
    // Model state
    std::unique_ptr<class ONNXModel> onnx_model_;
    ModelType model_type_;
    std::string model_path_;
    std::string model_version_;
    
    // Prediction cache
    std::unordered_map<size_t, PredictionResult> prediction_cache_;
    std::atomic<bool> caching_enabled_{true};
    size_t max_cache_size_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    float prediction_accuracy_;
    size_t total_predictions_;
    size_t cache_hits_;
    
    // Feature extraction helpers
    float calculateArithmeticIntensity(const KernelProfile& profile);
    float calculateParallelismDegree(const KernelProfile& profile);
    float calculateMemoryAccessPatternScore(const KernelProfile& profile);
    float calculateOperationComplexityScore(const KernelProfile& profile);
    
    // Cache management
    size_t computeFeatureHash(const FeatureVector& features);
    void trimCache();
    
    // Model inference
    std::vector<float> featuresToInput(const FeatureVector& features);
    PredictionResult runInference(const std::vector<float>& input);
    
    // Hardware context
    struct HardwareContext {
        int compute_capability_major;
        int compute_capability_minor;
        size_t total_global_memory_gb;
        int multiprocessor_count;
    };
    
    HardwareContext hardware_context_;
    bool initializeHardwareContext();
};

} // namespace cuda_scheduler 