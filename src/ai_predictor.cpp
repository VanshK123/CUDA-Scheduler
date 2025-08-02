#include "cuda_scheduler/ai_predictor.hpp"
#include "cuda_scheduler/onnx_model.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

namespace cuda_scheduler {

AIPredictor::AIPredictor() 
    : model_type_(ModelType::XGBOOST)
    , max_cache_size_(10000)
    , prediction_accuracy_(0.0f)
    , total_predictions_(0)
    , cache_hits_(0) {
}

AIPredictor::~AIPredictor() {
    shutdown();
}

bool AIPredictor::initialize(const std::string& model_path, ModelType model_type) {
    try {
        model_type_ = model_type;
        model_path_ = model_path;
        
        // Initialize hardware context
        if (!initializeHardwareContext()) {
            std::cerr << "Failed to initialize hardware context" << std::endl;
            return false;
        }
        
        // Load ONNX model
        onnx_model_ = std::make_unique<ONNXModel>();
        if (!onnx_model_->loadModel(model_path)) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return false;
        }
        
        // Initialize prediction cache
        prediction_cache_.clear();
        caching_enabled_ = true;
        
        std::cout << "AI Predictor initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "AI Predictor initialization failed: " << e.what() << std::endl;
        return false;
    }
}

PredictionResult AIPredictor::predict(const KernelProfile& profile) {
    PredictionResult result;
    result.prediction_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Extract features from kernel profile
        FeatureVector features = extractFeatures(profile);
        
        // Check cache first
        size_t feature_hash = computeFeatureHash(features);
        auto cache_it = prediction_cache_.find(feature_hash);
        
        if (caching_enabled_ && cache_it != prediction_cache_.end()) {
            // Cache hit
            cache_hits_++;
            result = cache_it->second;
            result.prediction_time = std::chrono::high_resolution_clock::now();
            return result;
        }
        
        // Convert features to model input
        std::vector<float> model_input = featuresToInput(features);
        
        // Run model inference
        result = runInference(model_input);
        result.model_version = onnx_model_->getModelInfo().model_version;
        
        // Cache the result
        if (caching_enabled_) {
            if (prediction_cache_.size() >= max_cache_size_) {
                trimCache();
            }
            prediction_cache_[feature_hash] = result;
        }
        
        // Update statistics
        total_predictions_++;
        
    } catch (const std::exception& e) {
        std::cerr << "Prediction failed: " << e.what() << std::endl;
        result.predicted_execution_time_ms = 0.0f;
        result.confidence_score = 0.0f;
    }
    
    return result;
}

FeatureVector AIPredictor::extractFeatures(const KernelProfile& profile) {
    FeatureVector features;
    
    // Primary features
    features.grid_x = profile.grid_dim.x;
    features.grid_y = profile.grid_dim.y;
    features.grid_z = profile.grid_dim.z;
    features.block_x = profile.block_dim.x;
    features.block_y = profile.block_dim.y;
    features.block_z = profile.block_dim.z;
    features.shared_mem_kb = profile.shared_mem_size / 1024;
    
    // Calculate input tensor volume
    features.input_tensor_volume = 1;
    for (size_t shape : profile.input_shapes) {
        features.input_tensor_volume *= shape;
    }
    
    // Derived features
    features.arithmetic_intensity = calculateArithmeticIntensity(profile);
    features.parallelism_degree = calculateParallelismDegree(profile);
    features.memory_access_pattern_score = calculateMemoryAccessPatternScore(profile);
    features.operation_complexity_score = calculateOperationComplexityScore(profile);
    
    // Historical features (simplified)
    features.recent_avg_execution_time_ms = profile.execution_time_ns / 1000000.0f;
    features.queue_depth_penalty = 0.0f;  // Would be calculated from queue state
    
    // Hardware context
    features.compute_capability_major = hardware_context_.compute_capability_major;
    features.compute_capability_minor = hardware_context_.compute_capability_minor;
    features.total_global_memory_gb = hardware_context_.total_global_memory_gb;
    features.multiprocessor_count = hardware_context_.multiprocessor_count;
    
    return features;
}

bool AIPredictor::updateModel(const std::vector<KernelProfile>& profiles, 
                             const std::vector<float>& actual_times) {
    // This would implement online learning or model retraining
    // For now, we'll just update the prediction accuracy
    if (profiles.size() != actual_times.size() || profiles.empty()) {
        return false;
    }
    
    float total_error = 0.0f;
    size_t valid_predictions = 0;
    
    for (size_t i = 0; i < profiles.size(); ++i) {
        auto prediction = predict(profiles[i]);
        if (prediction.predicted_execution_time_ms > 0) {
            float error = std::abs(prediction.predicted_execution_time_ms - actual_times[i]);
            total_error += error;
            valid_predictions++;
        }
    }
    
    if (valid_predictions > 0) {
        prediction_accuracy_ = 100.0f - (total_error / valid_predictions);
        prediction_accuracy_ = std::max(0.0f, std::min(100.0f, prediction_accuracy_));
    }
    
    return true;
}

float AIPredictor::getAccuracy() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return prediction_accuracy_;
}

void AIPredictor::enableCaching(bool enabled) {
    caching_enabled_ = enabled;
}

void AIPredictor::clearCache() {
    prediction_cache_.clear();
    cache_hits_ = 0;
}

AIPredictor::CacheStats AIPredictor::getCacheStats() const {
    CacheStats stats;
    stats.cache_size = prediction_cache_.size();
    stats.max_cache_size = max_cache_size_;
    stats.hit_rate = (total_predictions_ > 0) ? 
        static_cast<float>(cache_hits_) / total_predictions_ * 100.0f : 0.0f;
    return stats;
}

void AIPredictor::shutdown() {
    try {
        if (onnx_model_) {
            onnx_model_->unloadModel();
        }
        prediction_cache_.clear();
    } catch (const std::exception& e) {
        std::cerr << "Error during AI predictor shutdown: " << e.what() << std::endl;
    }
}

float AIPredictor::calculateArithmeticIntensity(const KernelProfile& profile) {
    // Arithmetic intensity = operations per byte accessed
    // Simplified calculation based on grid/block dimensions
    uint64_t total_threads = profile.grid_dim.x * profile.grid_dim.y * profile.grid_dim.z *
                            profile.block_dim.x * profile.block_dim.y * profile.block_dim.z;
    
    uint64_t memory_accesses = profile.input_shapes.empty() ? total_threads : 
                               std::accumulate(profile.input_shapes.begin(), profile.input_shapes.end(), 0ULL);
    
    if (memory_accesses > 0) {
        return static_cast<float>(total_threads) / static_cast<float>(memory_accesses);
    }
    
    return 1.0f;  // Default value
}

float AIPredictor::calculateParallelismDegree(const KernelProfile& profile) {
    // Parallelism degree = total threads / max threads per SM
    uint64_t total_threads = profile.grid_dim.x * profile.grid_dim.y * profile.grid_dim.z *
                            profile.block_dim.x * profile.block_dim.y * profile.block_dim.z;
    
    int max_threads_per_sm = 2048;  // Typical value, should be queried from device
    int num_sms = hardware_context_.multiprocessor_count;
    
    if (num_sms > 0) {
        return static_cast<float>(total_threads) / (max_threads_per_sm * num_sms);
    }
    
    return 1.0f;  // Default value
}

float AIPredictor::calculateMemoryAccessPatternScore(const KernelProfile& profile) {
    // Memory access pattern score based on shared memory usage and input shapes
    float shared_mem_ratio = static_cast<float>(profile.shared_mem_size) / (48 * 1024);  // 48KB typical shared memory
    float coalescing_score = 1.0f;  // Would be calculated based on memory access patterns
    
    return (shared_mem_ratio + coalescing_score) / 2.0f;
}

float AIPredictor::calculateOperationComplexityScore(const KernelProfile& profile) {
    // Operation complexity based on operation type
    const std::string& op_type = profile.operation_type;
    
    if (op_type.find("convolution") != std::string::npos) return 0.9f;
    if (op_type.find("matrix_multiply") != std::string::npos) return 0.8f;
    if (op_type.find("reduction") != std::string::npos) return 0.7f;
    if (op_type.find("elementwise") != std::string::npos) return 0.5f;
    if (op_type.find("memory_copy") != std::string::npos) return 0.3f;
    
    return 0.5f;  // Default complexity
}

size_t AIPredictor::computeFeatureHash(const FeatureVector& features) {
    // Simple hash function for feature vector
    size_t hash = 0;
    
    hash ^= std::hash<uint32_t>{}(features.grid_x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.grid_y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.grid_z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.block_x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.block_y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.block_z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>{}(features.shared_mem_kb) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

void AIPredictor::trimCache() {
    // Remove oldest entries when cache is full
    size_t target_size = max_cache_size_ / 2;
    
    std::vector<std::pair<size_t, PredictionResult>> cache_entries;
    cache_entries.reserve(prediction_cache_.size());
    
    for (const auto& entry : prediction_cache_) {
        cache_entries.emplace_back(entry.first, entry.second);
    }
    
    // Sort by prediction time (oldest first)
    std::sort(cache_entries.begin(), cache_entries.end(),
              [](const auto& a, const auto& b) {
                  return a.second.prediction_time < b.second.prediction_time;
              });
    
    // Keep only the newest entries
    prediction_cache_.clear();
    for (size_t i = cache_entries.size() - target_size; i < cache_entries.size(); ++i) {
        prediction_cache_[cache_entries[i].first] = cache_entries[i].second;
    }
}

std::vector<float> AIPredictor::featuresToInput(const FeatureVector& features) {
    // Convert feature vector to model input
    std::vector<float> input;
    input.reserve(20);  // Expected input size
    
    // Primary features
    input.push_back(static_cast<float>(features.grid_x));
    input.push_back(static_cast<float>(features.grid_y));
    input.push_back(static_cast<float>(features.grid_z));
    input.push_back(static_cast<float>(features.block_x));
    input.push_back(static_cast<float>(features.block_y));
    input.push_back(static_cast<float>(features.block_z));
    input.push_back(static_cast<float>(features.shared_mem_kb));
    input.push_back(static_cast<float>(features.input_tensor_volume));
    input.push_back(features.operation_complexity_score);
    
    // Derived features
    input.push_back(features.arithmetic_intensity);
    input.push_back(features.parallelism_degree);
    input.push_back(features.memory_access_pattern_score);
    
    // Historical features
    input.push_back(features.recent_avg_execution_time_ms);
    input.push_back(features.queue_depth_penalty);
    
    // Hardware context
    input.push_back(static_cast<float>(features.compute_capability_major));
    input.push_back(static_cast<float>(features.compute_capability_minor));
    input.push_back(static_cast<float>(features.total_global_memory_gb));
    input.push_back(static_cast<float>(features.multiprocessor_count));
    
    // Pad to expected size
    while (input.size() < 20) {
        input.push_back(0.0f);
    }
    
    return input;
}

PredictionResult AIPredictor::runInference(const std::vector<float>& input) {
    PredictionResult result;
    
    if (!onnx_model_ || !onnx_model_->isLoaded()) {
        result.predicted_execution_time_ms = 0.0f;
        result.confidence_score = 0.0f;
        return result;
    }
    
    // Run model inference
    float prediction = onnx_model_->predict(input);
    
    result.predicted_execution_time_ms = prediction;
    result.confidence_score = 0.8f;  // Default confidence
    result.prediction_time = std::chrono::high_resolution_clock::now();
    
    return result;
}

bool AIPredictor::initializeHardwareContext() {
    try {
        // Get CUDA device properties
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get device properties" << std::endl;
            return false;
        }
        
        // Fill hardware context
        hardware_context_.compute_capability_major = prop.major;
        hardware_context_.compute_capability_minor = prop.minor;
        hardware_context_.total_global_memory_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
        hardware_context_.multiprocessor_count = prop.multiProcessorCount;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize hardware context: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cuda_scheduler 