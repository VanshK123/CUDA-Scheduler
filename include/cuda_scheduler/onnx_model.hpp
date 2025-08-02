#pragma once

#include <string>
#include <vector>
#include <memory>

#ifndef DISABLE_AI_FEATURES
#include <onnxruntime_cxx_api.h>
#endif

namespace cuda_scheduler {

/**
 * @brief ONNX model wrapper for ML inference
 * 
 * This class provides a simplified interface for ONNX Runtime model inference,
 * handling model loading, input preparation, and output processing.
 */
class ONNXModel {
public:
    /**
     * @brief Constructor
     */
    ONNXModel();
    
    /**
     * @brief Destructor
     */
    ~ONNXModel();
    
    /**
     * @brief Load model from file
     * @param model_path Path to ONNX model file
     * @return true if loading successful
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Run inference on input features
     * @param input_features Input feature vector
     * @return Prediction result
     */
    float predict(const std::vector<float>& input_features);
    
    /**
     * @brief Get model information
     * @return Model info structure
     */
    struct ModelInfo {
        std::string model_path;
        size_t input_size;
        size_t output_size;
        std::string model_version;
        bool is_loaded;
    };
    
    ModelInfo getModelInfo() const;
    
    /**
     * @brief Check if model is loaded
     * @return true if model is loaded
     */
    bool isLoaded() const;
    
    /**
     * @brief Get expected input size
     * @return Number of input features expected
     */
    size_t getInputSize() const;
    
    /**
     * @brief Unload the model
     */
    void unloadModel();

private:
#ifndef DISABLE_AI_FEATURES
    // ONNX Runtime components
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    
    // Model metadata
    std::string model_path_;
    std::string model_version_;
    size_t input_size_;
    size_t output_size_;
    bool is_loaded_;
    
    // Input/output names
    std::string input_name_;
    std::string output_name_;
    
    // Helper methods
    bool initializeONNXRuntime();
    void cleanupONNXRuntime();
    std::vector<float> prepareInput(const std::vector<float>& features);
    float processOutput(const Ort::Value& output);
    
#else
    // Stub implementation when AI features are disabled
    std::string model_path_;
    bool is_loaded_;
    size_t input_size_;
    size_t output_size_;
    
    float predict(const std::vector<float>& input_features) {
        // Return a default prediction when AI is disabled
        return 1.0f;  // 1ms default prediction
    }
#endif
};

} // namespace cuda_scheduler 