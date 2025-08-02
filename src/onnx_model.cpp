#include "cuda_scheduler/onnx_model.hpp"
#include <iostream>
#include <fstream>

namespace cuda_scheduler {

ONNXModel::ONNXModel() 
    : is_loaded_(false)
    , input_size_(0)
    , output_size_(0) {
#ifndef DISABLE_AI_FEATURES
    initializeONNXRuntime();
#endif
}

ONNXModel::~ONNXModel() {
    unloadModel();
#ifndef DISABLE_AI_FEATURES
    cleanupONNXRuntime();
#endif
}

bool ONNXModel::loadModel(const std::string& model_path) {
    try {
        // Check if file exists
        std::ifstream file(model_path);
        if (!file.good()) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            return false;
        }
        file.close();
        
#ifndef DISABLE_AI_FEATURES
        // Load model with ONNX Runtime
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
        
        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input name
        input_name_ = session_.GetInputName(0, allocator);
        
        // Get output name
        output_name_ = session_.GetOutputName(0, allocator);
        
        // Get input shape
        auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = 1;
        for (auto dim : input_shape) {
            if (dim > 0) {
                input_size_ *= dim;
            }
        }
        
        // Get output shape
        auto output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_size_ = 1;
        for (auto dim : output_shape) {
            if (dim > 0) {
                output_size_ *= dim;
            }
        }
        
        model_path_ = model_path;
        model_version_ = "1.0.0";  // Default version
        is_loaded_ = true;
        
        std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
        std::cout << "Input size: " << input_size_ << ", Output size: " << output_size_ << std::endl;
        
        return true;
#else
        // Stub implementation when AI features are disabled
        model_path_ = model_path;
        input_size_ = 20;  // Default input size
        output_size_ = 1;  // Single output (execution time)
        model_version_ = "1.0.0";
        is_loaded_ = true;
        
        std::cout << "AI features disabled - using stub model: " << model_path << std::endl;
        return true;
#endif
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

float ONNXModel::predict(const std::vector<float>& input_features) {
    if (!is_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return 0.0f;
    }
    
    if (input_features.size() != input_size_) {
        std::cerr << "Input size mismatch. Expected: " << input_size_ 
                  << ", Got: " << input_features.size() << std::endl;
        return 0.0f;
    }
    
    try {
#ifndef DISABLE_AI_FEATURES
        // Prepare input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, 
            const_cast<float*>(input_features.data()), 
            input_features.size(),
            nullptr, 
            input_size_
        );
        
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            &input_name_, 
            &input_tensor, 
            1,
            &output_name_, 
            1
        );
        
        // Process output
        return processOutput(output_tensors[0]);
#else
        // Stub implementation - return a simple prediction based on input features
        float prediction = 1.0f;  // Base prediction of 1ms
        
        // Simple heuristic based on grid and block dimensions
        if (input_features.size() >= 6) {
            float grid_volume = input_features[0] * input_features[1] * input_features[2];
            float block_volume = input_features[3] * input_features[4] * input_features[5];
            float total_threads = grid_volume * block_volume;
            
            // Rough estimation: more threads = longer execution time
            prediction = std::min(100.0f, total_threads / 1000000.0f);
        }
        
        return prediction;
#endif
        
    } catch (const std::exception& e) {
        std::cerr << "Prediction failed: " << e.what() << std::endl;
        return 0.0f;
    }
}

ONNXModel::ModelInfo ONNXModel::getModelInfo() const {
    ModelInfo info;
    info.model_path = model_path_;
    info.input_size = input_size_;
    info.output_size = output_size_;
    info.model_version = model_version_;
    info.is_loaded = is_loaded_;
    return info;
}

bool ONNXModel::isLoaded() const {
    return is_loaded_;
}

size_t ONNXModel::getInputSize() const {
    return input_size_;
}

void ONNXModel::unloadModel() {
    is_loaded_ = false;
    model_path_.clear();
    input_size_ = 0;
    output_size_ = 0;
}

#ifndef DISABLE_AI_FEATURES
bool ONNXModel::initializeONNXRuntime() {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "cuda_scheduler");
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX Runtime: " << e.what() << std::endl;
        return false;
    }
}

void ONNXModel::cleanupONNXRuntime() {
    // ONNX Runtime cleanup is automatic
}

std::vector<float> ONNXModel::prepareInput(const std::vector<float>& features) {
    // Normalize features if needed
    std::vector<float> normalized_features = features;
    
    // Simple normalization: scale to [0, 1] range
    float max_val = 0.0f;
    for (float val : features) {
        max_val = std::max(max_val, std::abs(val));
    }
    
    if (max_val > 0.0f) {
        for (float& val : normalized_features) {
            val /= max_val;
        }
    }
    
    return normalized_features;
}

float ONNXModel::processOutput(const Ort::Value& output) {
    // Get output data
    float* output_data = output.GetTensorMutableData<float>();
    
    // For regression models, we expect a single output value
    if (output.GetTensorTypeAndShapeInfo().GetElementCount() > 0) {
        return output_data[0];
    }
    
    return 0.0f;
}
#endif

} // namespace cuda_scheduler 