#!/bin/bash

# CUDA Scheduler Build Script
# This script builds the entire project including the main library, examples, and tests

set -e  # Exit on any error

echo "=========================================="
echo "CUDA Scheduler Build Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
fi

# Create build directory
BUILD_DIR="build"
echo "Creating build directory: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building the project..."
make -j$(nproc)

# Run tests if available
if [ -f "tests/scheduler_tests" ]; then
    echo "Running unit tests..."
    ./tests/scheduler_tests
else
    echo "Warning: Unit tests not found. Skipping test execution."
fi

# Build examples
echo "Building examples..."
if [ -f "examples/basic_usage" ]; then
    echo "Example built successfully: examples/basic_usage"
else
    echo "Warning: Example not found."
fi

# Install Python dependencies for ML components
echo "Installing Python dependencies..."
pip install -r ../requirements.txt

# Train ML model if Python script exists
if [ -f "../ml/train_model.py" ]; then
    echo "Training ML model..."
    cd ..
    python ml/train_model.py --num-samples 5000
    cd $BUILD_DIR
else
    echo "Warning: ML training script not found."
fi

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the example: ./examples/basic_usage"
echo "2. Run benchmarks: ./benchmarks/mixed_workload_benchmark"
echo "3. Check performance: ./tests/performance_tests"
echo ""
echo "For more information, see README.md" 