#include "backend.hpp"
#include <iostream>
#include <algorithm> // For std::max
#include <chrono>
#include <map>

namespace minitrt {

    ExecutionEngine::ExecutionEngine(std::shared_ptr<Graph> graph) 
        : execution_graph(graph) {}
    
    void ExecutionEngine::run() {
        
        // --- STATIC SHAPE INFERENCE ---
        for (const auto& node : execution_graph->nodes) {
            auto input = node->inputs[0];
            auto output = node->outputs[0];

            if (node->op_type == "Conv" || node->op_type == "ConvRelu") {
                // For our MNIST model: 28x28 input -> 28x28 output (with pad=1)
                output->shape = {1, 4, 28, 28}; 
            } 
            else if (node->op_type == "Relu") {
                output->shape = input->shape;
            }
            
            // Now that the shape is set, allocate the memory
            if (output->data.empty() && output->elements() > 0) {
                output->data.resize(output->elements(), 0.0f);
            }
        }

        std::cout << "[Backend] Starting Execution Engine...\n";
        std::cout << "[Backend] Starting Execution Engine...\n";
        
        for (const auto& node : execution_graph->nodes) {
            // Start Timer
            auto start = std::chrono::high_resolution_clock::now();
            
            // Execute the node
            if (node->op_type == "ConvRelu") { execute_conv_relu(node); }
            else if (node->op_type == "Conv") { execute_conv2d(node); }
            else if (node->op_type == "Relu") { execute_relu(node); }
            else if (node->op_type == "MaxPool") { execute_maxpool(node); }
            else if (node->op_type == "Reshape") { execute_reshape(node); }
            else if (node->op_type == "Gemm") { execute_gemm(node); }
            
            // Stop Timer
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            // PRINT IMMEDIATELY so we see it even if it crashes later!
            std::cout << "  [Benchmark] " << node->op_type << " latency: " << duration << " us\n";
        }
    }

    void ExecutionEngine::execute_relu(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing ReLU on node: " << node->name << "\n";
        
        // Safety check: Ensure the node actually has inputs and outputs linked
        if (node->inputs.empty() || node->outputs.empty()) {
            std::cerr << "    [Error] ReLU node lacks input/output tensors. Graph topology is broken.\n";
            return;
        }

        // Grab the memory pointers
        auto input_tensor = node->inputs[0];
        auto output_tensor = node->outputs[0];

        // The Math Kernel: Loop through the raw data array and apply max(0, x)
        for (size_t i = 0; i < input_tensor->data.size(); ++i) {
            output_tensor->data[i] = std::max(0.0f, input_tensor->data[i]);
        }
    }

    void ExecutionEngine::execute_conv2d(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing Conv2D on node: " << node->name << "\n";
        
        // Safety check
        if (node->inputs.size() < 2 || node->outputs.empty()) return;
    
        auto input = node->inputs[0];        // The image
        auto weight = node->inputs[1];       // The filters/kernels
        auto output = node->outputs[0];      // The result
    
        // For our dummy model, we know it's a 28x28 image with a 3x3 kernel and padding=1.
        // In a full compiler, we would parse these attributes dynamically from the ONNX NodeProto.
        int in_h = 28, in_w = 28;
        int kernel_h = 3, kernel_w = 3;
        int pad = 1;
        int stride = 1;
    
        // Calculate output dimensions based on the standard spatial formula
        int out_h = ((in_h - kernel_h + 2 * pad) / stride) + 1;
        int out_w = ((in_w - kernel_w + 2 * pad) / stride) + 1;
    
        // Allocate memory for the output tensor if it was a "blank" intermediate tensor
        if (output->data.empty()) {
            output->shape = {1, 1, out_h, out_w}; // Batch 1, Channel 1
            output->data.resize(out_h * out_w, 0.0f);
        }
    
        // The Naive Sliding Window Loops
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                
                float sum = 0.0f;
                
                // Slide the kernel over the input
                for (int ky = 0; ky < kernel_h; ++ky) {
                    for (int kx = 0; kx < kernel_w; ++kx) {
                        
                        // Map the kernel position back to the original image coordinates (accounting for padding)
                        int in_y = (y * stride) + ky - pad;
                        int in_x = (x * stride) + kx - pad;
                        
                        // Boundary check: If we are outside the image, the value is 0 (Zero Padding)
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            
                            // Calculate the flattened 1D array index for the 2D matrices
                            int input_idx = in_y * in_w + in_x;
                            int weight_idx = ky * kernel_w + kx;
                            
                            // Multiply and accumulate
                            sum += input->data[input_idx] * weight->data[weight_idx];
                        }
                    }
                }
                
                // Store the final pixel value in the output tensor
                int out_idx = y * out_w + x;
                output->data[out_idx] = sum;
            }
        }
    }

    void ExecutionEngine::execute_conv_relu(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing Fused Conv2D+ReLU on node: " << node->name << "\n";
        
        if (node->inputs.size() < 2 || node->outputs.empty()) return;
    
        auto input = node->inputs[0];        // The image
        auto weight = node->inputs[1];       // The filters/kernels
        auto output = node->outputs[0];      // The result
    
        // 1. DYNAMIC SHAPE INFERENCE!
        // Read Height (index 2) and Width (index 3) directly from the tensors
        int in_h = input->shape[2];
        int in_w = input->shape[3];
        
        int kernel_h = weight->shape[2];
        int kernel_w = weight->shape[3];
        
        // For now, we will leave pad and stride hardcoded, but in a full 
        // production compiler, we would parse these from the ONNX Node attributes!
        int pad = 1;
        int stride = 1;
    
        // 2. Calculate the dynamic output dimensions
        int out_h = ((in_h - kernel_h + 2 * pad) / stride) + 1;
        int out_w = ((in_w - kernel_w + 2 * pad) / stride) + 1;
    
        // 3. The kernel must always own the output sizing
        output->shape = {1, 1, out_h, out_w};
        output->data.resize(out_h * out_w, 0.0f);
    
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float sum = 0.0f; // Reset sum for each new output pixel
                
                for (int ky = 0; ky < kernel_h; ++ky) {
                    for (int kx = 0; kx < kernel_w; ++kx) {
                        int in_y = (y * stride) + ky - pad;
                        int in_x = (x * stride) + kx - pad;
                        
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            sum += input->data[in_y * in_w + in_x] * weight->data[ky * kernel_w + kx];
                        }
                    }
                }
                
                // --- THE FUSION STEP ---
                // Apply ReLU to the final sum before writing it to RAM.
                // This happens once per output pixel, AFTER the kernel loops.
                int out_idx = y * out_w + x;
                output->data[out_idx] = std::max(0.0f, sum); 
            }
        }
    }

    // ---------------------------------------------------------
    // KERNEL: MaxPool (Downsampling the image)
    // ---------------------------------------------------------
    void ExecutionEngine::execute_maxpool(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing MaxPool on node: " << node->name << "\n";
        auto input = node->inputs[0];
        auto output = node->outputs[0];

        // For this milestone, we do a simple 2x2 pooling grid.
        // A 28x28 image becomes 14x14.
        int in_h = 28, in_w = 28;
        int out_h = 14, out_w = 14;
        
        output->shape = {1, 1, out_h, out_w};
        output->data.resize(out_h * out_w, 0.0f);

        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float max_val = -999999.0f; // Start with a very low number
                
                // Scan the 2x2 window
                for (int ky = 0; ky < 2; ++ky) {
                    for (int kx = 0; kx < 2; ++kx) {
                        int in_y = y * 2 + ky;
                        int in_x = x * 2 + kx;
                        float val = input->data[in_y * in_w + in_x];
                        if (val > max_val) max_val = val;
                    }
                }
                // Save the brightest pixel
                output->data[y * out_w + x] = max_val;
            }
        }
    }

    // ---------------------------------------------------------
    // KERNEL: Reshape (Flattening)
    // ---------------------------------------------------------
    void ExecutionEngine::execute_reshape(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing Reshape on node: " << node->name << "\n";
        auto input = node->inputs[0];
        auto output = node->outputs[0];

        // The beauty of C++: A 2D grid and a 1D flat array look identical in RAM.
        // We just do a pure memory copy and update the shape metadata.
        output->shape = {1, (int64_t)input->elements()}; 
        output->data = input->data; 
    }

    // ---------------------------------------------------------
    // KERNEL: Gemm (General Matrix Multiplication / Linear Layer)
    // ---------------------------------------------------------
    void ExecutionEngine::execute_gemm(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing Gemm (Linear) on node: " << node->name << "\n";
        auto input = node->inputs[0];   // Shape: [1, Features]
        auto weight = node->inputs[1];  // Shape: [10 classes, Features]
        auto bias = node->inputs[2];    // Shape: [10 classes]
        auto output = node->outputs[0];

        int in_features = weight->shape[1];  // Size of incoming data
        int out_features = weight->shape[0]; // 10 (Digits 0-9)

        output->shape = {1, out_features};
        output->data.resize(out_features, 0.0f);

        // Standard Matrix-Vector Dot Product
        for (int i = 0; i < out_features; ++i) {
            float sum = bias->data[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input->data[j] * weight->data[i * in_features + j];
            }
            output->data[i] = sum;
        }
        
        // Print the final prediction raw scores (Logits)
        std::cout << "     [Prediction Logits]: ";
        for(float val : output->data) std::cout << val << " ";
        std::cout << "\n";
    }

} // namespace minitrt