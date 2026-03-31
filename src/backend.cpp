#include "backend.hpp"
#include <iostream>
#include <algorithm> // For std::max

namespace minitrt {

    ExecutionEngine::ExecutionEngine(std::shared_ptr<Graph> graph) 
        : execution_graph(graph) {}

void ExecutionEngine::run() {
    std::cout << "[Backend] Starting Execution Engine...\n";
    
    for (const auto& node : execution_graph->nodes) {
        if (node->op_type == "ConvRelu") { // Match the name set in optimizer.cpp
            execute_conv_relu(node);
        } else if (node->op_type == "Conv") {
            execute_conv2d(node);
        } else if (node->op_type == "Relu") {
            execute_relu(node);
        } else {
            std::cout << "[Backend] Warning: No kernel for " << node->op_type << "\n";
        }
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
    
        auto input = node->inputs[0];
        auto weight = node->inputs[1];
        auto output = node->outputs[0];
    
        int in_h = 28, in_w = 28;
        int kernel_h = 3, kernel_w = 3;
        int pad = 1, stride = 1;
    
        int out_h = ((in_h - kernel_h + 2 * pad) / stride) + 1;
        int out_w = ((in_w - kernel_w + 2 * pad) / stride) + 1;
    
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

} // namespace minitrt