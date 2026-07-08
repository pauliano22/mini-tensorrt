#include "backend.hpp"
#include <iostream>
#include <algorithm> // For std::max
#include <chrono>
#include <map>

namespace minitrt {

    ExecutionEngine::ExecutionEngine(std::shared_ptr<Graph> graph) 
        : execution_graph(graph) {}
    
    void ExecutionEngine::run() {

        // Each kernel owns its output shape and allocation: shapes are inferred
        // dynamically from the incoming tensors at execution time.
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

        // The kernel owns the output sizing
        output_tensor->shape = input_tensor->shape;
        output_tensor->data.resize(input_tensor->data.size());

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
        auto weight = node->inputs[1];       // The filters/kernels, NCHW: [out_c, in_c, kh, kw]
        auto bias = node->inputs.size() > 2 ? node->inputs[2] : nullptr;
        auto output = node->outputs[0];      // The result

        // Read every dimension directly from the tensors (dynamic shape inference)
        int in_c = input->shape[1];
        int in_h = input->shape[2];
        int in_w = input->shape[3];

        int out_c = weight->shape[0];
        int kernel_h = weight->shape[2];
        int kernel_w = weight->shape[3];

        // For now, we will leave pad and stride hardcoded, but in a full
        // production compiler, we would parse these from the ONNX Node attributes!
        int pad = 1;
        int stride = 1;

        // Calculate output dimensions based on the standard spatial formula
        int out_h = ((in_h - kernel_h + 2 * pad) / stride) + 1;
        int out_w = ((in_w - kernel_w + 2 * pad) / stride) + 1;

        // The kernel must always own the output sizing
        output->shape = {1, out_c, out_h, out_w};
        output->data.assign((size_t)out_c * out_h * out_w, 0.0f);

        // The Naive Sliding Window Loops, one pass per output channel
        for (int oc = 0; oc < out_c; ++oc) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {

                    // Every output pixel starts from this filter's bias term
                    float sum = bias ? bias->data[oc] : 0.0f;

                    // Slide the kernel over every input channel
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            for (int kx = 0; kx < kernel_w; ++kx) {

                                // Map the kernel position back to the original image coordinates (accounting for padding)
                                int in_y = (y * stride) + ky - pad;
                                int in_x = (x * stride) + kx - pad;

                                // Boundary check: If we are outside the image, the value is 0 (Zero Padding)
                                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {

                                    // Calculate the flattened 1D array index for the NCHW layouts
                                    int input_idx  = (ic * in_h + in_y) * in_w + in_x;
                                    int weight_idx = ((oc * in_c + ic) * kernel_h + ky) * kernel_w + kx;

                                    // Multiply and accumulate
                                    sum += input->data[input_idx] * weight->data[weight_idx];
                                }
                            }
                        }
                    }

                    // Store the final pixel value in the output tensor
                    int out_idx = (oc * out_h + y) * out_w + x;
                    output->data[out_idx] = sum;
                }
            }
        }
    }

    void ExecutionEngine::execute_conv_relu(std::shared_ptr<Node> node) {
        std::cout << "  -> Executing Fused Conv2D+ReLU on node: " << node->name << "\n";
        
        if (node->inputs.size() < 2 || node->outputs.empty()) return;
    
        auto input = node->inputs[0];        // The image
        auto weight = node->inputs[1];       // The filters/kernels, NCHW: [out_c, in_c, kh, kw]
        auto bias = node->inputs.size() > 2 ? node->inputs[2] : nullptr;
        auto output = node->outputs[0];      // The result

        // 1. DYNAMIC SHAPE INFERENCE!
        // Read every dimension directly from the tensors
        int in_c = input->shape[1];
        int in_h = input->shape[2];
        int in_w = input->shape[3];

        int out_c = weight->shape[0];
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
        output->shape = {1, out_c, out_h, out_w};
        output->data.assign((size_t)out_c * out_h * out_w, 0.0f);

        for (int oc = 0; oc < out_c; ++oc) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    // Every output pixel starts from this filter's bias term
                    float sum = bias ? bias->data[oc] : 0.0f;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                int in_y = (y * stride) + ky - pad;
                                int in_x = (x * stride) + kx - pad;

                                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                    sum += input->data[(ic * in_h + in_y) * in_w + in_x]
                                         * weight->data[((oc * in_c + ic) * kernel_h + ky) * kernel_w + kx];
                                }
                            }
                        }
                    }

                    // --- THE FUSION STEP ---
                    // Apply ReLU to the final sum before writing it to RAM.
                    // This happens once per output pixel, AFTER the kernel loops.
                    int out_idx = (oc * out_h + y) * out_w + x;
                    output->data[out_idx] = std::max(0.0f, sum);
                }
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

        // For this milestone, we do a simple 2x2 pooling grid over every channel.
        // Each 28x28 feature map becomes 14x14.
        int channels = input->shape[1];
        int in_h = input->shape[2];
        int in_w = input->shape[3];
        int out_h = in_h / 2, out_w = in_w / 2;

        output->shape = {1, channels, out_h, out_w};
        output->data.assign((size_t)channels * out_h * out_w, 0.0f);

        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float max_val = -999999.0f; // Start with a very low number

                    // Scan the 2x2 window
                    for (int ky = 0; ky < 2; ++ky) {
                        for (int kx = 0; kx < 2; ++kx) {
                            int in_y = y * 2 + ky;
                            int in_x = x * 2 + kx;
                            float val = input->data[(c * in_h + in_y) * in_w + in_x];
                            if (val > max_val) max_val = val;
                        }
                    }
                    // Save the brightest pixel
                    output->data[(c * out_h + y) * out_w + x] = max_val;
                }
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

        // Guard against silent garbage: every buffer must match the declared shapes.
        // An undersized vector here means the parser skipped a weight or an
        // upstream kernel produced the wrong number of elements.
        if (input->data.size() < (size_t)in_features ||
            weight->data.size() < (size_t)out_features * in_features ||
            (bias && bias->data.size() < (size_t)out_features)) {
            std::cerr << "    [Error] Gemm buffer mismatch: input " << input->data.size()
                      << "/" << in_features << ", weight " << weight->data.size()
                      << "/" << (size_t)out_features * in_features << ". Aborting node.\n";
            return;
        }

        output->shape = {1, out_features};
        output->data.assign(out_features, 0.0f);

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