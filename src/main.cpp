#include <iostream>
#include <string>
#include <algorithm> // for std::max_element

// Define the stb_image implementation before including it
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "parser.hpp"
#include "ir.hpp"
#include "optimizer.hpp"
#include "backend.hpp"

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << " Mini-TensorRT Compiler Initializing... \n";
    std::cout << "========================================\n\n";
    
    if (argc < 3) {
        std::cerr << "Error: Missing arguments.\n";
        std::cerr << "Usage: ./mini_tensorrt <path_to_onnx_model> <path_to_image>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2]; // We now take an image path!

    // 1. FRONTEND: Parse the binary ONNX file
    minitrt::ONNXParser parser(model_path);
    std::shared_ptr<minitrt::Graph> my_graph = parser.parse();

    // 2. MIDDLE-END: Run Graph Optimization (Operator Fusion)
    minitrt::Optimizer optimizer;
    optimizer.run_passes(my_graph);

    // 3. BACKEND: Load the REAL image into memory
    int width, height, channels;
    // Load the image as grayscale (1 channel)
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
    
    if (!img_data) {
        std::cerr << "[Error] Failed to load image: " << image_path << "\n";
        return 1;
    }

    std::cout << "\n[System] Loaded image " << width << "x" << height << " into memory.\n";

    // Find the input tensor and fill it with the image data
    for (auto& tensor : my_graph->tensors) {
        if (tensor->name == "input_image") {
            tensor->shape = {1, 1, height, width}; // Dynamic shape from the image!
            tensor->data.resize(tensor->elements());
            
            // Convert standard 0-255 pixels into 0.0-1.0 floats for the neural network
            for (size_t i = 0; i < tensor->elements(); ++i) {
                tensor->data[i] = img_data[i] / 255.0f;
            }
        }
    }
    
    // Free the raw image data now that it's in our custom Tensor
    stbi_image_free(img_data);

    // 4. EXECUTE!
    minitrt::ExecutionEngine engine(my_graph);
    engine.run();

    // 5. POST-PROCESSING: Find the highest vote (Argmax)
    std::shared_ptr<minitrt::Tensor> final_output = nullptr;
    // The Gemm node creates an output named something like "predictions" or a numeric ID
    // Let's grab the last tensor in the graph, which is usually the final output
    final_output = my_graph->nodes.back()->outputs[0];

    if (final_output) {
        int best_digit = 0;
        float max_logit = final_output->data[0];
        
        for (int i = 1; i < 10; ++i) {
            if (final_output->data[i] > max_logit) {
                max_logit = final_output->data[i];
                best_digit = i;
            }
        }
        
        std::cout << "\n========================================\n";
        std::cout << " >> ENGINE PREDICTION: The digit is " << best_digit << " << \n";
        std::cout << "========================================\n";
    }

    return 0;
}