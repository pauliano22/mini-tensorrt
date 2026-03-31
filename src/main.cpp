#include <iostream>
#include <string>
#include "parser.hpp"
#include "ir.hpp"
#include "optimizer.hpp" // Added
#include "backend.hpp"   // Added

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << " Mini-TensorRT Compiler Initializing... \n";
    std::cout << "========================================\n\n";
    
    if (argc < 2) {
        std::cerr << "Error: No model provided.\n";
        std::cerr << "Usage: ./mini_tensorrt <path_to_onnx_model>\n";
        return 1;
    }

    std::string model_path = argv[1];

    // 1. FRONTEND: Parse the binary ONNX file
    minitrt::ONNXParser parser(model_path);
    std::shared_ptr<minitrt::Graph> my_graph = parser.parse();

    std::cout << "\n--- Graph Before Optimization ---\n";
    my_graph->print_summary();

    // 2. MIDDLE-END: Run Graph Optimization (Operator Fusion)
    minitrt::Optimizer optimizer;
    optimizer.run_passes(my_graph);

    std::cout << "\n--- Graph After Optimization ---\n";
    my_graph->print_summary();

    // 3. BACKEND: Execute the optimized graph
    minitrt::ExecutionEngine engine(my_graph);

    // --- Developer Hack: Initialize Tensors with Dummy Data ---
    for (auto& tensor : my_graph->tensors) {
        if (tensor->data.empty()) {
            if (tensor->name == "input_image") {
                tensor->shape = {1, 1, 28, 28};
                tensor->data.resize(tensor->elements(), 0.5f);
            } 
            else if (tensor->elements() > 0) {
                // If the tensor knows its size (like the 3x3 weights) but has no floats, fill it!
                tensor->data.resize(tensor->elements(), 0.1f); 
            }
        }
    }

    engine.run();

    std::cout << "\n========================================\n";
    std::cout << " Pipeline Finished Successfully. \n";
    std::cout << "========================================\n";

    return 0;
}