#include <iostream>
#include <string>
#include "ir.hpp"

int main(int argc, char** argv) {
    std::cout << "Mini-TensorRT Compiler Initializing...\n" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Error: No model provided.\n";
        std::cerr << "Usage: ./mini_tensorrt <path_to_onnx_model>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Target model to parse: " << model_path << "\n";

    // Quick test to ensure our IR library is linking correctly
    minitrt::Graph dummy_graph("Initialization_Test_Graph");
    dummy_graph.print_summary();

    std::cout << "[Ready for ONNX Protobuf Integration]\n";

    return 0;
}
