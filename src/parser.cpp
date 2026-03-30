#include "parser.hpp"
#include <iostream>

namespace minitrt {

    ONNXParser::ONNXParser(const std::string& file_path) : model_path(file_path) {
        std::cout << "[Parser] Initialized for model: " << model_path << "\n";
    }

    std::shared_ptr<Graph> ONNXParser::parse() {
        std::cout << "[Parser] WARNING: Protobuf decoding not yet implemented.\n";
        
        // Create an empty graph for now
        auto graph = std::make_shared<Graph>("Imported_ONNX_Model");
        
        // TODO: Open file via std::ifstream
        // TODO: Parse using ONNX Protobuf classes (e.g., onnx::ModelProto)
        // TODO: Loop through graph.node() and populate our internal Graph
        
        return graph;
    }

} // namespace minitrt