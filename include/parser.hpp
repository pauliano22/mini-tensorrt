#pragma once
#include <string>
#include <memory>
#include "ir.hpp"
#include "onnx.pb.h" // The file Google just generated for us

namespace minitrt {

    class ONNXParser {
    public:
        ONNXParser(const std::string& file_path);
        std::shared_ptr<Graph> parse();

    private:
        std::string model_path;
        
        // Helper to map ONNX data types to our internal types
        void parse_tensor(const onnx::TensorProto& onnx_tensor, std::shared_ptr<Graph> graph);
        
        // Helper to map ONNX nodes to our internal Node class
        void parse_node(const onnx::NodeProto& onnx_node, std::shared_ptr<Graph> graph);
    };

} // namespace minitrt