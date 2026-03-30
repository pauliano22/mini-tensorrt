#pragma once
#include <string>
#include <memory>
#include "ir.hpp"

namespace minitrt {

    class ONNXParser {
    public:
        // Constructor takes the path to the .onnx file
        ONNXParser(const std::string& file_path);

        // Parses the file and returns a populated Graph object
        std::shared_ptr<Graph> parse();

    private:
        std::string model_path;
        
        // Helper to map ONNX data types to our internal types
        void parse_tensor(/* ONNX TensorProto object */);
        
        // Helper to map ONNX nodes to our internal Node class
        void parse_node(/* ONNX NodeProto object */);
    };

} // namespace minitrt