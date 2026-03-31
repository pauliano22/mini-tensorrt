#include "parser.hpp"
#include <iostream>
#include <fstream>
#include <cstring> // Required for std::memcpy

namespace minitrt {

    ONNXParser::ONNXParser(const std::string& file_path) : model_path(file_path) {
        // Initialize the Protobuf library
        GOOGLE_PROTOBUF_VERIFY_VERSION;
    }

    std::shared_ptr<Graph> ONNXParser::parse() {
        std::cout << "[Parser] Loading model from: " << model_path << "\n";
        
        // 1. Create our empty target graph
        auto graph = std::make_shared<Graph>("Imported_ONNX_Model");

        // 2. Open the binary file
        std::ifstream input(model_path, std::ios::ate | std::ios::binary);
        if (!input.is_open()) {
            std::cerr << "[Error] Failed to open file: " << model_path << "\n";
            return graph;
        }

        // 3. Read the binary data into the ONNX ModelProto object
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        
        onnx::ModelProto model_proto;
        if (!model_proto.ParseFromIstream(&input)) {
            std::cerr << "[Error] Failed to parse ONNX protobuf.\n";
            return graph;
        }
        
        std::cout << "[Parser] Successfully decoded Protobuf. IR Version: " << model_proto.ir_version() << "\n";

        // 4. Extract the graph layer
        const onnx::GraphProto& onnx_graph = model_proto.graph();

        // 5. Loop through the Initializers (these are the static weights/tensors)
        std::cout << "[Parser] Parsing " << onnx_graph.initializer_size() << " tensors...\n";
        for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
            parse_tensor(onnx_graph.initializer(i), graph);
        }

        // 6. Loop through the Nodes (the math operations like Conv and ReLU)
        std::cout << "[Parser] Parsing " << onnx_graph.node_size() << " nodes...\n";
        for (int i = 0; i < onnx_graph.node_size(); ++i) {
            parse_node(onnx_graph.node(i), graph);
        }

        // Optional: Clean up protobuf memory
        google::protobuf::ShutdownProtobufLibrary();

        return graph;
    }

    void ONNXParser::parse_tensor(const onnx::TensorProto& onnx_tensor, std::shared_ptr<Graph> graph) {
        // 1. Extract the shape
        std::vector<int64_t> shape;
        for (int i = 0; i < onnx_tensor.dims_size(); ++i) {
            shape.push_back(onnx_tensor.dims(i));
        }

        // 2. Create our custom Tensor object
        auto tensor = std::make_shared<Tensor>(onnx_tensor.name(), shape);
        
        // 3. EXTRACT THE REAL WEIGHTS!
        // Check if the protobuf actually contains raw binary data (Proto3 style)
        if (!onnx_tensor.raw_data().empty()) {
            const std::string& raw_data = onnx_tensor.raw_data();
            
            // Calculate how many floats we expect to find (Total Bytes / 4 bytes per float)
            size_t num_floats = raw_data.size() / sizeof(float);
            
            // Allocate the exact amount of memory needed in our tensor
            tensor->data.resize(num_floats);
            
            // The Magic: Copy the raw bytes directly into our floating-point array
            std::memcpy(tensor->data.data(), raw_data.data(), raw_data.size());
            
            std::cout << "  [Parser] Loaded " << num_floats << " trained weights for " << tensor->name << "\n";
        }

        // Add it to our Graph's memory
        graph->add_tensor(tensor);
    }

    void ONNXParser::parse_node(const onnx::NodeProto& onnx_node, std::shared_ptr<Graph> graph) {
        auto node = std::make_shared<Node>(onnx_node.name(), onnx_node.op_type());
        
        // Link Inputs
        for (int i = 0; i < onnx_node.input_size(); ++i) {
            auto tensor_ptr = graph->get_or_create_tensor(onnx_node.input(i));
            node->add_input(tensor_ptr);
        }
        
        // Link Outputs
        for (int i = 0; i < onnx_node.output_size(); ++i) {
            auto tensor_ptr = graph->get_or_create_tensor(onnx_node.output(i));
            node->add_output(tensor_ptr);
        }
        
        graph->add_node(node);
    }

} // namespace minitrt