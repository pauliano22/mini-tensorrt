#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace minitrt {

    // ==========================================
    // Tensor: Represents N-dimensional data
    // ==========================================
    class Tensor {
    public:
        std::string name;
        std::vector<int64_t> shape;
        
        // The actual numerical data. We use float for prototype simplicity.
        // Industrial engines use void* to support float16, int8, etc.
        std::vector<float> data; 

        // Constructor
        Tensor(const std::string& name, const std::vector<int64_t>& shape);

        // Utility to compute total number of elements (e.g., 3x224x224)
        size_t get_size() const;

        // Utility to print tensor info for debugging
        void print() const;

        size_t elements() const {
            if (shape.empty()) return 0;
            size_t total = 1;
            for (auto dim : shape) {
                total *= dim;
            }
            return total;
        }
    };

    // ==========================================
    // Node: Represents a math operation
    // ==========================================
    class Node {
    public:
        std::string name;
        std::string op_type; // e.g., "Conv", "Relu", "Add"
        
        // Pointers to the data this operation reads from and writes to
        std::vector<std::shared_ptr<Tensor>> inputs;
        std::vector<std::shared_ptr<Tensor>> outputs;
        
        // Attributes for specific ops (e.g., kernel_size, stride, padding)
        std::unordered_map<std::string, int64_t> int_attributes;
        std::unordered_map<std::string, std::vector<int64_t>> ints_attributes;

        // Constructor
        Node(const std::string& name, const std::string& op_type);

        // Utility to link a tensor as an input to this node
        void add_input(std::shared_ptr<Tensor> tensor);

        // Utility to link a tensor as an output of this node
        void add_output(std::shared_ptr<Tensor> tensor);

        // Utility to print node info for debugging
        void print() const;
    };

    // ==========================================
    // Graph: The container for the entire model
    // ==========================================
    class Graph {
    public:
        std::string model_name;
        
        // Look up a tensor by name, or create a blank one if it doesn't exist
        std::shared_ptr<Tensor> get_or_create_tensor(const std::string& name);

        // Fast lookup table
        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensor_map;

        // Master lists holding all operations and data buffers in memory
        std::vector<std::shared_ptr<Node>> nodes;
        std::vector<std::shared_ptr<Tensor>> tensors;

        // Constructor
        Graph(const std::string& name);

        // Graph builder methods
        void add_node(std::shared_ptr<Node> node);
        void add_tensor(std::shared_ptr<Tensor> tensor);

        // High-level summary of the loaded model
        void print_summary() const;
    };

} // namespace minitrt