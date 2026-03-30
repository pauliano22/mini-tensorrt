#include "ir.hpp"

namespace minitrt {

    // ==========================================
    // Tensor Implementation
    // ==========================================

    Tensor::Tensor(const std::string& n, const std::vector<int64_t>& s) 
        : name(n), shape(s) {
        // Automatically allocate memory for the tensor data initialized to 0.0
        // In a production C++ engine, you might defer this to save RAM 
        // until the exact moment the tensor is needed, but we allocate immediately for simplicity.
        size_t total_elements = get_size();
        if (total_elements > 0) {
            data.resize(total_elements, 0.0f);
        }
    }

    size_t Tensor::get_size() const {
        if (shape.empty()) return 0;
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        return size;
    }

    void Tensor::print() const {
        std::cout << "Tensor: " << name << " | Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << "] | Elements: " << get_size() << "\n";
    }

    // ==========================================
    // Node Implementation
    // ==========================================

    Node::Node(const std::string& n, const std::string& op) 
        : name(n), op_type(op) {}

    void Node::add_input(std::shared_ptr<Tensor> tensor) {
        inputs.push_back(tensor);
    }

    void Node::add_output(std::shared_ptr<Tensor> tensor) {
        outputs.push_back(tensor);
    }

    void Node::print() const {
        std::cout << "Node: " << name << " | Op: " << op_type 
                  << " | Inputs: " << inputs.size() 
                  << " | Outputs: " << outputs.size() << "\n";
    }

    // ==========================================
    // Graph Implementation
    // ==========================================

    Graph::Graph(const std::string& name) : model_name(name) {}

    void Graph::add_node(std::shared_ptr<Node> node) {
        nodes.push_back(node);
    }

    void Graph::add_tensor(std::shared_ptr<Tensor> tensor) {
        tensors.push_back(tensor);
    }

    void Graph::print_summary() const {
        std::cout << "\n========================================\n";
        std::cout << "Graph Summary: " << model_name << "\n";
        std::cout << "========================================\n";
        
        std::cout << "Total Tensors: " << tensors.size() << "\n";
        for (const auto& t : tensors) {
            std::cout << "  - ";
            t->print();
        }
        
        std::cout << "\nTotal Nodes: " << nodes.size() << "\n";
        for (const auto& n : nodes) {
            std::cout << "  - ";
            n->print();
        }
        std::cout << "========================================\n\n";
    }

} // namespace minitrt