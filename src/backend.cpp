#include "backend.hpp"
#include <iostream>

namespace minitrt {

    ExecutionEngine::ExecutionEngine(std::shared_ptr<Graph> graph) 
        : execution_graph(graph) {}

    void ExecutionEngine::set_input(const std::string& tensor_name, const std::vector<float>& data) {
        // TODO: Find the input tensor by name and copy 'data' into it
    }

    void ExecutionEngine::run() {
        std::cout << "[Backend] Executing graph...\n";
        
        // Topological sort ensures we execute nodes in the correct order
        for (const auto& node : execution_graph->nodes) {
            if (node->op_type == "Relu") {
                execute_relu(node);
            } else if (node->op_type == "MatMul") {
                execute_matmul(node);
            }
            // Add more operators here
        }
    }

    std::vector<float> ExecutionEngine::get_output(const std::string& tensor_name) {
        // TODO: Return the data vector from the final output tensor
        return {}; 
    }

    void Execution.execute_relu(std::shared_ptr<Node> node) {
        // TODO: Loop through input tensor, apply max(0, x), write to output tensor
    }

    void ExecutionEngine::execute_matmul(std::shared_ptr<Node> node) {
        // TODO: Implement matrix multiplication (naive loops first, then BLAS/AVX)
    }

} // namespace minitrt