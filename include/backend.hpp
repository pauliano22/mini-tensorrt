#pragma once
#include <memory>
#include <unordered_map>
#include "ir.hpp"

namespace minitrt {

    class ExecutionEngine {
    public:
        ExecutionEngine(std::shared_ptr<Graph> graph);

        // Feed input data (e.g., an image) into the network
        void set_input(const std::string& tensor_name, const std::vector<float>& data);

        // Execute the graph sequentially
        void run();

        // Retrieve the final prediction
        std::vector<float> get_output(const std::string& tensor_name);

    private:
        std::shared_ptr<Graph> execution_graph;

        // Mathematical kernels
        void execute_relu(std::shared_ptr<Node> node);
        void execute_matmul(std::shared_ptr<Node> node);
        void execute_conv(std::shared_ptr<Node> node);
    };

} // namespace minitrt