#pragma once
#include <memory>
#include <vector>
#include "ir.hpp"

namespace minitrt {

    class ExecutionEngine {
    public:
        ExecutionEngine(std::shared_ptr<Graph> graph);
        void run();

    private:
        std::shared_ptr<Graph> execution_graph;

        // The Kernels
        void execute_relu(std::shared_ptr<Node> node);
        void execute_conv2d(std::shared_ptr<Node> node);
        void execute_conv_relu(std::shared_ptr<Node> node);
        void execute_maxpool(std::shared_ptr<Node> node);
        void execute_reshape(std::shared_ptr<Node> node);
        void execute_gemm(std::shared_ptr<Node> node);
    };

}