#include "optimizer.hpp"
#include <iostream>

namespace minitrt {

    void Optimizer::run_passes(std::shared_ptr<Graph> graph) {
        std::cout << "[Optimizer] Running Graph Optimization Passes...\n";
        fuse_conv_relu(graph);
    }

    void Optimizer::fuse_conv_relu(std::shared_ptr<Graph> graph) {
        for (size_t i = 0; i < graph->nodes.size(); ++i) {
            // Safety check for look-ahead
            if (i + 1 >= graph->nodes.size()) break;

            auto current = graph->nodes[i];
            auto next = graph->nodes[i+1];

            if (current->op_type == "Conv" && next->op_type == "Relu") {
                std::cout << "[Optimizer] Found pattern: Conv -> Relu. Fusing into 'ConvRelu'...\n";
                
                current->op_type = "ConvRelu";
                current->name = "fused_" + current->name;

                // Re-link the Conv node's output to the Relu node's output
                current->outputs = next->outputs;

                // Remove the Relu node
                graph->nodes.erase(graph->nodes.begin() + i + 1);
                
                std::cout << "[Optimizer] Fusion successful.\n";
            }
        }
    }
}