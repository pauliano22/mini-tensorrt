#include "optimizer.hpp"
#include <iostream>

namespace minitrt {

    Optimizer::Optimizer(std::shared_ptr<Graph> graph) : target_graph(graph) {}

    void Optimizer::optimize() {
        std::cout << "[Optimizer] Starting optimization passes...\n";
        pass_constant_folding();
        pass_operator_fusion();
        pass_dead_code_elimination();
        std::cout << "[Optimizer] Graph optimized.\n";
    }

    void Optimizer::pass_constant_folding() {
        // TODO: Iterate through target_graph->nodes
        // If a node's inputs are all constant tensors (weights), compute 
        // the result now, create a new constant tensor, and delete the node.
    }

    void Optimizer::pass_operator_fusion() {
        // TODO: Look for patterns like Node A (Conv) -> Node B (ReLU).
        // Replace with Node C (ConvReLU) and rewire the input/output pointers.
    }

    void Optimizer::pass_dead_code_elimination() {
        // TODO: Remove nodes whose outputs are never read by another node.
    }

} // namespace minitrt