#pragma once
#include <memory>
#include "ir.hpp"

namespace minitrt {

    class Optimizer {
    public:
        Optimizer(std::shared_ptr<Graph> graph);

        // Runs all optimization passes sequentially
        void optimize();

    private:
        std::shared_ptr<Graph> target_graph;

        // Pass 1: Pre-compute math that only involves static weights
        void pass_constant_folding();

        // Pass 2: Merge adjacent nodes (e.g., Conv2D -> ReLU into ConvReLU)
        // to save memory bandwidth
        void pass_operator_fusion();
        
        // Pass 3: Remove nodes that don't contribute to the final output
        void pass_dead_code_elimination();
    };

} // namespace minitrt