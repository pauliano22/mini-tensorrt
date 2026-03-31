#pragma once
#include <memory>
#include "ir.hpp"

namespace minitrt {

    class Optimizer {
    public:
        Optimizer() = default; // Allows "minitrt::Optimizer optimizer;" in main
        
        void run_passes(std::shared_ptr<Graph> graph);

    private:
        void fuse_conv_relu(std::shared_ptr<Graph> graph);
    };

}