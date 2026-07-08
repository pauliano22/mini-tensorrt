// WebAssembly entry point for the browser demo (pauliacobucci.com/demo.html).
//
// This wrapper reuses the engine's real IR, optimizer, and backend kernels —
// the only thing it skips is the protobuf ONNX parser (protobuf is painful to
// cross-compile to WASM), so the 5-node MNIST graph is constructed directly
// and the trained weights are baked in via weights.h.
//
// Build: see build.sh in this directory.

#include <memory>

#include "backend.hpp"
#include "ir.hpp"
#include "optimizer.hpp"
#include "weights.h"

extern "C" {

// Classify a 28x28 grayscale image (784 floats, 0.0-1.0, row-major).
// Writes the 10 raw logits to logits_out and returns the argmax digit.
int predict(const float* pixels, float* logits_out) {
    auto graph = std::make_shared<minitrt::Graph>("mnist_wasm");

    // --- Tensors ---
    auto input = graph->get_or_create_tensor("input_image");
    input->shape = {1, 1, 28, 28};
    input->data.assign(pixels, pixels + 784);

    auto conv_w = graph->get_or_create_tensor("conv1.weight");
    conv_w->shape = {4, 1, 3, 3};
    conv_w->data.assign(CONV1_W, CONV1_W + 36);

    auto conv_b = graph->get_or_create_tensor("conv1.bias");
    conv_b->shape = {4};
    conv_b->data.assign(CONV1_B, CONV1_B + 4);

    auto fc_w = graph->get_or_create_tensor("fc.weight");
    fc_w->shape = {10, 784};
    fc_w->data.assign(FC_W, FC_W + 7840);

    auto fc_b = graph->get_or_create_tensor("fc.bias");
    fc_b->shape = {10};
    fc_b->data.assign(FC_B, FC_B + 10);

    // The Reshape kernel flattens by element count; its shape input is unused
    auto reshape_shape = graph->get_or_create_tensor("reshape_shape");
    reshape_shape->shape = {2};

    // --- Nodes (same topology the ONNX parser produces) ---
    auto conv = std::make_shared<minitrt::Node>("conv1", "Conv");
    conv->add_input(input);
    conv->add_input(conv_w);
    conv->add_input(conv_b);
    conv->add_output(graph->get_or_create_tensor("conv_out"));
    graph->add_node(conv);

    auto relu = std::make_shared<minitrt::Node>("relu", "Relu");
    relu->add_input(graph->get_or_create_tensor("conv_out"));
    relu->add_output(graph->get_or_create_tensor("relu_out"));
    graph->add_node(relu);

    auto pool = std::make_shared<minitrt::Node>("pool", "MaxPool");
    pool->add_input(graph->get_or_create_tensor("relu_out"));
    pool->add_output(graph->get_or_create_tensor("pool_out"));
    graph->add_node(pool);

    auto reshape = std::make_shared<minitrt::Node>("reshape", "Reshape");
    reshape->add_input(graph->get_or_create_tensor("pool_out"));
    reshape->add_input(reshape_shape);
    reshape->add_output(graph->get_or_create_tensor("flat_out"));
    graph->add_node(reshape);

    auto gemm = std::make_shared<minitrt::Node>("fc", "Gemm");
    gemm->add_input(graph->get_or_create_tensor("flat_out"));
    gemm->add_input(fc_w);
    gemm->add_input(fc_b);
    gemm->add_output(graph->get_or_create_tensor("logits"));
    graph->add_node(gemm);

    // --- Optimize (Conv+ReLU fusion) and execute, exactly like the CLI ---
    minitrt::Optimizer optimizer;
    optimizer.run_passes(graph);

    minitrt::ExecutionEngine engine(graph);
    engine.run();

    auto logits = graph->get_or_create_tensor("logits");
    int best = 0;
    for (int i = 0; i < 10; ++i) {
        logits_out[i] = logits->data[i];
        if (logits->data[i] > logits->data[best]) best = i;
    }
    return best;
}

} // extern "C"
