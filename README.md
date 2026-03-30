# Mini-TensorRT: Deep Learning Graph Compiler

## Overview
Mini-TensorRT is a custom, from-scratch deep learning graph compiler written in C++. It is designed to parse ONNX computational graphs, lower them into an optimized Intermediate Representation (IR), apply graph-level optimizations (like operator fusion and constant folding), and execute the network via a custom backend engine.

## Project Architecture & Work Division

To facilitate parallel development, the architecture is decoupled into two primary domains:

### 1. Frontend & Infrastructure (Graph Parsing & IR)
* **Goal:** Ingest binary model files and construct the in-memory graph.
* **Responsibilities:** Protocol Buffer integration, ONNX deserialization, memory management for tensors, and graph topology construction.

### 2. Middle-end & Backend (Optimization & Execution)
* **Goal:** Rewrite the graph for performance and execute the math.
* **Responsibilities:** Graph traversal passes (fusion, folding), memory bandwidth optimization, and the execution engine (CPU/AVX/BLAS).

## File Manifest & Directory Structure

```text
mini-tensorrt/
├── CMakeLists.txt         # Build system configuration, links Protobuf and source files.
├── README.md              # Project documentation and roadmap.
├── .gitignore             # Ignores /build directory and .onnx binaries.
│
├── include/               # Public C++ headers (Declarations)
│   ├── ir.hpp             # Defines the `Graph`, `Node`, and `Tensor` memory structures.
│   ├── parser.hpp         # Defines the `ONNXParser` class for reading the .onnx file.
│   ├── optimizer.hpp      # Defines graph transformation passes (fusion, constant folding).
│   └── backend.hpp        # Defines the execution engine interface and hardware mappings.
│
├── src/                   # C++ source code (Implementations)
│   ├── main.cpp           # Entry point: handles CLI args, initializes parser, and triggers execution.
│   ├── ir.cpp             # Implements graph traversal, node connection, and tensor memory allocation.
│   ├── parser.cpp         # Implements the Protobuf decoding logic to populate the IR.
│   ├── optimizer.cpp      # Implements the logic to merge or pre-compute specific IR nodes.
│   └── backend.cpp        # Implements the actual math operations (e.g., Matrix Multiplication, ReLU).
│
├── models/                # Directory for storing test models
│   └── dummy_model.onnx   # A simple exported model (e.g., a single linear layer) for initial testing.
│
└── scripts/               # Python utility scripts
    └── export_onnx.py     # A short PyTorch script to generate `.onnx` files for the C++ engine to ingest.