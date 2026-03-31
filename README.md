# Mini-TensorRT

**A zero-dependency, optimizing deep learning inference engine written from scratch in C++.**

Mini-TensorRT is a custom inference compiler built to explore machine learning systems and hardware-software co-design. It ingests standard ONNX models, builds an Intermediate Representation (IR) graph, applies compiler optimizations to reduce memory bottlenecks, and executes operations using custom C++ math kernels.

![Computational Graph Overview](images/netron_graph.png)
*Visualizing the ingested ONNX topology prior to operator fusion.*

## Core Architecture

This engine operates independently of large frameworks like PyTorch or OpenCV, relying only on standard C++17 and Google's Protocol Buffers.

* **Frontend Parser:** A custom deserializer that reads binary ONNX files to extract static shapes, topologies, and raw floating-point weights into memory.
* **Intermediate Representation (IR):** Custom `Tensor` and `Graph` classes that handle NCHW multidimensional data and track execution dependencies.
* **Optimizing Compiler:** A graph-traversal pass that mutates the execution plan. It currently implements operator fusion (e.g., merging `Conv` and `ReLU` nodes) to keep data in CPU registers and reduce DRAM round-trips.
* **Backend Engine:** A dynamic execution engine that reads tensor shapes on the fly (dynamic shape inference) and routes data through custom C++ math kernels.

## Supported Operators

The backend currently supports the operations required to run a complete CNN (like LeNet):
* `Conv` (Spatial Convolution with dynamic stride/padding)
* `Relu` (Non-linear Activation)
* `ConvRelu` (Fused Super-Kernel)
* `MaxPool` (Spatial Downsampling)
* `Reshape` (Zero-math memory flattening)
* `Gemm` (General Matrix Multiplication / Linear Layers)

## End-to-End Execution

The engine handles full image preprocessing and inference. Below is the terminal output of the engine parsing a trained MNIST classifier, applying operator fusion, and identifying a handwritten digit from a raw `.png` file.

![Terminal Output of Engine Execution](images/guess_number.png)

## Building and Running

**Dependencies:**
* CMake (>= 3.10)
* Make
* Protocol Buffers (`libprotobuf-dev`, `protobuf-compiler`)

**Build Instructions:**
```bash
mkdir build && cd build
cmake ..
make