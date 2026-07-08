import os

import onnx
import torch
import torch.nn as nn

class FullMNISTClassifier(nn.Module):
    def __init__(self):
        super(FullMNISTClassifier, self).__init__()
        # 1 input channel (grayscale), 4 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After a 28x28 image is pooled by 2, it becomes 14x14.
        # 4 channels * 14 * 14 = 784 total features
        self.fc = nn.Linear(4 * 14 * 14, 10) # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Static reshape so ONNX emits a 'Reshape' node; torch.flatten exports
        # as 'Flatten' on PyTorch >= 2.9, which the engine does not implement.
        x = x.reshape(1, 4 * 14 * 14)
        x = self.fc(x)              # ONNX exports this as a 'Gemm' node
        return x

# Instantiate and create a dummy input (Batch=1, Channel=1, H=28, W=28)
torch.manual_seed(22)
model = FullMNISTClassifier()
model.eval()
dummy_input = torch.randn(1, 1, 28, 28)

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_cnn.onnx")

# Export the graph with the legacy exporter (the dynamo exporter changes op
# choices and graph structure in ways the engine's parser does not expect).
torch.onnx.export(
    model,
    dummy_input,
    model_path,
    input_names=["input_image"],
    output_names=["predictions"],
    dynamo=False,
)

# The engine's parser only reads initializer tensors, not 'Constant' nodes.
# Fold the Reshape target-shape Constant into a graph initializer.
graph_model = onnx.load(model_path)
for node in [n for n in graph_model.graph.node if n.op_type == "Constant"]:
    tensor = next(a.t for a in node.attribute if a.name == "value")
    tensor.name = node.output[0]
    graph_model.graph.initializer.append(tensor)
    graph_model.graph.node.remove(node)

# Store large weights (fc.weight) in mnist_cnn.onnx.data, keeping the .onnx small.
# Remove stale outputs first: onnx.save appends to an existing external-data file.
os.remove(model_path)
data_path = model_path + ".data"
if os.path.exists(data_path):
    os.remove(data_path)
onnx.save(
    graph_model,
    model_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="mnist_cnn.onnx.data",
    size_threshold=20000,
)
onnx.checker.check_model(model_path)

print("Full MNIST CNN exported to ../models/mnist_cnn.onnx!")
