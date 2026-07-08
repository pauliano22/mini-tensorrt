import gzip
import os
import struct
import urllib.request

import numpy as np
import onnx
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# MNIST loading (no torchvision dependency): download the raw IDX files once
# and cache them under scripts/mnist_data/.
# ---------------------------------------------------------------------------
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def load_mnist():
    cache_dir = os.path.join(os.path.dirname(__file__), "mnist_data")
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {}
    for key, fname in MNIST_FILES.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(MNIST_MIRROR + fname, path)
        with gzip.open(path, "rb") as f:
            if "images" in key:
                _, n, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols)
                # Scale to [0, 1] to match the engine's image loader (pixel / 255)
                arrays[key] = torch.from_numpy(data.copy()).float() / 255.0
            else:
                _, n = struct.unpack(">II", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
                arrays[key] = torch.from_numpy(labels.copy()).long()
    return arrays

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
        x = x.reshape(-1, 4 * 14 * 14)
        x = self.fc(x)              # ONNX exports this as a 'Gemm' node
        return x

# ---------------------------------------------------------------------------
# Train: this network is tiny (~8k parameters), a couple of epochs on CPU is
# enough to make the README demo genuinely recognize digits.
# ---------------------------------------------------------------------------
torch.manual_seed(22)
data = load_mnist()
model = FullMNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_x, train_y = data["train_images"], data["train_labels"]
batch_size = 128
for epoch in range(2):
    perm = torch.randperm(len(train_x))
    for i in range(0, len(train_x), batch_size):
        idx = perm[i : i + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(train_x[idx]), train_y[idx])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        preds = model(data["test_images"]).argmax(dim=1)
        accuracy = (preds == data["test_labels"]).float().mean().item()
        model.train()
    print(f"Epoch {epoch + 1}: test accuracy {accuracy:.2%}")

model.eval()
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_cnn.onnx")

# ---------------------------------------------------------------------------
# Export the graph with the legacy exporter (the dynamo exporter changes op
# choices and graph structure in ways the engine's parser does not expect).
# ---------------------------------------------------------------------------
dummy_input = torch.randn(1, 1, 28, 28)
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

print(f"Trained MNIST CNN ({accuracy:.2%} test accuracy) exported to ../models/mnist_cnn.onnx!")
