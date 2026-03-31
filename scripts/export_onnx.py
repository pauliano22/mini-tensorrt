import torch
import torch.nn as nn
import os

# 1. Define the simplest possible neural network
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # A single Convolution layer followed by a ReLU activation
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# 2. Instantiate the model and set it to evaluation (inference) mode
model = DummyModel()
model.eval() 

# 3. Create a dummy input tensor (Batch Size: 1, Channels: 1, Height: 28, Width: 28)
# Think of this like a single blank 28x28 grayscale image
dummy_input = torch.randn(1, 1, 28, 28)

# 4. Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(output_dir, exist_ok=True)
export_path = os.path.join(output_dir, 'dummy_model.onnx')

print(f"Exporting model to {export_path}...")

# 5. Export to ONNX binary format
torch.onnx.export(
    model, 
    dummy_input, 
    export_path, 
    export_params=True,        # Store the trained weights
    opset_version=14,          # Standard ONNX operator set
    do_constant_folding=True,  # PyTorch will do some basic optimizations first
    input_names=['input_image'], 
    output_names=['output_activation']
)

print("Success! The test oracle has generated the ONNX file.")