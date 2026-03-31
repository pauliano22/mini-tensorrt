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
        x = torch.flatten(x, 1) # ONNX exports this as a 'Reshape' node
        x = self.fc(x)          # ONNX exports this as a 'Gemm' node
        return x

# Instantiate and create a dummy input (Batch=1, Channel=1, H=28, W=28)
model = FullMNISTClassifier()
dummy_input = torch.randn(1, 1, 28, 28)

# Export the graph!
torch.onnx.export(
    model, 
    dummy_input, 
    "../models/mnist_cnn.onnx", 
    input_names=["input_image"], 
    output_names=["predictions"]
)

print("Full MNIST CNN exported to ../models/mnist_cnn.onnx!")