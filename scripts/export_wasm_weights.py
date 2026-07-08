"""Regenerate wasm/weights.h from the trained model in models/mnist_cnn.onnx."""
import os

import numpy as np
import onnx
from onnx import numpy_helper

REPO = os.path.join(os.path.dirname(__file__), "..")
NAMES = {
    "conv1.weight": "CONV1_W",
    "conv1.bias": "CONV1_B",
    "fc.weight": "FC_W",
    "fc.bias": "FC_B",
}

model = onnx.load(os.path.join(REPO, "models", "mnist_cnn.onnx"))
onnx.load_external_data_for_model(model, os.path.join(REPO, "models"))

out_path = os.path.join(REPO, "wasm", "weights.h")
with open(out_path, "w") as f:
    f.write("// Auto-generated from models/mnist_cnn.onnx by scripts/export_wasm_weights.py\n#pragma once\n\n")
    for init in model.graph.initializer:
        if init.name in NAMES:
            arr = numpy_helper.to_array(init).astype(np.float32).ravel()
            vals = ",".join(f"{v:.8e}f" for v in arr)
            f.write(f"static const float {NAMES[init.name]}[{len(arr)}] = {{{vals}}};\n\n")

print(f"Wrote {out_path}")
