import matplotlib.pyplot as plt
import pandas as pd
import io

# Paste the data from your terminal here (or modify to read from a file)
csv_data = """OpType,Latency_us
ConvRelu,450.5
MaxPool,120.2
Reshape,5.1
Gemm,85.3"""

df = pd.read_csv(io.StringIO(csv_data))

plt.figure(figsize=(10, 6))
bars = plt.bar(df['OpType'], df['Latency_us'], color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])

plt.title('Mini-TensorRT Inference Latency per Operator', fontsize=14, fontweight='bold')
plt.ylabel('Latency (microseconds)', fontsize=12)
plt.xlabel('Operator Type', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}us', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../images/benchmark_results.png')
print("Benchmark graph saved to images/benchmark_results.png")