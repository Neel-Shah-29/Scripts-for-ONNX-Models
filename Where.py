import torch
import torch.nn as nn
import torch.onnx as onnx

class WhereModule(nn.Module):
    def forward(self, condition, x, y):
        return torch.where(condition, x, y)

# Define the inputs
condition = torch.tensor([[True, False], [False, True]])
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.tensor([[5., 6.], [7., 8.]])

# Instantiate the custom module
model = WhereModule()

# Execute the where operator
output = model(condition, x, y)

# Export the model to ONNX
dummy_input = (condition, x, y)
onnx_filename = "where.onnx"
onnx.export(model, dummy_input, onnx_filename)

print(f"ONNX model saved to: {onnx_filename}")