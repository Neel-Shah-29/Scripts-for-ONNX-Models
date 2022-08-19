import onnx
import os

# Load the ONNX model
onnx_model = onnx.load('Generator.onnx')
print(onnx_model.graph.node)

