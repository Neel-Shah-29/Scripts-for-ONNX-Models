import onnxruntime
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import TensorProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

# Create one input (ValueInfoProto)
X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [2])

# Create one input (ValueInfoProto)
X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [2])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2,2])

input = onnx.helper.make_tensor_sequence_value_info('X',TensorProto.FLOAT,[2,None])

Shape_node = onnx.helper.make_node(
    name="ConcatFromSequence",  # Name is optional.
    op_type="ConcatFromSequence",
    inputs=['X'],
    outputs=['Y'],
    axis=0,
    new_axis=0
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [Shape_node],
    'test-model',
    [input],
    [Y],
)

 # Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
model_def.opset_import[0].version = 13

model_def = onnx.shape_inference.infer_shapes(model_def)

onnx.checker.check_model(model_def)

onnx.save(model_def, "ConcatFromSequence.onnx")