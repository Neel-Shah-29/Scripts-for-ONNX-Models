# import torch
# import torch.nn as nn
# import numpy as np
# class TinyModel(torch.nn.Module):

#     def forward(self, a, b):
#         s1 = a.size()
#         c = torch.reshape(b,s1)
#         d = torch.concat([a,c])
#         return d

# tinymodel = TinyModel()

# x1 = np.array([[1,2,3],[4,5,6]])
# x2 = np.array([11,12,13,14,15,16])  

# a=torch.FloatTensor(x1)
# b=torch.FloatTensor(x2)
# s1 = a.size()
# c = torch.reshape(b,s1)
# d = torch.concat([a,c])
# output = d
# print("This are the input tensors:"+str(a)+" & "+str(b))
# print("----------------------------------------------------")
# print("This is the output:", output)
# saveOnnx=True
# loadModel=False
# savePtModel = False


# if savePtModel :
#     torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

# if saveOnnx:
#     torch.onnx.export(
#             tinymodel,
#             (a,b),
#             "Concat_Shape" + ".onnx",
#             export_params=True
#     )

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
X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 6])

# Create one input (ValueInfoProto)
X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [2, 3])

# Create one output (ValueInfoProto)
Y1 = helper.make_tensor_value_info('Y1', TensorProto.INT64, [2])

# Create one output (ValueInfoProto)
Y2 = helper.make_tensor_value_info('Y2', TensorProto.INT64, [2,3])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y1', TensorProto.INT64, [4,3])


Shape_node = onnx.helper.make_node(
    name="S1",  # Name is optional.
    op_type="Shape",
    inputs=['X1'],
    outputs=['Y1']
)

Reshape_node = onnx.helper.make_node(
    name="RS1",  # Name is optional.
    op_type="Reshape",
    inputs=['X2','S1'],
    outputs=['Y2']
)

Concat_node = onnx.helper.make_node(
    name="C1",  # Name is optional.
    op_type="Concat",
    inputs=['X1','RS1'],
    outputs=['Y']
)
# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [Shape_node,Reshape_node,Concat_node],
    'test-model',
    [X1,X2],
    [Y1,Y2,Y],
)

 # Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
model_def.opset_import[0].version = 13

model_def = onnx.shape_inference.infer_shapes(model_def)

onnx.checker.check_model(model_def)

onnx.save(model_def, "Concat_Rehape_Shape.onnx")