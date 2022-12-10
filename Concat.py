import onnxruntime
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import TensorProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer  
from onnx.helper import make_node
input_value_infos = [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, (220, 220, 3))]
output_value_infos = [
            onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, (220, 220, 3))]
body_graph = helper.make_graph(
            [make_node("Identity", ["input"], ["output"])],
            "body_graph",
            input_value_infos,
            output_value_infos,
        )
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None,None,3])

graph = onnx.helper.make_graph(
    [
        make_node(
            "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
        ),
        make_node("SequenceMap", ["in_sequence"], ["shapes"], body=body_graph),
    ],"Concat",
    [
        ("input1", TensorProto.FLOAT, (220, 310, 3)),
        ("input2", TensorProto.FLOAT, (110, 210, 3)),
        ("input3", TensorProto.FLOAT, (90, 110, 3)),
    ],
    [Y]
)

 # Create the model (ModelProto)
model_def = onnx.helper.make_model(graph, producer_name="onnx-example")
model_def.opset_import[0].version = 13

model_def = onnx.shape_inference.infer_shapes(model_def)

onnx.checker.check_model(model_def)

onnx.save(model_def, "ConcatFromSequence.onnx")