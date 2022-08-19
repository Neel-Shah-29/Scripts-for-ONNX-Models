import torch
import torch.nn as nn


a=torch.tensor([1,2,3.])
def f1(a):
    a_sz = a.size()[0]
    b=torch.ones(a_sz)
    return b

def f2(a):
    a_sz = a.size()[0]
    a_sz = int(a_sz)
    b=torch.ones(a_sz)
    return b

def get_onnx_graph(f, input):
    trace, z = torch.jit.get_trace_graph(f, input)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    return trace

print("f1/f2")
print(get_onnx_graph(f1, (a)))
print(get_onnx_graph(f2, (a)))

