import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, start, limit, delta):
        return torch.arange(start, limit, delta)

tinymodel = TinyModel()

start = 3
limit = 10
delta = 2

a=torch.arange(start, limit, delta)
 
output = a

print("This are the input scalars:"+str(start) +  str(limit) + str(delta))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,(start, limit, delta),
            "Range" + ".onnx",
            export_params=True
    )