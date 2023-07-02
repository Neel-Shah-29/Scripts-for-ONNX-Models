import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, a):
        
        return torch.log(a)

tinymodel = TinyModel()

# Create a tensor
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

a=torch.FloatTensor(x)

output = torch.log(a)
print("This are the input tensors:"+str(a))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,
            (a),
            "Log" + ".onnx",
            export_params=True
    )