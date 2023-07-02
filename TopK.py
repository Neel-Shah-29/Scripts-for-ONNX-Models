import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, tensor, k):
        return torch.topk(tensor,k)

tinymodel = TinyModel()

a=torch.tensor([5, 2, 9, 1, 7, 3, 8, 4, 6])
k=3
output = torch.topk(a, k)

print("This are the input scalars:"+str(a) +  str(k) )
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,(a,k),
            "TopK" + ".onnx",
            export_params=True
    )