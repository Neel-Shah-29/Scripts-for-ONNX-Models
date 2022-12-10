import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, x):
        return torch.prod(x,1)

tinymodel = TinyModel()

x = np.array([[[5,2,3],[5,5,4]]])

a=torch.FloatTensor(x)
 
output = torch.prod(a,1)

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
            tinymodel,(a),
            "ReduceProd" + ".onnx",
            export_params=True
    )
