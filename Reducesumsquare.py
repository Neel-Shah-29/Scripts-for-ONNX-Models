import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, x):
        y = np.sum(np.square(x),axis=axes, keepdims= 1)
        return torch.FloatTensor(y)

tinymodel = TinyModel()
axes=None
a = np.array([[[1,1,1],[1,1,1]]])
x = torch.FloatTensor(a)
y =  np.sum(np.square(a),axis=axes, keepdims= 1)
print(y)
# a = torch.FloatTensor(y)
 
output = torch.FloatTensor(y)

print("This are the input tensors:"+str(x))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,(x),
            "Reducesumsquare" + ".onnx",
            export_params=True
    )