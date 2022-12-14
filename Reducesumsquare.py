import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, x):
        y = np.sum(np.square(x),0,float)
        return torch.FloatTensor(y)

tinymodel = TinyModel()
axes=None
a = [[1.0,1.0,1.0],[1.0,1.0,1.0]]
x = torch.FloatTensor(a)

y =  np.sum(np.square(a),0,float)
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
            tinymodel,(a),
            "Reducesumsquare" + ".onnx",
            export_params=True
    )