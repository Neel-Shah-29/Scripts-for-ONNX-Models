import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, a, new_shape):
        
        return a.expand(new_shape)

tinymodel = TinyModel()

x = np.array([[1],[2],[3]])
y = (3,4) 

a=torch.FloatTensor(x)
new_shape = y
output = a.expand(new_shape)
print("This are the input tensors:"+str(a)+" & output shape is  "+str(new_shape))
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
            (a,new_shape),
            "Expand" + ".onnx",
            export_params=True
    )