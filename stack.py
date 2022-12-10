import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, a, b, c, d):
        
        return torch.cat((a,b,c,d),0)

tinymodel = TinyModel()

x = np.array([1,2,3])
y = np.array([4,2,6]) 
z = np.array([5,6,7])
w = np.array([8,6,7])

a=torch.FloatTensor(x)
b=torch.FloatTensor(y)
c=torch.FloatTensor(z)
d=torch.FloatTensor(w)
output = torch.cat((a, b,c,d),0)
print("This are the input tensors:"+str(a)+str(b)+str(c)+" & "+str(d))
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
            (a,b,c,d),
            "Stack" + ".onnx",
            export_params=True,
            opset_version=11
    )