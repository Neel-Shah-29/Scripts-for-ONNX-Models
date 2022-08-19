import torch
import torch.nn as nn

class TinyModel(torch.nn.Module):

    def forward(self, x, y):
        
        return torch.mul(x,y)

tinymodel = TinyModel()

a = torch.FloatTensor((1, 2))
b = torch.FloatTensor((0, 1))
 
output = torch.mul(a, b)

print("This are the input tensors:"+str(a)+"&"+str(b))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,(a,b),
            "Mul" + ".onnx",
            export_params=True
    )