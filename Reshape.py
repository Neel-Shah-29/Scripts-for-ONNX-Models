import torch
import torch.nn as nn

class TinyModel(torch.nn.Module):

    def forward(self,x):
        
        return torch.reshape(x,(2, 2))

tinymodel = TinyModel()

input = torch.arange(4)
output = torch.reshape(input, (2, 2))

print("This is the input:",input)
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
            input,
            "Reshape" + ".onnx",
            export_params=True
    )