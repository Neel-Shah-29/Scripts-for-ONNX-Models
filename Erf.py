import torch
import torch.nn as nn

class TinyModel(torch.nn.Module):

    def forward(self, x):
        
        return torch.erf(x)

tinymodel = TinyModel()

input = torch.randn(12)
output = torch.erf(input)
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
            "Erf" + ".onnx",
            export_params=True
    )
