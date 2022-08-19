import torch
import torch.nn as nn

class TinyModel(torch.nn.Module):

    def forward(self, x):
        
        return torch.neg(x)

tinymodel = TinyModel()

input = torch.randn(12)
output = torch.neg(input)
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
            "Neg" + ".onnx",
            export_params=True
    )