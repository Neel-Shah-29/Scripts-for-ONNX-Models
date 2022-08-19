import torch
import torch.nn as nn
m = nn.MaxPool1d(3, stride=1)
input = torch.randn(1,6,10)
output = m(input)
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
            m,
            input,
            "MaxPool1d" + ".onnx",
            export_params=True
    )