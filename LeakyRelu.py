import torch
import torch.nn as nn
m = nn.LeakyReLU(0.1)
input = torch.randn(24)
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
            "LeakyRelu" + ".onnx",
            export_params=True
    )
