import torch
import torch.nn as nn
m = nn.AvgPool2d(3, stride=2)
# pool of non-square window
m = nn.AvgPool2d((3, 2), stride=(2, 1))

input = torch.randn(1, 1, 5, 10)
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
            "AvgPool" + ".onnx",
            export_params=True
    )