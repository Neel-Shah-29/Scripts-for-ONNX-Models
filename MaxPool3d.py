import torch
import torch.nn as nn
# pool of square window of size=3, stride=2
m = nn.MaxPool3d(3, stride=2)
# pool of non-square window
m = nn.MaxPool3d((3, 2, 2), stride=(1, 1, 1))
input = torch.randn(1, 1, 3, 3, 3)
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
            "MaxPool3d" + ".onnx",
            export_params=True
    )