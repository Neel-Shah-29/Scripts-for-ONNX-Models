import torch
import torch.nn as nn
m = nn.Softmax(dim=1)
input = torch.randn(4, 5)
output = m(input)

print("This is the input:",input)
print("----------------------------------------------------------")
print("This is the output:",output)