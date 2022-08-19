import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, x, y):
        
        return torch.add(x,y)

tinymodel = TinyModel()

# a = torch.FloatTensor((1, 2, 3),(4,5,6))
# b = torch.FloatTensor((1,2,3))
 
# # output = torch.add(a, b)
# # import torch
# a = torch.from_numpy(numpy.array([[1.0,2.0,3.0], [4.0, 5.0, 6.0]]))
# b = torch.from_numpy(numpy.array([1.0,2.0,3.0],[4.0,5.0,6.0]))
x = np.array([[[1,2,3],[3,4,5]]])
y = np.array([[5,6,7],[8,9,10]])  #note the extra []
# print(x.shape,y.shape)
# z = x+y
# print(z.shape,z)
a=torch.FloatTensor(x)
b=torch.FloatTensor(y)

print(a)
print(b)
print(torch.add(a, b))

# print("This are the input tensors:"+str(a)+"&"+str(b))
# print("----------------------------------------------------")
# print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,(a,b),
            "Add" + ".onnx",
            export_params=True
    )