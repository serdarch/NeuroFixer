import torch
from neurofixer.modules import EncodingGate

x = torch.randn(1, 64, 32, 32)
module = EncodingGate(64, num_classes=19)
y, pred, controls = module(x, return_prediction=True)
print({"input": tuple(x.shape), "output": tuple(y.shape), "prediction": tuple(pred.shape)})
