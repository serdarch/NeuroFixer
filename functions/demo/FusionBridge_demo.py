import torch
from neurofixer.modules import FusionBridge

x1 = torch.randn(1, 128, 32, 32)
x2 = torch.randn(1, 64, 16, 16)
module = FusionBridge([128, 64], out_channels=128, num_classes=19)
y, pred, weights = module([x1, x2], return_prediction=True)
print({"output": tuple(y.shape), "prediction": tuple(pred.shape), "weights": tuple(weights.shape)})
