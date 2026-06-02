import torch
from neurofixer.modules import EncodingModule

x = torch.randn(1, 64, 32, 32)
module = EncodingModule(64, latent_dim=128, num_classes=19, out_size=(16, 16))
y, pred, controls = module(x, return_prediction=True)
print({"input": tuple(x.shape), "output": tuple(y.shape), "prediction": tuple(pred.shape)})
