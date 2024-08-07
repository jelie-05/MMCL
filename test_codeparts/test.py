import torch
import torch.nn as nn

pred = torch.tensor([0.42, 0.2, 0.9, 0.7, 0.3, 0.01, 0.25], dtype=torch.float32)
label = torch.tensor([1, 0, 1, 1, 0, 0, 0], dtype=torch.float32)

loss_fun = nn.BCELoss()

loss = loss_fun(pred, label)

print(loss.item())