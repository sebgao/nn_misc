import torch
import torch.nn as nn

def train_model(model, x, y, loss_func, optimizer):
    y_pred = model.forward(x)
    loss = loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    