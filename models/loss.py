import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, batch_size, nneg):
        super(InfoNCELoss, self).__init__()
        self.batch_size = batch_size
        self.nneg = nneg
    
    def forward(self, y_hat, y):
        p = torch.exp(-y_hat[:self.batch_size])
        pneg = torch.sum(torch.exp(-y_hat[self.batch_size:]).reshape(self.batch_size, self.nneg), axis=-1)
        return torch.sum(-torch.log(p / (p + pneg)))
