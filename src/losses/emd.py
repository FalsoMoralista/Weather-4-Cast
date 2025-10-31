import torch


class EMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.cumsum(dim=-1)
        emd = torch.sum(torch.abs(input - target))
        return emd
