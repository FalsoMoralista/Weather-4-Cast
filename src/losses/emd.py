import torch


class EMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.softmax(dim=-1)
        input = input.cumsum(dim=-1)
        m = target.mean(dim=(2, 3))
        y_true_mm = m.sum(dim=1) / 4.0
        bins = torch.linspace(0.0, 128.0, 100, device=y_true_mm.device)
        T = (bins.unsqueeze(0).ge(y_true_mm.unsqueeze(1))).float()
        emd = torch.sum(torch.abs(input - T))
        return emd
