import torch


def crps_discrete_from_probs(probs, y_true_mm, bins):
    """
    probs: [B, K]  (probabilidades por bin; soma=1)
    y_true_mm: [B] (alvo em mm, acumulado 4h)
    bins: [K]      (valores y_k crescentes, em mm)
    """
    # CDF prevista
    probs = probs / probs.sum(dim=-1, keepdim=True)
    F_pred = probs.cumsum(dim=-1)  # [B, K]

    # CDF-verdade (degrau em x): T_k = 1{ y_k >= x }
    T = (
        y_true_mm.unsqueeze(1).ge(bins.unsqueeze(0))
    ).float()  # (bins.unsqueeze(0) <= y_true_mm.unsqueeze(1)).float()  # [B, K]

    # Weights Δ_k (larguras)
    delta = torch.diff(bins, prepend=bins[:1])  # useless as bins have uniform width

    return ((F_pred - T) ** 2 * delta.unsqueeze(0)).sum(dim=-1).mean()


def crps_loss(y_hat: torch.Tensor, y: torch.Tensor):
    probs = torch.softmax(y_hat, dim=-1)
    m = y.mean(dim=(2, 3))  # [B, 16] média espacial por slot (mm/h)
    y_true_mm = m.sum(dim=1) / 4.0  # [B]  acum. 4h em mm
    return crps_discrete_from_probs(
        probs,
        y_true_mm,
        bins=torch.arange(0.0, 512.0 + 4, 4.0, device=y.device),
    )
