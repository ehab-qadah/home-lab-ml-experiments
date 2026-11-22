"""The model itself: a small configurable MLP.

Deliberately plain. Tabular data of this size is won by feature treatment and
regularisation, not depth — and on a CPU-only box the difference between this and a
transformer is a rounding error in AUC against a 100x training bill.
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(in_dim: int, task: str, n_classes: int | None, cfg: dict) -> MLP:
    out_dim = n_classes if task == "multiclass" else 1
    return MLP(
        in_dim=in_dim,
        out_dim=int(out_dim),
        hidden=tuple(cfg.get("hidden", (64, 32))),
        dropout=float(cfg.get("dropout", 0.2)),
        batchnorm=bool(cfg.get("batchnorm", True)),
    )


def export_torchscript(model: MLP, in_dim: int, path) -> None:
    """Script it, fall back to trace.

    TorchScript keeps the artifact free of pickle, so the predictor can load it with a
    stock torch and no import of this repository.
    """
    model = model.cpu().eval()
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, torch.zeros(2, in_dim), check_trace=False)
    scripted.save(str(path))
