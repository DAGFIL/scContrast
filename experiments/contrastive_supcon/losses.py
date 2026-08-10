"""Supervised Contrastive Loss (Khosla et al. 2020, https://arxiv.org/abs/2004.11362)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [n_views * batch, dim], L2-normalized, views for the same sample
                  must NOT be adjacent — caller passes labels aligned 1:1 with rows.
        labels:   [n_views * batch] integer class id, repeated per view.
        """
        device = features.device
        n = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        self_mask = torch.eye(n, device=device)
        mask = mask * (1 - self_mask)

        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_counts = mask.sum(1)
        valid = pos_counts > 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / pos_counts[valid]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()
