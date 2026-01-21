from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneNCA(nn.Module):
    """Neural Cellular Automata backbone aligned with the reference inference implementation."""

    def __init__(
        self,
        channel_n: int,
        fire_rate: float,
        device: torch.device,
        hidden_size: int = 128,
        input_channels: int = 3,
        steps_default: int = 32,
        init_method: str = "standard",
    ) -> None:
        super().__init__()
        if channel_n <= 0:
            raise ValueError("channel_n must be positive.")
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.input_channels = max(0, input_channels)
        self.steps_default = max(1, steps_default)
        self.device = device

        self.p0 = nn.Conv2d(
            channel_n,
            channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channel_n,
            padding_mode="reflect",
        )
        self.p1 = nn.Conv2d(
            channel_n,
            channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channel_n,
            padding_mode="reflect",
        )
        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        if init_method.lower() == "xavier":
            nn.init.xavier_uniform_(self.fc0.weight)
            nn.init.xavier_uniform_(self.fc1.weight)

        self.to(self.device)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise perception that mirrors the reference inference code."""
        z1 = self.p0(x)
        z2 = self.p1(x)
        return torch.cat((x, z1, z2), dim=1)

    def update(self, state: torch.Tensor, fire_rate: Optional[float] = None) -> torch.Tensor:
        """Apply one stochastic NCA step (state is BHWC)."""
        inputs = state.permute(0, 3, 1, 2).contiguous()
        dx = self.perceive(inputs).permute(0, 2, 3, 1)
        dx = self.fc1(F.relu(self.fc0(dx)))
        rate = self.fire_rate if fire_rate is None else fire_rate
        if rate < 1.0:
            mask = (torch.rand_like(dx[..., :1]) <= rate).float()
            dx = dx * mask
        return state + dx

    def forward(
        self,
        state: torch.Tensor,
        steps: Optional[int] = None,
        fire_rate: Optional[float] = None,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run multiple NCA steps keeping the RGB channels clamped to the original input."""
        if state.size(-1) < self.channel_n:
            pad = self.channel_n - state.size(-1)
            state = torch.cat(
                (state, torch.zeros(state.size(0), state.size(1), state.size(2), pad, device=state.device, dtype=state.dtype)),
                dim=-1,
            )
        else:
            state = state[..., : self.channel_n]

        cond = conditioning
        if cond is None and self.input_channels > 0:
            cond = state[..., : self.input_channels].clone()

        total_steps = steps if steps is not None else self.steps_default
        total_steps = max(1, total_steps)
        for _ in range(total_steps):
            if cond is not None:
                state[..., : self.input_channels] = cond
            state = self.update(state, fire_rate=fire_rate)
        if cond is not None:
            state[..., : self.input_channels] = cond
        return state
