"""Shared perturb-and-recover inference for resilience experiments."""
from __future__ import annotations

import argparse
import hashlib
import math
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch


def add_resilience_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resilience_mode", choices=["single", "double", "both"], default="single",
        help="Inject once, twice sequentially, or save both from a shared trajectory.",
    )
    rgb = parser.add_mutually_exclusive_group()
    rgb.add_argument(
        "--perturb_rgb", action="store_true", dest="perturb_rgb",
        help="Add noise to input/RGB channels too; hold noisy RGB fixed during recovery.",
    )
    rgb.add_argument(
        "--no_perturb_rgb", action="store_false", dest="perturb_rgb",
        help="Perturb only non-input channels and keep original RGB fixed (default).",
    )
    parser.set_defaults(perturb_rgb=False)


def selected_modes(mode: str) -> tuple[str, ...]:
    if mode not in {"single", "double", "both"}:
        raise ValueError("resilience_mode must be single, double or both.")
    return ("single", "double") if mode == "both" else (mode,)


def result_method(mode: str, perturb_rgb: bool) -> str:
    if mode not in {"single", "double"}:
        raise ValueError("A result must identify single or double perturbation.")
    return f"resilience_{mode}_{'rgb' if perturb_rgb else 'latent'}"


def validate_parameters(noise_std: float, relax_steps: int) -> None:
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Perturbation noise std must be finite and non-negative.")
    if relax_steps < 1:
        raise ValueError("Recovery steps must be at least 1.")


def experiment_seed(seed: int, dataset: str, index: int, stage: str) -> int:
    """Stable seeds independent of selected modes and global RNG consumption."""
    key = f"{seed}:{dataset}:{index}:{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


@contextmanager
def isolated_rng(seed: int, device: torch.device) -> Iterator[None]:
    devices = [] if device.type != "cuda" else [
        device.index if device.index is not None else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


def perturb_state(
    state: torch.Tensor, input_channels: int, noise_std: float,
    perturb_rgb: bool, seed: int,
) -> torch.Tensor:
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Perturbation noise std must be finite and non-negative.")
    if not 0 <= input_channels <= state.size(-1):
        raise ValueError("input_channels is outside the state channel range.")
    generator = torch.Generator(device=state.device).manual_seed(seed)
    # Match latent noise across RGB choices by drawing the same full tensor.
    noise = torch.randn(
        state.shape, dtype=state.dtype, device=state.device, generator=generator
    ) * noise_std
    if not perturb_rgb:
        noise[..., :input_channels] = 0
    return state.clone() + noise


def recover_state(
    model, state: torch.Tensor, relax_steps: int, seed: int,
    on_state: Optional[Callable[[torch.Tensor, int], None]] = None,
) -> torch.Tensor:
    """Record the injected state at step zero, then exactly relax_steps updates."""
    if relax_steps < 1:
        raise ValueError("Recovery steps must be at least 1.")
    state = state.clone()
    # Hold this cycle's RGB fixed: original RGB or explicitly perturbed RGB.
    conditioning = state[..., :model.input_channels].clone()
    with isolated_rng(seed, state.device):
        if on_state is None:
            return model(state, steps=relax_steps, conditioning=conditioning)
        on_state(state, 0)
        for step in range(1, relax_steps + 1):
            state = model(state, steps=1, conditioning=conditioning)
            on_state(state, step)
    return state


def perturb_and_recover(
    model, baseline_state: torch.Tensor, noise_std: float, relax_steps: int,
    mode: str, perturb_rgb: bool, seed: int,
    on_state: Optional[Callable[[torch.Tensor, int, int], None]] = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Cycle two starts from cycle one's recovered state, never the baseline."""
    validate_parameters(noise_std, relax_steps)
    modes = selected_modes(mode)
    state = baseline_state.clone()
    cycles = 2 if "double" in modes else 1
    for cycle in range(1, cycles + 1):
        state = perturb_state(
            state, model.input_channels, noise_std, perturb_rgb,
            experiment_seed(seed, "resilience", cycle, "noise"),
        )
        callback = None
        if on_state is not None:
            callback = lambda current, step: on_state(current, cycle, step)
        state = recover_state(
            model, state, relax_steps,
            experiment_seed(seed, "resilience", cycle, "recovery"), callback,
        )
        endpoint = "single" if cycle == 1 else "double"
        if endpoint in modes:
            yield endpoint, state


def rollout_metadata(mode: str, perturb_rgb: bool, steps: int, relax_steps: int) -> dict:
    count = 1 if mode == "single" else 2
    return {
        "resilience_mode": mode,
        "perturb_rgb": perturb_rgb,
        "input_conditioning": "noisy_rgb_per_cycle" if perturb_rgb else "original_rgb",
        "steps": steps,
        "resilience_relax_steps": relax_steps,
        "perturbation_count": count,
        "injection_steps": [steps + cycle * relax_steps for cycle in range(count)],
        "evaluation_step": steps + count * relax_steps,
    }
