"""Training loop: AdamW, warmup + cosine schedule, optional distillation from a teacher.

When a teacher is given (the parent generation), the loss mixes the usual
next-byte cross-entropy with a KL term pulling the student's distribution toward
the teacher's softened one::

    loss = (1 - alpha) * CE(student, data) + alpha * T^2 * KL(teacher_T || student_T)

That is the mechanism by which a model *trains its successor*: the successor
learns from the data and from what its parent believes about the data.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .data import get_batch
from .model import GPT


@dataclass(frozen=True)
class TrainRecipe:
    steps: int = 500
    batch_size: int = 32
    lr: float = 3e-3
    warmup: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    distill_alpha: float = 0.0
    distill_temp: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


def lr_at(step: int, recipe: TrainRecipe) -> float:
    if step < recipe.warmup:
        return recipe.lr * (step + 1) / recipe.warmup
    progress = (step - recipe.warmup) / max(1, recipe.steps - recipe.warmup)
    return recipe.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def make_optimizer(model: GPT, recipe: TrainRecipe) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": recipe.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=recipe.lr,
        betas=(0.9, 0.95),
    )


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temp: float) -> torch.Tensor:
    V = student_logits.size(-1)
    s = F.log_softmax(student_logits.reshape(-1, V) / temp, dim=-1)
    t = F.log_softmax(teacher_logits.reshape(-1, V) / temp, dim=-1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * temp**2


def train(
    model: GPT,
    data: torch.Tensor,
    recipe: TrainRecipe,
    teacher: GPT | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
    log_every: int = 100,
) -> list[dict]:
    """Train ``model`` in place on ``data``; returns a loss history."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, recipe)
    use_teacher = teacher is not None and recipe.distill_alpha > 0
    if use_teacher:
        teacher.eval()
    model.train()
    history: list[dict] = []
    t0 = time.time()
    for step in range(recipe.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, recipe)
        x, y = get_batch(data, recipe.batch_size, model.cfg.block_size, gen)
        x, y = x.to(device), y.to(device)
        logits, ce = model(x, y)
        loss = ce
        if use_teacher:
            # The teacher may have a shorter context: it scores the trailing positions.
            tl = min(x.size(1), teacher.cfg.block_size)
            with torch.no_grad():
                t_logits, _ = teacher(x[:, -tl:])
            kd = distill_loss(logits[:, -tl:], t_logits, recipe.distill_temp)
            loss = (1 - recipe.distill_alpha) * ce + recipe.distill_alpha * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
        opt.step()
        if step % log_every == 0 or step == recipe.steps - 1:
            rec = {"step": step, "ce": round(ce.item(), 4), "loss": round(loss.item(), 4),
                   "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            if log:
                log(f"  step {step:5d}  ce {rec['ce']:.4f}  loss {rec['loss']:.4f}")
    return history
