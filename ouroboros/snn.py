"""A spiking neural network: computation by discrete spikes evolving in time.

A standard layer computes ``y = f(Wx)`` once. A spiking layer is a dynamical system. Each
neuron holds a **membrane potential** that

* **leaks** a little every time step (``v <- beta * v``),
* **integrates** incoming current (``v <- v + W s_in``),
* **fires** a binary spike when it crosses a threshold, and
* **resets** by subtracting the threshold after firing.

This is the leaky integrate-and-fire (LIF) neuron used on neuromorphic hardware such as
Intel's Loihi and SpiNNaker. Information lives in *when* and *how often* neurons spike, and
most neurons are silent most of the time, which is where the energy savings come from.

Spikes are all-or-nothing, so their true derivative is zero almost everywhere and
backpropagation can't pass through them. The standard fix, used here, is a **surrogate
gradient**: the forward pass emits a real spike, while the backward pass pretends the spike
was a smooth "fast sigmoid" so errors can flow.

This is a faithful small model of the idea, trained on a synthetic task. It is not a
benchmark result and not a claim about any hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Spike(torch.autograd.Function):
    """Heaviside step forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad / (ctx.slope * x.abs() + 1.0) ** 2, None


def spike(x: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return _Spike.apply(x, slope)


class LIFLayer(nn.Module):
    """A fully connected layer of leaky integrate-and-fire neurons."""

    def __init__(self, n_in: int, n_out: int, beta: float = 0.9, threshold: float = 1.0, slope: float = 10.0):
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta (the leak) must be in [0, 1)")
        self.fc = nn.Linear(n_in, n_out, bias=False)
        self.n_out, self.beta, self.threshold, self.slope = n_out, beta, threshold, slope

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``: (time, batch, n_in). Returns (spikes, membrane), each (time, batch, n_out)."""
        steps, batch, _ = x.shape
        v = x.new_zeros(batch, self.n_out)
        spikes, membrane = [], []
        for t in range(steps):
            v = self.beta * v + self.fc(x[t])  # leak, then integrate
            s = spike(v - self.threshold, self.slope)  # fire
            membrane.append(v)
            v = v - s.detach() * self.threshold  # reset by subtraction
            spikes.append(s)
        return torch.stack(spikes), torch.stack(membrane)


class SNN(nn.Module):
    """Input spikes -> hidden LIF layer -> output LIF layer. The class score is the output
    neuron's membrane potential averaged over time."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.hidden = LIFLayer(n_in, n_hidden, beta, threshold)
        self.out = LIFLayer(n_hidden, n_out, beta, threshold)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden_spikes, _ = self.hidden(x)
        out_spikes, out_membrane = self.out(hidden_spikes)
        return out_membrane.mean(dim=0), {"hidden": hidden_spikes, "out": out_spikes}


# ----------------------------------------------------------------------------- data
def make_spike_dataset(
    n: int, n_in: int = 40, steps: int = 50, n_classes: int = 4, seed: int = 0,
    signal_rate: float = 0.15, background_rate: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy spike trains. Class k raises the firing rate of channel group k during a burst
    window at a random time; every other channel fires at a low background rate. Returns
    (spikes of shape (steps, n, n_in), labels of shape (n,))."""
    if n_in % n_classes:
        raise ValueError("n_in must be divisible by n_classes")
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(n_classes, (n,), generator=g)
    starts = torch.randint(0, steps // 2 + 1, (n,), generator=g)
    group = n_in // n_classes
    prob = torch.full((n, steps, n_in), background_rate)
    for i in range(n):
        k, s = int(labels[i]), int(starts[i])
        prob[i, s : s + steps // 2, k * group : (k + 1) * group] = signal_rate
    x = (torch.rand(n, steps, n_in, generator=g) < prob).float()
    return x.permute(1, 0, 2).contiguous(), labels


# --------------------------------------------------------------------------- training
@dataclass
class SNNResult:
    train_accuracy: float
    test_accuracy: float
    hidden_rate: float  # mean spikes per neuron per time step
    out_rate: float
    losses: list[float]


@torch.no_grad()
def evaluate(model: SNN, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    model.eval()
    logits, spikes = model(x)
    acc = (logits.argmax(1) == y).float().mean().item()
    return acc, spikes["hidden"].mean().item(), spikes["out"].mean().item()


def train_snn(
    seed: int = 0, n_train: int = 800, n_test: int = 400, epochs: int = 25, hidden: int = 64,
    lr: float = 3e-3, batch_size: int = 64, n_in: int = 40, steps: int = 50, n_classes: int = 4,
    beta: float = 0.9, log=None,
) -> tuple[SNN, SNNResult, tuple[torch.Tensor, torch.Tensor]]:
    xtr, ytr = make_spike_dataset(n_train, n_in, steps, n_classes, seed=seed)
    xte, yte = make_spike_dataset(n_test, n_in, steps, n_classes, seed=seed + 10_000)
    torch.manual_seed(seed)
    model = SNN(n_in, hidden, n_classes, beta=beta)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train, generator=gen)
        total = 0.0
        for i in range(0, n_train, batch_size):
            idx = order[i : i + batch_size]
            logits, _ = model(xtr[:, idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / n_train)
        if log:
            log(f"epoch {epoch + 1:2d}  loss {losses[-1]:.4f}")
    tr_acc, _, _ = evaluate(model, xtr, ytr)
    te_acc, h_rate, o_rate = evaluate(model, xte, yte)
    return model, SNNResult(tr_acc, te_acc, h_rate, o_rate, losses), (xte, yte)


def raster(spikes: torch.Tensor, max_neurons: int = 16) -> str:
    """ASCII spike raster: one row per neuron, one column per time step, `|` = spike."""
    s = spikes[:, :max_neurons].t().bool()
    return "\n".join(f"{i:2d} " + "".join("|" if v else "." for v in row) for i, row in enumerate(s.tolist()))
