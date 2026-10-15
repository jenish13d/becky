"""Make Becky's brain smaller, and measure what it cost.

Weight-only int8 quantization: each row of every big weight matrix (the attention and MLP
layers, which hold nearly all the parameters) is stored as 8-bit integers plus one float
scale per row, instead of 32-bit floats. Small, sensitive pieces (embeddings, biases,
LayerNorms) stay as they were. The file is about 3x smaller, which is what matters for
copying the brain to a small device and for holding it in limited storage.

At load time the integers are expanded back to floats, so the model runs through exactly the
same code as before. That means **less storage, not less RAM or faster math**. For speed, see
the key/value cache in ``model.py``. Rounding to 8 bits loses a little precision, so
:func:`compress_report` measures accuracy before and after and says whether it is safe.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import GPT, GPTConfig

FORMAT = "int8-v1"
_BIG = (".attn.qkv.weight", ".attn.proj.weight", ".mlp.fc.weight", ".mlp.proj.weight")


def _is_big(name: str) -> bool:
    return name.startswith("blocks.") and name.endswith(_BIG)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: ``w ≈ q * scale``. Returns (int8 tensor, float32 scale per row)."""
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.unsqueeze(1)


def compress_state(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    packed: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    kept: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        if _is_big(name):
            packed[name], scales[name] = quantize_rows(t.detach().float())
        else:
            kept[name] = t.detach().clone()
    return {"int8": packed, "scales": scales, "float": kept}


def expand_state(blob: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    state = dict(blob["float"])
    for name, q in blob["int8"].items():
        state[name] = dequantize_rows(q, blob["scales"][name])
    return state


def save_compressed(model: GPT, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "model_config": model.cfg.to_dict(), **compress_state(model.state_dict())}, path)
    return path


def is_compressed(ckpt: dict) -> bool:
    return ckpt.get("format") == FORMAT


def model_from_compressed(ckpt: dict) -> GPT:
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(expand_state(ckpt))
    return model


def compress_report(model: GPT, out: Path, source: Path | None = None, eval_n: int = 100, seed: int = 31337) -> dict:
    """Compress to ``out`` and compare the original with the round-tripped copy."""
    from .brain import OwnLLM, evaluate_brain

    save_compressed(model, out)
    restored = model_from_compressed(torch.load(out, map_location="cpu", weights_only=True))
    before = evaluate_brain(OwnLLM(model), n=eval_n, seed=seed)["overall"]
    after = evaluate_brain(OwnLLM(restored), n=eval_n, seed=seed)["overall"]
    worst = max((model.state_dict()[k] - restored.state_dict()[k]).abs().max().item()
                for k in model.state_dict() if _is_big(k))
    size_before = Path(source).stat().st_size if source else None
    size_after = Path(out).stat().st_size
    return {
        "size_before_bytes": size_before, "size_after_bytes": size_after,
        "ratio": (size_before / size_after) if size_before else None,
        "accuracy_before": before, "accuracy_after": after,
        "accuracy_drop": before - after, "max_weight_error": worst,
        "safe": (before - after) <= 0.02,
    }
