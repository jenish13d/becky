"""Offline tests on a tiny synthetic corpus; the whole suite runs in seconds on CPU."""

from __future__ import annotations

import json
import random

import pytest
import torch

from ouroboros.data import bits_per_byte, decode, encode, get_batch, split
from ouroboros.evolve import (
    SPACE, Designer, EvolveConfig, Genome, count_params, evolve, load_model, within_budget,
)
from ouroboros.model import GPT, GPTConfig, inherit_weights
from ouroboros.train import TrainRecipe, distill_loss, train

TINY = GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=16)


def word_corpus(words: list[str], seed: int):
    rng = random.Random(seed)
    text = "\n".join(" ".join(rng.choice(words) for _ in range(8)) + "." for _ in range(1500))
    return split(text.encode())


@pytest.fixture(scope="module")
def splits():
    return word_corpus(["the", "king", "queen", "sword", "crown", "speaks", "falls", "rises",
                        "night", "day"], seed=0)


def test_tokenizer_roundtrip_including_unicode():
    s = "Hello, wörld — ✓"
    assert decode(encode(s)) == s
    assert all(0 <= i < 256 for i in encode(s))


def test_forward_shapes_and_loss():
    model = GPT(TINY)
    x = torch.randint(0, 256, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 256)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(256.0)).item(), rel=0.1)


def test_attention_is_causal():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    x = torch.randint(0, 256, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % 256  # change only the future
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-6)
    assert not torch.allclose(a[0, 8:], b[0, 8:])


def test_rejects_context_longer_than_block():
    with pytest.raises(ValueError):
        GPT(TINY)(torch.zeros(1, 17, dtype=torch.long))


def test_param_count_formula_matches_model():
    for g in [Genome(), Genome(n_layer=5, n_head=3, head_dim=24, block_size=96)]:
        assert count_params(g) == GPT(g.model_config()).num_params()


def test_split_is_contiguous_and_disjoint(splits):
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total > 50_000
    assert len(splits.val) == len(splits.test)


def test_bits_per_byte_scores_every_byte_once_regardless_of_block_size(splits):
    # A model that always predicts uniform scores exactly 8 bits per byte.
    for bs in (16, 37):
        model = GPT(GPTConfig(block_size=bs, n_layer=1, n_head=1, n_embd=8))
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        assert bits_per_byte(model, splits.val[:1001]) == pytest.approx(8.0, abs=1e-5)


def test_training_reduces_loss(splits):
    torch.manual_seed(0)
    model = GPT(TINY)
    before = bits_per_byte(model, splits.val, 5000)
    train(model, splits.train, TrainRecipe(steps=150, batch_size=16, lr=5e-3, warmup=10))
    after = bits_per_byte(model, splits.val, 5000)
    assert after < before - 2.0


def test_distillation_pulls_student_toward_teacher(splits):
    # The teacher learned *different* text, so only distillation can make a student
    # trained on `splits` behave like it.
    torch.manual_seed(0)
    teacher = GPT(TINY)
    other = word_corpus(["zebra", "quokka", "yak", "vixen", "jaguar", "ox"], seed=1)
    train(teacher, other.train, TrainRecipe(steps=150, batch_size=16, lr=5e-3, warmup=10))
    x, _ = get_batch(splits.val, 8, 16, torch.Generator().manual_seed(1))

    def gap(student):
        with torch.no_grad():
            return distill_loss(student(x)[0], teacher(x)[0], temp=1.0).item()

    recipe = TrainRecipe(steps=60, batch_size=16, lr=5e-3, warmup=5)
    plain, distilled = GPT(TINY), GPT(TINY)
    distilled.load_state_dict(plain.state_dict())
    train(plain, splits.train, recipe, seed=3)
    train(distilled, splits.train, TrainRecipe(**{**recipe.to_dict(), "distill_alpha": 0.9}),
          teacher=teacher, seed=3)
    assert gap(distilled) < 0.5 * gap(plain)


def test_inherit_weights_copies_matching_layers_only():
    parent = GPT(GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=16))
    child = GPT(GPTConfig(block_size=32, n_layer=3, n_head=2, n_embd=16))
    copied = inherit_weights(child, parent)
    assert copied > 0
    assert torch.equal(child.blocks[1].mlp.fc.weight, parent.blocks[1].mlp.fc.weight)
    assert torch.equal(child.pos_emb.weight[:16], parent.pos_emb.weight)
    assert not torch.equal(child.pos_emb.weight[16:], torch.zeros(16, 16))  # new rows keep their init
    wider = GPT(GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32))
    assert inherit_weights(wider, parent) == 0  # a width change leaves nothing shape-compatible


def test_knob_roundtrip_and_bounds():
    for name, k in SPACE.items():
        assert k.from_unit(-5) == k.lo and k.from_unit(5) == k.hi
        mid = k.from_unit(0.5)
        assert k.lo <= mid <= k.hi
        if k.step is not None:
            assert isinstance(mid, int)


def test_designer_proposals_stay_in_space_and_differ():
    rng = random.Random(0)
    d = Designer()
    parent = Genome()
    for _ in range(200):
        d = d.spawn(rng)
        g = d.propose(parent, rng)
        assert g is not None and g != parent
        for name, k in SPACE.items():
            assert k.lo <= getattr(g, name) <= k.hi
        assert all(0.05 <= p <= 0.9 for p in d.p.values())
        assert all(0.02 <= s <= 0.5 for s in d.sigma.values())


def test_designer_learns_drift_from_promotions():
    d = Designer()
    parent = Genome(n_layer=2)
    d.learn(parent, Genome(n_layer=4))
    assert d.drift["n_layer"] > 0
    assert all(v == 0 for k, v in d.drift.items() if k != "n_layer")


def test_budget_rejects_oversized_designs():
    cfg = EvolveConfig(max_params=100_000)
    assert within_budget(Genome(n_layer=1, n_head=1, head_dim=32), cfg)
    assert not within_budget(Genome(n_layer=8, n_head=8, head_dim=64), cfg)


def test_evolve_end_to_end(splits, tmp_path):
    cfg = EvolveConfig(generations=2, children=2, steps=40, seed_steps=60, max_params=200_000,
                       max_flops=1e12, margin=0.0, patience=5, eval_bytes=3000, sample_tokens=20)
    seed = Genome(n_layer=1, n_head=2, head_dim=8, block_size=32, batch_size=16)
    lineage = evolve(splits, cfg, tmp_path, seed_genome=seed, log=lambda *_: None)

    on_disk = json.loads((tmp_path / "lineage.json").read_text())
    assert on_disk["result"] == lineage["result"]
    # one control + two designed children per generation
    assert len(lineage["candidates"]) == 6
    assert {c["kind"] for c in lineage["candidates"]} == {"control", "designed"}
    for c in lineage["candidates"]:
        assert c["params"] <= cfg.max_params
    # champions only ever get better on validation
    bpbs = [c["val_bpb"] for c in lineage["champions"]]
    assert bpbs == sorted(bpbs, reverse=True)
    assert lineage["result"]["val_bpb"] <= lineage["result"]["seed_val_bpb"]
    assert 0 < lineage["result"]["test_bpb"] < 8.5

    model = load_model(tmp_path / "champion.pt")
    assert model.cfg == Genome(**lineage["champions"][-1]["genome"]).model_config()
    assert (tmp_path / "gen00.pt").exists()


def test_evolve_rejects_seed_over_budget(splits, tmp_path):
    with pytest.raises(ValueError, match="budget"):
        evolve(splits, EvolveConfig(max_params=1000), tmp_path, log=lambda *_: None)


def test_evolve_respects_patience(splits, tmp_path):
    # An impossible margin means nothing is ever promoted, so the run stops early.
    cfg = EvolveConfig(generations=10, children=1, steps=5, seed_steps=5, margin=100.0,
                       patience=2, eval_bytes=1000, sample_tokens=5, max_params=200_000)
    seed = Genome(n_layer=1, n_head=1, head_dim=8, block_size=32, batch_size=8)
    lineage = evolve(splits, cfg, tmp_path, seed_genome=seed, log=lambda *_: None)
    assert lineage["result"]["stop_reason"] == "patience"
    assert lineage["result"]["champion_gen"] == 0
    assert max(c["gen"] for c in lineage["candidates"]) == 2


def test_inherit_into_identical_shape_copies_every_parameter_once():
    parent, child = GPT(TINY), GPT(TINY)
    assert inherit_weights(child, parent) == child.num_params()
    x = torch.randint(0, 256, (1, 8))
    assert torch.equal(child.eval()(x)[0], parent.eval()(x)[0])


def test_genome_roundtrip_through_unit_space_is_exact():
    for g in [Genome(), Genome(lr=7.37e-4, dropout=0.04, distill_temp=2.5, n_layer=3)]:
        assert Genome.from_unit(g.unit()) == g
