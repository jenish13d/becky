"""Tests for the spiking network: first the neuron physics, then learning."""

from __future__ import annotations

import pytest
import torch

from ouroboros.snn import LIFLayer, SNN, make_spike_dataset, raster, spike, train_snn


def one_neuron(beta=0.9, threshold=1.0):
    layer = LIFLayer(1, 1, beta=beta, threshold=threshold)
    with torch.no_grad():
        layer.fc.weight.fill_(1.0)  # identity wiring: input current goes straight in
    return layer


def drive(layer, currents):
    x = torch.tensor(currents, dtype=torch.float32).view(-1, 1, 1)
    s, v = layer(x)
    return s.view(-1).tolist(), v.view(-1).tolist()


def test_spike_is_binary_in_the_forward_pass():
    out = spike(torch.linspace(-2, 2, 101))
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert out[0] == 0 and out[-1] == 1


def test_surrogate_gradient_flows_and_peaks_at_threshold():
    x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], requires_grad=True)
    spike(x).sum().backward()
    g = x.grad
    assert torch.isfinite(g).all() and (g > 0).all()
    assert g[2] == g.max()  # steepest exactly at the threshold
    assert g[0] == pytest.approx(g[4])  # symmetric


def test_membrane_leaks_by_beta_each_step():
    layer = one_neuron(beta=0.5)
    _, v = drive(layer, [0.8, 0, 0, 0])
    assert v == pytest.approx([0.8, 0.4, 0.2, 0.1])


def test_subthreshold_input_never_fires_and_settles_at_steady_state():
    layer = one_neuron(beta=0.9)
    s, v = drive(layer, [0.05] * 300)  # steady state = 0.05 / (1 - 0.9) = 0.5 < threshold
    assert sum(s) == 0
    assert v[-1] == pytest.approx(0.5, abs=1e-3)


def test_integration_then_fire_then_reset_by_subtraction():
    layer = one_neuron(beta=0.9)
    s, v = drive(layer, [0.3] * 6)
    # v: 0.3, 0.57, 0.813, 1.0317 -> spike on the 4th step
    assert s[:4] == [0.0, 0.0, 0.0, 1.0]
    assert v[3] == pytest.approx(1.0317, abs=1e-4)
    # after the spike the threshold is subtracted: 0.9*0.0317 + 0.3 = 0.3285
    assert v[4] == pytest.approx(0.9 * (v[3] - 1.0) + 0.3, abs=1e-5)


def test_stronger_input_fires_earlier_and_more_often():
    weak, _ = drive(one_neuron(), [0.25] * 30)
    strong, _ = drive(one_neuron(), [0.6] * 30)
    assert strong.index(1.0) < weak.index(1.0)
    assert sum(strong) > sum(weak) > 0


def test_invalid_leak_is_rejected():
    for beta in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            LIFLayer(2, 2, beta=beta)


def test_network_shapes_and_binary_spikes():
    net = SNN(8, 5, 3)
    logits, spikes = net(torch.rand(20, 4, 8).round())
    assert logits.shape == (4, 3)
    assert spikes["hidden"].shape == (20, 4, 5) and spikes["out"].shape == (20, 4, 3)
    assert set(spikes["hidden"].unique().tolist()) <= {0.0, 1.0}


def test_gradients_reach_every_layer():
    net = SNN(40, 16, 4)
    x, y = make_spike_dataset(32, seed=1)
    loss = torch.nn.functional.cross_entropy(net(x)[0], y)
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_dataset_is_reproducible_balanced_and_binary():
    x1, y1 = make_spike_dataset(200, seed=3)
    x2, y2 = make_spike_dataset(200, seed=3)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (50, 200, 40) and set(x1.unique().tolist()) <= {0.0, 1.0}
    counts = torch.bincount(y1, minlength=4)
    assert counts.min() > 25
    with pytest.raises(ValueError):
        make_spike_dataset(10, n_in=10, n_classes=4)


def test_it_actually_learns_far_above_chance_on_unseen_data():
    _, res, _ = train_snn(seed=0, epochs=25)
    assert res.test_accuracy > 0.85  # chance is 0.25
    assert res.losses[-1] < 0.5 * res.losses[0]
    assert 0.0 < res.hidden_rate < 0.6 and 0.0 < res.out_rate < 0.6


def test_training_is_deterministic():
    _, a, _ = train_snn(seed=5, epochs=3, n_train=128, n_test=64, hidden=16)
    _, b, _ = train_snn(seed=5, epochs=3, n_train=128, n_test=64, hidden=16)
    assert a.losses == b.losses and a.test_accuracy == b.test_accuracy


def test_raster_renders_spikes():
    s = torch.zeros(6, 3)
    s[1, 0] = s[4, 2] = 1
    assert raster(s).splitlines() == [" 0 .|....", " 1 ......", " 2 ....|."]
