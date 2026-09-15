"""Dreamer `perf.channels_last`: same encoder/decoder numerics, NHWC layout.

NHWC and NCHW run different conv kernels, so equality is to float tolerance,
not bitwise. Runs on CPU without gym/env.
"""

from types import SimpleNamespace as NS

import pytest
import torch

from src.algorithms.dreamer import perf_flags

ENC = NS(act="SiLU", norm=True, kernel_size=5, minres=4, depth=4, mults=[2, 3, 4, 4])
DEC = NS(act="SiLU", norm=True, kernel_size=5, minres=4, depth=4, mults=[2, 3, 4, 4],
         units=32, bspace=8, outscale=1.0)
B, T, DETER, FSTOCH = 2, 3, 64, 16


@pytest.fixture(autouse=True)
def _reset_flags():
    yield
    perf_flags.configure()


def _build(channels_last: bool):
    from src.algorithms.dreamer import networks

    perf_flags.configure({"channels_last": channels_last})
    torch.manual_seed(0)
    enc = networks.ConvEncoder(ENC, (64, 64, 3))
    dec = networks.ConvDecoder(DEC, DETER, FSTOCH, (3, 64, 64))
    if channels_last:
        # What DreamerV3.__init__ does for the whole model.
        enc.to(memory_format=torch.channels_last)
        dec.to(memory_format=torch.channels_last)
    return enc, dec


def _run(enc, dec):
    g = torch.Generator().manual_seed(1)
    obs = torch.rand(B, T, 3, 64, 64, generator=g)
    stoch = torch.randn(B, T, FSTOCH, generator=g)
    deter = torch.randn(B, T, DETER, generator=g)
    emb, rec = enc(obs), dec(stoch, deter)
    (emb.pow(2).mean() + (rec - obs).pow(2).mean()).backward()
    grads = [p.grad for p in (*enc.parameters(), *dec.parameters())]
    return emb, rec, grads


def test_channels_last_matches_default():
    emb0, rec0, g0 = _run(*_build(False))
    emb1, rec1, g1 = _run(*_build(True))
    torch.testing.assert_close(emb1, emb0)
    torch.testing.assert_close(rec1, rec0)
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b)


def test_channels_last_layout():
    enc, dec = _build(True)
    for conv in (m for m in (*enc.modules(), *dec.modules()) if isinstance(m, torch.nn.Conv2d)):
        assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    seen = []
    enc.layers[0].register_forward_pre_hook(lambda m, args: seen.append(args[0]))
    enc(torch.rand(B, T, 3, 64, 64))
    assert seen[0].is_contiguous(memory_format=torch.channels_last)
