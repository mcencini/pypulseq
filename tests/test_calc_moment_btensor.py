import numpy as np
import pytest
from pypulseq import Sequence, make_block_pulse, make_delay, make_trapezoid

γ = 2 * np.pi * 42.576e6  # rad/s/T


def b_fact_calc(g, delta, DELTA):
    # Approximation for rectangular gradients
    kappa_minus_lambda = 1 / 3 - 1 / 2
    return (2 * np.pi * g) ** 2 * delta**2 * (DELTA + 2 * kappa_minus_lambda * delta)


@pytest.fixture
def seq():
    seq = Sequence()
    rf = make_block_pulse(90.0, duration=4e-3, use='excitation')
    rf180 = make_block_pulse(180.0, duration=4e-3, use='refocusing')
    grad_x = make_trapezoid(channel='x', system=seq.system, amplitude=10, rise_time=100e-6, flat_time=200e-6)
    grad_y = make_trapezoid(channel='y', system=seq.system, amplitude=15, rise_time=100e-6, flat_time=200e-6)
    grad_z = make_trapezoid(channel='z', system=seq.system, amplitude=20, rise_time=100e-6, flat_time=200e-6)
    delay = make_delay(1e-3)

    for _n in range(3):
        seq.add_block(rf)
        seq.add_block(grad_x, grad_y, grad_z)
        seq.add_block(delay)
        seq.add_block(rf180)
        seq.add_block(grad_x, grad_y, grad_z)

    return seq


@pytest.fixture
def diffusion_pgse_seq():
    seq = Sequence()
    sys = seq.system

    gmax = 0.04  # 40 mT/m
    max_slew = 150  # T/m/s
    grad_raster = 10e-6
    rf180_dur = 3.5e-3

    delayTE1 = 40e-3
    delayTE2 = 20e-3

    ramp_time = np.ceil(gmax / max_slew / grad_raster) * grad_raster
    delta = delayTE2 - ramp_time
    DELTA = delayTE1 + rf180_dur

    b_target = 800e6  # s/m² (≈800 s/mm²)
    g = np.sqrt(b_target / b_fact_calc(1, delta, DELTA))

    rise_time = np.ceil(g / max_slew / grad_raster) * grad_raster
    rf = make_block_pulse(90.0, duration=4e-3, use='excitation')
    rf180 = make_block_pulse(180.0, duration=rf180_dur, use='refocusing')
    gDiff = make_trapezoid(channel='z', system=sys, amplitude=g, rise_time=rise_time, flat_time=delta - rise_time)
    delay = make_delay(DELTA - delta)
    gDiffNeg = make_trapezoid(channel='z', system=sys, amplitude=-g, rise_time=rise_time, flat_time=delta - rise_time)

    seq.add_block(rf)
    seq.add_block(gDiff)
    seq.add_block(delay)
    seq.add_block(rf180)
    seq.add_block(gDiffNeg)

    return seq, g, delta, DELTA


# ========== Tests ==========


def test_moments_shapes(seq):
    B, m1, m2, m3 = seq.calc_moments_b_tensor(calcB=True, calcM1=True, calcM2=True, calcM3=True, n_dummy=0)
    assert B.shape == (3, 3, 3)
    assert m1.shape == (3, 3)
    assert m2.shape == (3, 3)
    assert m3.shape == (3, 3)


def test_moments_with_dummy(seq):
    B, m1, _, _ = seq.calc_moments_b_tensor(calcB=True, calcM1=True, n_dummy=1)
    assert B.shape[0] == 2
    assert m1.shape[0] == 2


def test_b_tensor_symmetry(seq):
    B, _, _, _ = seq.calc_moments_b_tensor(calcB=True, n_dummy=0)
    for i in range(B.shape[0]):
        assert np.allclose(B[i], B[i].T, atol=1e-6), f'B tensor at index {i} is not symmetric.'


def test_diffusion_b_tensor_properties(diffusion_pgse_seq):
    seq, g, delta, DELTA = diffusion_pgse_seq
    B, _, eigvals, eigvecs = seq.calc_moments_b_tensor(calcB=True)

    assert B.shape == (1, 3, 3)
    B0 = B[0]

    # Diagonal matrix
    assert np.allclose(B0, np.diag(np.diagonal(B0)), atol=1e-10), 'B tensor should be diagonal.'

    # Principal axis should be Z
    max_idx = np.argmax(eigvals[0])
    principal_axis = eigvecs[0][:, max_idx]
    assert np.abs(np.dot(principal_axis, [0, 0, 1])) > 0.9999, 'Principal axis not aligned with Z.'


def test_diffusion_b_tensor_value(diffusion_pgse_seq):
    seq, g, delta, DELTA = diffusion_pgse_seq
    B, _, _, _ = seq.calc_moments_b_tensor(calcB=True)
    B0 = B[0]
    b_meas = B0[2, 2]
    b_theory = b_fact_calc(g, delta, DELTA)

    rel_err = np.abs(b_meas - b_theory) / b_theory
    assert rel_err < 0.01, f'B value mismatch: got {b_meas:.2e}, expected {b_theory:.2e}, rel. error = {rel_err:.2%}'
