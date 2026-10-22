import numpy as np

from cinemamba.phase import estimate_phase_from_kspace


def test_phase_shape_range_and_finiteness():
    rng = np.random.default_rng(0)
    kspace = rng.normal(size=(10, 2, 16, 16)) + 1j * rng.normal(size=(10, 2, 16, 16))
    phase = estimate_phase_from_kspace(kspace, kc=8)
    assert phase.shape == (10,)
    assert phase.dtype == np.float32
    assert np.all(np.isfinite(phase))
    assert np.all((phase >= 0) & (phase < 1))


def test_short_sequence_uses_uniform_fallback():
    phase = estimate_phase_from_kspace(np.zeros((3, 1, 8, 8), dtype=np.complex64))
    np.testing.assert_allclose(phase, [0, 1 / 3, 2 / 3])
