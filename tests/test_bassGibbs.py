import numpy as np
import pytest

import pyBASS as pb


def _data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    f = np.sin(2 * np.pi * X[:, 0]) + 2 * (X[:, 1] - 0.5) ** 2
    return X, f, f + rng.normal(0, 0.1, n)


def test_default_bass_unchanged_by_split_update():
    # update() is rj_step() then gibbs_step(); bass() must still fit well
    X, f, y = _data()
    np.random.seed(0)
    m = pb.bass(X, y, nmcmc=3000, nburn=2000, verbose=False)
    assert np.sqrt(np.mean((m.predict(X).mean(0) - f) ** 2)) < 0.1


def test_gibbs_block_fits_and_predicts():
    X, f, y = _data()
    np.random.seed(1)
    g = pb.bassGibbs(X, y, 100)
    g.step(2000)
    for _ in range(100):
        g.set_response(y)
        g.step(1)
        g.save()
    P = g.model().predict(X)
    assert P.shape == (100, len(y))
    assert np.allclose(P[-1], g.fitted())
    assert np.sqrt(np.mean((P.mean(0) - f) ** 2)) < 0.1
    assert 0 < g.accept_rate() < 1
    with pytest.raises(RuntimeError):
        g.save()


def test_set_response_keeps_basis():
    X, f, y = _data()
    np.random.seed(2)
    g = pb.bassGibbs(X, y)
    g.step(200)
    nb = g.nbasis()
    g.set_response(-y)
    st = g.bm.state
    assert g.nbasis() == nb
    B = st.basis
    bh = np.linalg.solve(B.T @ B, B.T @ -y)
    assert np.allclose(st.bhat, bh)
    assert np.isclose(st.qf, bh @ (B.T @ -y))
    with pytest.raises(ValueError):
        g.set_response(y[:-1])


def test_centred_without_intercept_averages_zero():
    X, f, y = _data()
    np.random.seed(3)
    g = pb.bassGibbs(X, y - y.mean(), 20, intercept=False, center_basis=True)
    g.save()                                 # an empty basis is a valid draw
    g.step(500)
    for _ in range(19):
        g.step(1)
        g.save()
    P = g.model().predict(X)
    assert np.allclose(P[0], 0)
    assert np.max(np.abs(P.mean(1))) < 1e-10
    assert np.allclose(P[-1], g.fitted())


def test_fixed_s2():
    X, f, y = _data()
    np.random.seed(4)
    g = pb.bassGibbs(X, y, 3, s2_fixed=0.05)
    g.step(50)
    g.save()
    assert g.s2() == 0.05
    assert g.model().samples.s2[0] == 0.05
