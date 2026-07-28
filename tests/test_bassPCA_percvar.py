import numpy as np

import pyBASS as pb


def test_bassPCA_percVar_100():
    # percVar is documented as a percent between 0 and 100, so 100 should keep
    # every principal component rather than raise.
    def f(x):
        return np.array([x[0] + x[1], 2.0 * x[0], x[1] - x[0]])

    np.random.seed(0)
    n = 30
    p = 3
    x = np.random.rand(n, p)
    y = np.apply_along_axis(f, 1, x) + np.random.normal(size=[n, 3]) * 0.01

    mod = pb.bassPCA(x, y, percVar=100, nmcmc=300, nburn=200, verbose=False)

    assert mod.nbasis == 3
    assert mod.predict(x).shape == (100, n, 3)


def test_bassPCA_percVar_below_100_unchanged():
    # A threshold under 100 must select the same number of components as before.
    def f(x):
        return np.array([x[0] + x[1], 2.0 * x[0], x[1] - x[0]])

    np.random.seed(0)
    n = 30
    p = 3
    x = np.random.rand(n, p)
    y = np.apply_along_axis(f, 1, x) + np.random.normal(size=[n, 3]) * 0.01

    mod = pb.bassPCA(x, y, percVar=99.9, nmcmc=300, nburn=200, verbose=False)

    assert mod.nbasis == 2
