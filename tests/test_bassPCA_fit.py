import numpy as np

import pyBASS as pb

from .util import rootmeansqerror


def test_bassPCA_fit():
    # Friedman function with functional response
    def f2(x):
        out = (
            10.0 * np.sin(np.pi * np.linspace(0, 1, 50) * x[1])
            + 20.0 * (x[2] - 0.5) ** 2
            + 10 * x[3]
            + 5.0 * x[4]
        )
        return out

    np.random.seed(0)
    tt = np.linspace(0, 1, 50)  # functional variable grid
    n = 500  # sample size
    p = 9  # number of predictors other (only 4 are used)
    x = np.random.rand(n, p)  # training inputs
    xx = np.random.rand(1000, p)
    e = np.random.normal(size=[n, len(tt)]) * 0.1  # noise
    y = np.apply_along_axis(f2, 1, x) + e  # training response
    ftest = np.apply_along_axis(f2, 1, xx)

    # fit BASS model with RJMCMC
    mod = pb.bassPCA(x, y, maxInt=5)

    # predict at new inputs (xnew)
    pred = mod.predict(xx, nugget=True)

    # Root mean squred error
    rmse = rootmeansqerror(pred.mean(0), ftest)
    print("RMSE: ", rmse)

    # Test that RMSE is less than 0.05 for this model, which should be the case
    # from previous tests.
    assert rmse < 0.05
