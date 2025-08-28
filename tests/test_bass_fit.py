import numpy as np

import pyBASS as pb

from .util import rootmeansqerror


def test_bass_fit():
    # Friedman function (Friedman, 1991, Multivariate Adaptive Regression Splines)
    def f(x):
        return (
            10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20.0 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3]
            + 5.0 * x[:, 4]
        )

    # Set random seed for reproducibility.
    np.random.seed(0)

    # Generate data.
    n = 500  # sample size
    p = 10  # number of predictors (only 5 are used)
    x = np.random.rand(n, p)  # predictors (training set)
    y = np.random.normal(f(x), 0.1)  # response (training set) with noise.

    # fit BASS model with RJMCMC
    mod = pb.bass(x, y, nmcmc=10000, nburn=9000)

    # predict at new inputs (xnew)
    xnew = np.random.rand(1000, p)
    pred = mod.predict(xnew, nugget=True)

    # True values at new inputs.
    ynew = f(xnew)

    # Root mean squred error
    rmse = rootmeansqerror(pred.mean(0), ynew)
    print("RMSE: ", rmse)

    # Test that RMSE is less than 0.1 for this model, which should be the case
    # from previous tests.
    assert rmse < 0.1
