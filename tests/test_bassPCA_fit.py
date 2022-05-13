import pyBASS as pb
import numpy as np

def test_bassPCA_fit():
    # Friedman function with functional response
    def f2(x):
        out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
        return out

    np.random.seed(0)
    tt = np.linspace(0, 1, 50) # functional variable grid
    n = 500 # sample size
    p = 9 # number of predictors other (only 4 are used)
    x = np.random.rand(n, p) # training inputs
    xx = np.random.rand(1000, p)
    e = np.random.normal(size=[n, len(tt)]) * .1 # noise
    y = np.apply_along_axis(f2, 1, x) + e # training response
    ftest = np.apply_along_axis(f2, 1, xx)

    # fit BASS model with RJMCMC
    mod = pb.bassPCA(x, y)

    # predict at new inputs (xnew)
    pred = mod.predict(xx, nugget=True)

    # Root mean squred error
    rmse = np.sqrt(np.mean((pred.mean(0) - ftest) ** 2))

    # Test that RMSE is less than 0.05 for this model, which should be the case
    # from previous tests.
    assert rmse < 0.05
