import numpy as np

import pyBASS as pb

tt = np.linspace(0, 1, 50)  # functional variable grid


# Get dataset
# Friedman function with functional response
def f2(x):
    out = (
        10.0 * np.sin(np.pi * tt * x[0])
        + 20.0 * (x[1] - 0.5) ** 2
        + 10.0 * x[2]
        + 5.0 * x[3]
    )
    return out


np.random.seed(0)
n = 500  # sample size
p = 9  # number of predictors other (only 4 are used)
x = np.random.rand(n, p)  # training inputs
e = np.random.normal(size=[n, len(tt)]) * 0.1  # noise
y = np.apply_along_axis(f2, 1, x) + e  # training response

# fit BASS model with RJMCMC
mod = pb.bassPCA(x, y)

sob = pb.sobolBasis(mod)
sob.decomp(int_order=1)
sob.plot()

# Check that T_var is computed correctly
S_var = sob.S_var
T_var = sob.T_var
np.all(
    T_var[0]
    == np.sum(
        [S_var[i] for i in range(len(S_var)) if str(1) in sob.names_ind[i]],
        axis=0,
    )
)
np.all(
    T_var[1]
    == np.sum(
        [S_var[i] for i in range(len(S_var)) if str(2) in sob.names_ind[i]],
        axis=0,
    )
)
np.all(
    T_var[2]
    == np.sum(
        [S_var[i] for i in range(len(S_var)) if str(3) in sob.names_ind[i]],
        axis=0,
    )
)
np.all(
    T_var[3]
    == np.sum(
        [S_var[i] for i in range(len(S_var)) if str(4) in sob.names_ind[i]],
        axis=0,
    )
)
### etc
np.all(
    T_var[8]
    == np.sum(
        [S_var[i] for i in range(len(S_var)) if str(9) in sob.names_ind[i]],
        axis=0,
    )
)
