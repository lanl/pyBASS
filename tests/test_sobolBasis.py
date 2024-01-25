import pyBASS as pb
import numpy as np

# Get dataset
# Friedman function with functional response
def f2(x):
    out = (10. * np.sin(np.pi * np.linspace(0, 1, 50) * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]) * x[5]
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

sob = pb.sobolBasis(mod)
sob.decomp(int_order=3)
sob.plot()

# Check that T_var is computed correctly
S_var = sob.S_var
T_var = sob.T_var
np.all(T_var[0] == np.sum([S_var[i] for i in range(len(S_var)) if str(1) in sob.names_ind[i]], axis=0))
np.all(T_var[1] == np.sum([S_var[i] for i in range(len(S_var)) if str(2) in sob.names_ind[i]], axis=0))
np.all(T_var[2] == np.sum([S_var[i] for i in range(len(S_var)) if str(3) in sob.names_ind[i]], axis=0))
np.all(T_var[3] == np.sum([S_var[i] for i in range(len(S_var)) if str(4) in sob.names_ind[i]], axis=0))
### etc
np.all(T_var[8] == np.sum([S_var[i] for i in range(len(S_var)) if str(9) in sob.names_ind[i]], axis=0))
