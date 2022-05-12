# pyBASS
[![Build Status][build-status-img]](https://github.com/lanl/pyBASS/actions)

A python implementation of Bayesian adaptive spline surfaces (BASS).  Similar
to Bayesian multivariate adaptive regression splines (Bayesian MARS) introduced
in Denison _et al_. (1998).

## Installation
Use
```bash
pip install git+https://github.com/lanl/pyBASS.git
```
[Example 1](examples/ex1.md)
## Example 1: Univariate Response

```python
import pyBASS as pb
import numpy as np
import matplotlib.pyplot as plt

# Friedman function (Friedman, 1991, Multivariate Adaptive Regression Splines)
def f(x):
    return (10. * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20. * (x[:, 2] - .5) ** 2 
            + 10 * x[:, 3] + 5. * x[:, 4])


n = 500 # sample size
p = 10 # number of predictors (only 5 are used)
x = np.random.rand(n, p) # training inputs
xx = np.random.rand(1000, p) # test inputs
y = f(x) + np.random.normal(size=n) # training outputs

# fit BASS model with RJMCMC
mod = pb.bass(x, y)
mod.plot()

# predict at new inputs (xx)
pred = mod.predict(xx)
plt.scatter(f(xx), pred.mean(axis=0)) # posterior mean prediction
plt.show()
```

## Example 2: Multivariate/Functional Response

```python
import pyBASS as pb
import numpy as np

# Friedman function where first variable is the functional variable
def f2(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out


tt = np.linspace(0, 1, 50) # functional variable grid
n = 500 # sample size
p = 9 # number of predictors other (only 4 are used)
x = np.random.rand(n, p) # training inputs
xx = np.random.rand(1000, p)
e = np.random.normal(size=[n, len(tt)]) # noise
y = np.apply_along_axis(f2, 1, x) + e # training response

modf = pb.bassPCA(x, y, ncores=2, percVar=99.99)
modf.plot()

pred = modf.predict(xx, mcmc_use=np.array([1,100]), nugget=True)
```


## References
1. Friedman, J.H., 1991. Multivariate adaptive regression splines. _The annals of statistics_, pp.1-67.

2. Denison, D.G., Mallick, B.K. and Smith, A.F., 1998. Bayesian MARS. _Statistics and Computing_, 8(4), pp.337-346.

3. Francom, D., Sansó, B., Kupresanin, A. and Johannesson, G., 2018. Sensitivity analysis and emulation for functional data using Bayesian adaptive splines. _Statistica Sinica_, pp.791-816.

4. Francom, D., Sansó, B., Bulaevskaya, V., Lucas, D. and Simpson, M., 2019. Inferring atmospheric release characteristics in a large computer experiment using Bayesian adaptive splines. _Journal of the American Statistical Association_, 114(528), pp.1450-1465.

5. Francom, D. and Sansó, B., 2020. BASS: An R package for fitting and performing sensitivity analysis of Bayesian adaptive spline surfaces. _Journal of Statistical Software_, 94(1), pp.1-36.



************

Copyright 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

LANL software release C19112

Author: Devin Francom

[build-status-img]: https://github.com/lanl/pyBASS/workflows/Build/badge.svg

