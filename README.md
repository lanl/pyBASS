# pyBASS
[![Build Status][build-status-img]](https://github.com/lanl/pyBASS/actions)
[![PyPI Version][pypi-version]](https://pypi.org/project/pybass-emu/)
[![PyPI Downloads][monthly-downloads]](https://pypistats.org/packages/pybass-emu)

A python implementation of Bayesian adaptive spline surfaces (BASS).  Similar
to Bayesian multivariate adaptive regression splines (Bayesian MARS) introduced
in Denison _et al_. (1998).

## Installation
```bash
# pip
pip install pybass-emu

# uv
uv add pybass-emu
```

## Examples
* [Example 1][ex1] - univariate response
* [Example 2][ex2] - multivariate/functional response


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
[pypi-version]: https://img.shields.io/pypi/v/pybass-emu?style=flat-square&label=PyPI
[monthly-downloads]: https://img.shields.io/pypi/dm/pybass-emu?style=flat-square&label=Downloads&color=blue
[ex1]: https://github.com/lanl/pyBASS/blob/main/examples/ex1.ipynb
[ex2]: https://github.com/lanl/pyBASS/blob/main/examples/ex2.ipynb
