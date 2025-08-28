#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020. Triad National Security, LLC. All rights reserved.  This 
program was produced under U.S. Government contract 89233218CNA000001 for 
Los Alamos National Laboratory (LANL), which is operated by Triad National 
Security, LLC for the U.S.  Department of Energy/National Nuclear Security 
Administration. All rights in the program are reserved by Triad National 

Security, LLC, and the U.S. Department of Energy/National Nuclear Security 
Administration. The Government is granted for itself and others acting on 
its behalf a nonexclusive, paid-up, irrevocable worldwide license in this 
material to reproduce, prepare derivative works, distribute copies to the 
public, perform publicly and display publicly,and to permit others to do so.

LANL software release C19112
Author: Devin Francom
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.special import comb
from itertools import combinations, chain
from collections import namedtuple


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color="red")


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [
        bind.get(itm, None) for itm in a
    ]  # None can be replaced by any other "not in b" value


pos = lambda a: (abs(a) + a) / 2  # same as max(0,a)


def const(signs, knots):
    """Get max value of BASS basis function, assuming 0-1 range of inputs"""
    cc = np.prod(((signs + 1) / 2 - signs * knots))
    if cc == 0:
        return 1
    return cc


def makeBasis(signs, vs, knots, xdata):
    """Make basis function using continuous variables"""
    cc = const(signs, knots)
    temp1 = pos(signs * (xdata[:, vs] - knots))
    if len(signs) == 1:
        return temp1 / cc
    temp2 = np.prod(temp1, axis=1) / cc
    return temp2


def normalize(x, bounds):
    """Normalize to 0-1 scale"""
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize(z, bounds):
    """Inverse of normalize"""
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def comb_index(n, k):
    """Get all combinations of indices from 0:n of length k"""
    # https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    count = comb(n, k, exact=True)
    index = np.fromiter(
        chain.from_iterable(combinations(range(n), k)), int, count=count * k
    )
    return index.reshape(-1, k)


def dmwnchBass(z_vec, vars_use):
    """
    Multivariate Walenius' noncentral hypergeometric density function with 
    some variables fixed
    """
    with np.errstate(divide="ignore"):
        alpha = z_vec[vars_use - 1] / sum(np.delete(z_vec, vars_use))
    j = len(alpha)
    ss = 1 + (-1) ** j * 1 / (sum(alpha) + 1)
    for i in range(j - 1):
        idx = comb_index(j, i + 1)
        temp = alpha[idx]
        ss = ss + (-1) ** (i + 1) * sum(1 / (temp.sum(axis=1) + 1))
    return ss


Qf = namedtuple("Qf", "R bhat qf")


def getQf(XtX, Xty):
    """
    Get the quadratic form y'X solve(X'X) X'y, as well as least squares 
    beta and cholesky of X'X
    """
    try:
        R = sp.linalg.cholesky(
            XtX, lower=False
        )  # might be a better way to do this with sp.linalg.cho_factor
    except np.linalg.LinAlgError as e:
        return None
    dr = np.diag(R)
    if len(dr) > 1:
        if max(dr[1:]) / min(dr) > 1e3:
            return None
    bhat = sp.linalg.solve_triangular(R, sp.linalg.solve_triangular(R, Xty, trans=1))
    qf = np.dot(bhat, Xty)
    return Qf(R, bhat, qf)


def logProbChangeMod(n_int, vars_use, I_vec, z_vec, p, maxInt):
    """Get reversibility factor for RJMCMC acceptance ratio, and also prior"""
    if n_int == 1:
        out = (
            np.log(I_vec[n_int - 1])
            - np.log(2 * p)  # proposal
            + np.log(2 * p)
            + np.log(maxInt)
        )
    else:
        x = np.zeros(p)
        x[vars_use] = 1
        lprob_vars_noReplace = np.log(dmwnchBass(z_vec, vars_use))
        out = (
            np.log(I_vec[n_int - 1])
            + lprob_vars_noReplace
            - n_int * np.log(2)  # proposal
            + n_int * np.log(2)
            + np.log(comb(p, n_int))
            + np.log(maxInt)
        )  # prior
    return out


CandidateBasis = namedtuple("CandidateBasis", "basis n_int signs vs knots lbmcmp")


def genCandBasis(maxInt, I_vec, z_vec, p, xdata):
    """
    Generate a candidate basis for birth step, as well as the RJMCMC 
    reversibility factor and prior
    """
    n_int = int(np.random.choice(range(maxInt), p=I_vec) + 1)
    signs = np.random.choice([-1, 1], size=n_int, replace=True)
    # knots = np.random.rand(n_int)
    knots = np.zeros(n_int)
    if n_int == 1:
        vs = np.random.choice(p)
        knots = np.random.choice(xdata[:, vs], size=1)
    else:
        vs = np.sort(np.random.choice(p, size=n_int, p=z_vec, replace=False))
        for i in range(n_int):
            knots[i] = np.random.choice(xdata[:, vs[i]], size=1)

    basis = makeBasis(signs, vs, knots, xdata)
    lbmcmp = logProbChangeMod(n_int, vs, I_vec, z_vec, p, maxInt)
    return CandidateBasis(basis, n_int, signs, vs, knots, lbmcmp)


BasisChange = namedtuple("BasisChange", "basis signs vs knots")


def genBasisChange(knots, signs, vs, tochange_int, xdata):
    """Generate a condidate basis for change step"""
    knots_cand = knots.copy()
    signs_cand = signs.copy()
    signs_cand[tochange_int] = np.random.choice([-1, 1], size=1)
    knots_cand[tochange_int] = np.random.choice(
        xdata[:, vs[tochange_int]], size=1
    )  # np.random.rand(1)
    basis = makeBasis(signs_cand, vs, knots_cand, xdata)
    return BasisChange(basis, signs_cand, vs, knots_cand)
