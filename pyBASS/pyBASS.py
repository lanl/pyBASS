#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020. Triad National Security, LLC. All rights reserved.  This program
was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC
for the U.S.  Department of Energy/National Nuclear Security Administration. All
rights in the program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The Government is
granted for itself and others acting on its behalf a nonexclusive, paid-up,
irrevocable worldwide license in this material to reproduce, prepare derivative
works, distribute copies to the public, perform publicly and display publicly,
and to permit others to do so.

LANL software release C19112
Author: Devin Francom
"""

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations, chain
from scipy.special import comb
import itertools
from collections import namedtuple
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import time
import re


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='red')


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

pos = lambda a: (abs(a) + a) / 2 # same as max(0,a)


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
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count * k)
    return index.reshape(-1, k)


def dmwnchBass(z_vec, vars_use):
    """Multivariate Walenius' noncentral hypergeometric density function with some variables fixed"""
    with np.errstate(divide='ignore'):
        alpha = z_vec[vars_use - 1] / sum(np.delete(z_vec, vars_use))
    j = len(alpha)
    ss = 1 + (-1) ** j * 1 / (sum(alpha) + 1)
    for i in range(j - 1):
        idx = comb_index(j, i + 1)
        temp = alpha[idx]
        ss = ss + (-1) ** (i + 1) * sum(1 / (temp.sum(axis=1) + 1))
    return ss


Qf = namedtuple('Qf', 'R bhat qf')

def getQf(XtX, Xty):
    """Get the quadratic form y'X solve(X'X) X'y, as well as least squares beta and cholesky of X'X"""
    try:
        R = sp.linalg.cholesky(XtX, lower=False)  # might be a better way to do this with sp.linalg.cho_factor
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
        out = (np.log(I_vec[n_int - 1]) - np.log(2 * p)  # proposal
               + np.log(2 * p) + np.log(maxInt))
    else:
        x = np.zeros(p)
        x[vars_use] = 1
        lprob_vars_noReplace = np.log(dmwnchBass(z_vec, vars_use))
        out = (np.log(I_vec[n_int - 1]) + lprob_vars_noReplace - n_int * np.log(2)  # proposal
               + n_int * np.log(2) + np.log(comb(p, n_int)) + np.log(maxInt))  # prior
    return out


CandidateBasis = namedtuple('CandidateBasis', 'basis n_int signs vs knots lbmcmp')


def genCandBasis(maxInt, I_vec, z_vec, p, xdata):
    """Generate a candidate basis for birth step, as well as the RJMCMC reversibility factor and prior"""
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


BasisChange = namedtuple('BasisChange', 'basis signs vs knots')


def genBasisChange(knots, signs, vs, tochange_int, xdata):
    """Generate a condidate basis for change step"""
    knots_cand = knots.copy()
    signs_cand = signs.copy()
    signs_cand[tochange_int] = np.random.choice([-1, 1], size=1)
    knots_cand[tochange_int] = np.random.choice(xdata[:, vs[tochange_int]], size=1)  # np.random.rand(1)
    basis = makeBasis(signs_cand, vs, knots_cand, xdata)
    return BasisChange(basis, signs_cand, vs, knots_cand)


class BassPrior:
    """Structure to store prior"""
    def __init__(self, maxInt, maxBasis, npart, g1, g2, s2_lower, h1, h2, a_tau, b_tau, w1, w2):
        self.maxInt = maxInt
        self.maxBasis = maxBasis
        self.npart = npart
        self.g1 = g1
        self.g2 = g2
        self.s2_lower = s2_lower
        self.h1 = h1
        self.h2 = h2
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.w1 = w1
        self.w2 = w2
        return


class BassData:
    """Structure to store data"""
    def __init__(self, xx, y):
        self.xx_orig = xx
        self.y = y
        self.ssy = sum(y * y)
        self.n, self.p = xx.shape
        self.bounds = np.column_stack([xx.min(0), xx.max(0)])
        self.xx = normalize(self.xx_orig, self.bounds)
        return


Samples = namedtuple('Samples', 's2 lam tau nbasis nbasis_models n_int signs vs knots beta')
Sample = namedtuple('Sample', 's2 lam tau nbasis nbasis_models n_int signs vs knots beta')


class BassState:
    """The current state of the RJMCMC chain, with methods for getting the log posterior and for updating the state"""
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.s2 = 1.
        self.nbasis = 0
        self.tau = 1.
        self.s2_rate = 1.
        self.R = 1
        self.lam = 1
        self.I_star = np.ones(prior.maxInt) * prior.w1
        self.I_vec = self.I_star / np.sum(self.I_star)
        self.z_star = np.ones(data.p) * prior.w2
        self.z_vec = self.z_star / np.sum(self.z_star)
        self.basis = np.ones([data.n, 1])
        self.nc = 1
        self.knots = np.zeros([prior.maxBasis, prior.maxInt])
        self.signs = np.zeros([prior.maxBasis, prior.maxInt],
                              dtype=int)  # could do "bool_", but would have to transform 0 to -1
        self.vs = np.zeros([prior.maxBasis, prior.maxInt], dtype=int)
        self.n_int = np.zeros([prior.maxBasis], dtype=int)
        self.Xty = np.zeros(prior.maxBasis + 2)
        self.Xty[0] = np.sum(data.y)
        self.XtX = np.zeros([prior.maxBasis + 2, prior.maxBasis + 2])
        self.XtX[0, 0] = data.n
        self.R = np.array([[np.sqrt(data.n)]])  # np.linalg.cholesky(self.XtX[0, 0])
        self.R_inv_t = np.array([[1 / np.sqrt(data.n)]])
        self.bhat = np.mean(data.y)
        self.qf = pow(np.sqrt(data.n) * np.mean(data.y), 2)
        self.count = np.zeros(3)
        self.cmod = False  # has the state changed since the last write (i.e., has a birth, death, or change been accepted)?
        return

    def log_post(self):  # needs updating
        """get current log posterior"""
        lp = (
                - (self.s2_rate + self.prior.g2) / self.s2
                - (self.data.n / 2 + 1 + (self.nbasis + 1) / 2 + self.prior.g1) * np.log(self.s2)
                + np.sum(np.log(abs(np.diag(self.R))))  # .5*determinant of XtX
                + (self.prior.a_tau + (self.nbasis + 1) / 2 - 1) * np.log(self.tau) - self.prior.a_tau * self.tau
                - (self.nbasis + 1) / 2 * np.log(2 * np.pi)
                + (self.prior.h1 + self.nbasis - 1) * np.log(self.lam) - self.lam * (self.prior.h2 + 1)
        )  # curr$nbasis-1 because poisson prior is excluding intercept (for curr$nbasis instead of curr$nbasis+1)
        # -lfactorial(curr$nbasis) # added, but maybe cancels with prior
        self.lp = lp
        return

    def update(self):
        """Update the current state using a RJMCMC step (and Gibbs steps at the end of this function)"""

        move_type = np.random.choice([1, 2, 3])

        if self.nbasis == 0:
            move_type = 1

        if self.nbasis == self.prior.maxBasis:
            move_type = np.random.choice(np.array([2, 3]))

        if move_type == 1:
            ## BIRTH step

            cand = genCandBasis(self.prior.maxInt, self.I_vec, self.z_vec, self.data.p, self.data.xx)

            if (cand.basis > 0).sum() < self.prior.npart:  # if proposed basis function has too few non-zero entries, dont change the state
                return

            ata = np.dot(cand.basis, cand.basis)
            Xta = np.dot(self.basis.T, cand.basis)
            aty = np.dot(cand.basis, self.data.y)

            self.Xty[self.nc] = aty
            self.XtX[0:self.nc, self.nc] = Xta
            self.XtX[self.nc, 0:(self.nc)] = Xta
            self.XtX[self.nc, self.nc] = ata

            qf_cand = getQf(self.XtX[0:(self.nc + 1), 0:(self.nc + 1)], self.Xty[0:(self.nc + 1)])

            fullRank = qf_cand != None
            if not fullRank:
                return

            alpha = .5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau) + np.log(self.lam) - np.log(self.nc) + np.log(
                1 / 3) - np.log(1 / 3) - cand.lbmcmp + .5 * np.log(self.tau) - .5 * np.log(1 + self.tau)

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                # note, XtX and Xty are already updated
                self.nbasis = self.nbasis + 1
                self.nc = self.nbasis + 1
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nc))
                self.count[0] = self.count[0] + 1
                self.n_int[self.nbasis - 1] = cand.n_int
                self.knots[self.nbasis - 1, 0:(cand.n_int)] = cand.knots
                self.signs[self.nbasis - 1, 0:(cand.n_int)] = cand.signs
                self.vs[self.nbasis - 1, 0:(cand.n_int)] = cand.vs

                self.I_star[cand.n_int - 1] = self.I_star[cand.n_int - 1] + 1
                self.I_vec = self.I_star / sum(self.I_star)
                self.z_star[cand.vs] = self.z_star[cand.vs] + 1
                self.z_vec = self.z_star / sum(self.z_star)

                self.basis = np.append(self.basis, cand.basis.reshape(self.data.n, 1), axis=1)


        elif move_type == 2:
            ## DEATH step

            tokill_ind = np.random.choice(self.nbasis)
            ind = list(range(self.nc))
            del ind[tokill_ind + 1]

            qf_cand = getQf(self.XtX[np.ix_(ind, ind)], self.Xty[ind])

            fullRank = qf_cand != None
            if not fullRank:
                return

            I_star = self.I_star.copy()
            I_star[self.n_int[tokill_ind] - 1] = I_star[self.n_int[tokill_ind] - 1] - 1
            I_vec = I_star / sum(I_star)
            z_star = self.z_star.copy()
            z_star[self.vs[tokill_ind, 0:self.n_int[tokill_ind]]] = z_star[self.vs[tokill_ind,
                                                                           0:self.n_int[tokill_ind]]] - 1

            z_vec = z_star / sum(z_star)

            lbmcmp = logProbChangeMod(self.n_int[tokill_ind], self.vs[tokill_ind, 0:self.n_int[tokill_ind]], I_vec,
                                      z_vec, self.data.p, self.prior.maxInt)

            alpha = .5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau) - np.log(self.lam) + np.log(self.nbasis) + np.log(
                1 / 3) - np.log(1 / 3) + lbmcmp - .5 * np.log(self.tau) + .5 * np.log(1 + self.tau)

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                self.nbasis = self.nbasis - 1
                self.nc = self.nbasis + 1
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nc))
                self.count[1] = self.count[1] + 1

                self.Xty[0:self.nc] = self.Xty[ind]
                self.XtX[0:self.nc, 0:self.nc] = self.XtX[np.ix_(ind, ind)]

                temp = self.n_int[0:(self.nbasis + 1)]
                temp = np.delete(temp, tokill_ind)
                self.n_int = self.n_int * 0
                self.n_int[0:(self.nbasis)] = temp[:]

                temp = self.knots[0:(self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.knots = self.knots * 0
                self.knots[0:(self.nbasis), :] = temp[:]

                temp = self.signs[0:(self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.signs = self.signs * 0
                self.signs[0:(self.nbasis), :] = temp[:]

                temp = self.vs[0:(self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.vs = self.vs * 0
                self.vs[0:(self.nbasis), :] = temp[:]

                self.I_star = I_star[:]
                self.I_vec = I_vec[:]
                self.z_star = z_star[:]
                self.z_vec = z_vec[:]

                self.basis = np.delete(self.basis, tokill_ind + 1, 1)

        else:
            ## CHANGE step

            tochange_basis = np.random.choice(self.nbasis)
            tochange_int = np.random.choice(self.n_int[tochange_basis])

            cand = genBasisChange(self.knots[tochange_basis, 0:self.n_int[tochange_basis]],
                                  self.signs[tochange_basis, 0:self.n_int[tochange_basis]],
                                  self.vs[tochange_basis, 0:self.n_int[tochange_basis]], tochange_int, self.data.xx)

            if (cand.basis > 0).sum() < self.prior.npart:  # if proposed basis function has too few non-zero entries, dont change the state
                return

            ata = np.dot(cand.basis.T, cand.basis)
            Xta = np.dot(self.basis.T, cand.basis).reshape(self.nc)
            aty = np.dot(cand.basis.T, self.data.y)

            ind = list(range(self.nc))
            XtX_cand = self.XtX[np.ix_(ind, ind)].copy()
            XtX_cand[tochange_basis + 1, :] = Xta
            XtX_cand[:, tochange_basis + 1] = Xta
            XtX_cand[tochange_basis + 1, tochange_basis + 1] = ata

            Xty_cand = self.Xty[0:self.nc].copy()
            Xty_cand[tochange_basis + 1] = aty

            qf_cand = getQf(XtX_cand, Xty_cand)

            fullRank = qf_cand != None
            if not fullRank:
                return

            alpha = .5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau)

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nc))  # check this
                self.count[2] = self.count[2] + 1

                self.Xty[0:self.nc] = Xty_cand
                self.XtX[0:self.nc, 0:self.nc] = XtX_cand

                self.knots[tochange_basis, 0:self.n_int[tochange_basis]] = cand.knots
                self.signs[tochange_basis, 0:self.n_int[tochange_basis]] = cand.signs

                self.basis[:, tochange_basis + 1] = cand.basis.reshape(self.data.n)

        a_s2 = self.prior.g1 + self.data.n / 2
        b_s2 = self.prior.g2 + .5 * (self.data.ssy - np.dot(self.bhat.T, self.Xty[0:self.nc]) / (1 + self.tau))
        if b_s2 < 0:
            self.prior.g2 = self.prior.g2 + 1.e-10
            b_s2 = self.prior.g2 + .5 * (self.data.ssy - np.dot(self.bhat.T, self.Xty[0:self.nc]) / (1 + self.tau))
        self.s2 = 1 / np.random.gamma(a_s2, 1 / b_s2, size=1)

        self.beta = self.bhat / (1 + self.tau) + np.dot(self.R_inv_t, np.random.normal(size=self.nc)) * np.sqrt(
            self.s2 / (1 + self.tau))

        a_lam = self.prior.h1 + self.nbasis
        b_lam = self.prior.h2 + 1
        self.lam = np.random.gamma(a_lam, 1 / b_lam, size=1)

        temp = np.dot(self.R, self.beta)
        qf2 = np.dot(temp, temp)
        a_tau = self.prior.a_tau + (self.nbasis + 1) / 2
        b_tau = self.prior.b_tau + .5 * qf2 / self.s2
        self.tau = np.random.gamma(a_tau, 1 / b_tau, size=1)




class BassModel:
    """The model structure, including the current RJMCMC state and previous saved states; with methods for saving the
        state, plotting MCMC traces, and predicting"""
    def __init__(self, data, prior, nstore):
        """Get starting state, build storage structures"""
        self.data = data
        self.prior = prior
        self.state = BassState(self.data, self.prior)
        self.nstore = nstore
        s2 = np.zeros(nstore)
        lam = np.zeros(nstore)
        tau = np.zeros(nstore)
        nbasis = np.zeros(nstore, dtype=int)
        nbasis_models = np.zeros(nstore, dtype=int)
        n_int = np.zeros([nstore, self.prior.maxBasis], dtype=int)
        signs = np.zeros([nstore, self.prior.maxBasis, self.prior.maxInt], dtype=int)
        vs = np.zeros([nstore, self.prior.maxBasis, self.prior.maxInt], dtype=int)
        knots = np.zeros([nstore, self.prior.maxBasis, self.prior.maxInt])
        beta = np.zeros([nstore, self.prior.maxBasis + 1])
        self.samples = Samples(s2, lam, tau, nbasis, nbasis_models, n_int, signs, vs, knots, beta)
        self.k = 0
        self.k_mod = -1
        self.model_lookup = np.zeros(nstore, dtype=int)
        return

    def writeState(self):
        """Take relevant parts of state and write to storage (only manipulates storage vectors created in init)"""
        self.samples.s2[self.k] = self.state.s2
        self.samples.lam[self.k] = self.state.lam
        self.samples.tau[self.k] = self.state.tau
        self.samples.beta[self.k, 0:(self.state.nbasis + 1)] = self.state.beta
        self.samples.nbasis[self.k] = self.state.nbasis

        if self.state.cmod: # basis part of state was changed
            self.k_mod = self.k_mod + 1
            self.samples.nbasis_models[self.k_mod] = self.state.nbasis
            self.samples.n_int[self.k_mod, 0:self.state.nbasis] = self.state.n_int[0:self.state.nbasis]
            self.samples.signs[self.k_mod, 0:self.state.nbasis, :] = self.state.signs[0:self.state.nbasis, :]
            self.samples.vs[self.k_mod, 0:self.state.nbasis, :] = self.state.vs[0:self.state.nbasis, :]
            self.samples.knots[self.k_mod, 0:self.state.nbasis, :] = self.state.knots[0:self.state.nbasis, :]
            self.state.cmod = False

        self.model_lookup[self.k] = self.k_mod
        self.k = self.k + 1

    def plot(self):
        """
        Trace plots and predictions/residuals

        * top left - trace plot of number of basis functions (excluding burn-in and thinning)
        * top right - trace plot of residual variance
        * bottom left - training data against predictions
        * bottom right - histogram of residuals (posterior mean) with assumed Gaussian overlaid.
        """
        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        plt.plot(self.samples.nbasis)
        plt.ylabel("number of basis functions")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 2)
        plt.plot(self.samples.s2)
        plt.ylabel("error variance")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 3)
        yhat = self.predict(self.data.xx_orig).mean(axis=0)  # posterior predictive mean
        plt.scatter(self.data.y, yhat)
        abline(1, 0)
        plt.xlabel("observed")
        plt.ylabel("posterior prediction")

        ax = fig.add_subplot(2, 2, 4)
        plt.hist(self.data.y - yhat, color="skyblue", ec="white", density=True)
        axes = plt.gca()
        x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
        plt.plot(x, sp.stats.norm.pdf(x, scale=np.sqrt(self.samples.s2.mean())), color='red')
        plt.xlabel("residuals")
        plt.ylabel("density")

        fig.tight_layout()

        plt.show()

    def makeBasisMatrix(self, model_ind, X):
        """Make basis matrix for model"""
        nb = self.samples.nbasis_models[model_ind]
        #ind_list = [np.arange(self.samples.n_int[model_ind, m]) for m in range(nb)]
        #mat = np.column_stack([
        #    makeBasis(
        #        self.samples.signs[model_ind, m, ind],
        #        self.samples.vs[model_ind, m, ind],
        #        self.samples.knots[model_ind, m, ind],
        #        X
        #    ).squeeze()
        #    for m, ind in enumerate(ind_list)
        #])
        #return np.column_stack([np.ones(len(X)), mat])
    
        n = len(X)
        mat = np.zeros([n, nb + 1])
        mat[:, 0] = 1
        for m in range(nb):
            ind = list(range(self.samples.n_int[model_ind, m]))
            mat[:, m + 1] = makeBasis(self.samples.signs[model_ind, m, ind], self.samples.vs[model_ind, m, ind],
                                      self.samples.knots[model_ind, m, ind], X).reshape(n)
        return mat


    def predict(self, X, mcmc_use=None, nugget=False):
        """
        BASS prediction using new inputs (after training).

        :param X: matrix (numpy array) of predictors with dimension nxp, where n is the number of prediction points and
            p is the number of inputs (features). p must match the number of training inputs, and the order of the
            columns must also match.
        :param mcmc_use: which MCMC samples to use (list of integers of length m).  Defaults to all MCMC samples.
        :param nugget: whether to use the error variance when predicting.  If False, predictions are for mean function.
        :return: a matrix (numpy array) of predictions with dimension mxn, with rows corresponding to MCMC samples and
            columns corresponding to prediction points.
        """
        if X.ndim == 1:
            X = X[None, :]

        Xs = normalize(X, self.data.bounds)
        if np.any(mcmc_use == None):
            mcmc_use = np.array(range(self.nstore))
        out = np.zeros([len(mcmc_use), len(Xs)])
        models = self.model_lookup[mcmc_use]
        umodels = set(models)
        k = 0
        for j in umodels:
            mcmc_use_j = mcmc_use[np.ix_(models == j)]
            nn = len(mcmc_use_j)
            out[range(k, nn + k), :] = np.dot(
                self.samples.beta[mcmc_use_j, 0:(self.samples.nbasis_models[j] + 1)],
                self.makeBasisMatrix(j, Xs).T
            )
            k += nn
        if nugget:
            out += np.random.normal(
                scale=np.sqrt(self.samples.s2[mcmc_use]),
                size=[len(Xs), len(mcmc_use)]
            ).T
        return out


def bass(xx, y, nmcmc=10000, nburn=9000, thin=1, w1=5, w2=5, maxInt=3,
         maxBasis=1000, npart=None, g1=0, g2=0, s2_lower=0, h1=10, h2=10,
         a_tau=0.5, b_tau=None, verbose=True):
    """
    **Bayesian Adaptive Spline Surfaces - model fitting**

    This function takes training data, priors, and algorithmic constants and fits a BASS model.  The result is a set of
    posterior samples of the model.  The resulting object has a predict function to generate posterior predictive
    samples.  Default settings of priors and algorithmic parameters should only be changed by users who understand
    the model.

    :param xx: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and p is
        the number of inputs (features).
    :param y: response vector (numpy array) of length n.
    :param nmcmc: total number of MCMC iterations (integer)
    :param nburn: number of MCMC iterations to throw away as burn-in (integer, less than nmcmc).
    :param thin: number of MCMC iterations to thin (integer).
    :param w1: nominal weight for degree of interaction, used in generating candidate basis functions. Should be greater
        than 0.
    :param w2: nominal weight for variables, used in generating candidate basis functions. Should be greater than 0.
    :param maxInt: maximum degree of interaction for spline basis functions (integer, less than p)
    :param maxBasis: maximum number of tensor product spline basis functions (integer)
    :param npart: minimum number of non-zero points in a basis function. If the response is functional, this refers only
        to the portion of the basis function coming from the non-functional predictors. Defaults to 20 or 0.1 times the
        number of observations, whichever is smaller.
    :param g1: shape for IG prior on residual variance.
    :param g2: scale for IG prior on residual variance.
    :param s2_lower: lower bound for residual variance.
    :param h1: shape for gamma prior on mean number of basis functions.
    :param h2: scale for gamma prior on mean number of basis functions.
    :param a_tau: shape for gamma prior on 1/g in g-prior.
    :param b_tau: scale for gamma prior on 1/g in g-prior.
    :param verbose: boolean for printing progress
    :return: an object of class BassModel, which includes predict and plot functions.
    """

    t0 = time.time()
    if b_tau == None:
        b_tau = len(y) / 2
    if npart == None:
        npart = min(20, .1 * len(y))
    bd = BassData(xx, y)
    if bd.p < maxInt:
        maxInt = bd.p
    bp = BassPrior(maxInt, maxBasis, npart, g1, g2, s2_lower, h1, h2, a_tau, b_tau, w1, w2)
    nstore = int((nmcmc - nburn) / thin)
    bm = BassModel(bd, bp, nstore)  # if we add tempering, bm should have as many states as temperatures
    for i in range(nmcmc):  # rjmcmc loop
        bm.state.update()
        if i > (nburn - 1) and ((i - nburn + 1) % thin) == 0:
            bm.writeState()
        if verbose and i % 500 == 0:
            print('\rBASS MCMC {:.1%} Complete'.format(i / nmcmc), end='')
            # print(str(datetime.now()) + ', nbasis: ' + str(bm.state.nbasis))
    t1 = time.time()
    print('\rBASS MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    # del bm.writeState # the user should not have access to this
    return bm


class PoolBass(object):
    # adapted from https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map/41959862#41959862 answer by parisjohn
    # somewhat slow collection of results
   def __init__(self, x, y, **kwargs):
       self.x = x
       self.y = y
       self.kw = kwargs

   def rowbass(self, i):
       return bass(self.x, self.y[i,:], **self.kw)

   def fit(self, ncores, nrow_y):
      pool = Pool(ncores)
      out = pool.map(self, range(nrow_y))
      return out

   def __call__(self, i):
     return self.rowbass(i)

class PoolBassPredict(object):
   def __init__(self, X, mcmc_use, nugget, bm_list):
       self.X = X
       self.mcmc_use = mcmc_use
       self.nugget = nugget
       self.bm_list = bm_list

   def listpredict(self, i):
       return self.bm_list[i].predict(self.X, self.mcmc_use, self.nugget)

   def predict(self, ncores, nlist):
      pool = Pool(ncores)
      out = pool.map(self, range(nlist))
      return out

   def __call__(self, i):
     return self.listpredict(i)


class BassBasis:
    """Structure for functional response BASS model using a basis decomposition, gets a list of BASS models"""
    def __init__(self, xx, y, basis, newy, y_mean, y_sd, trunc_error, ncores=1, **kwargs):
        """
        Fit BASS model with multivariate/functional response by projecting onto user specified basis.

        :param xx: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and
            p is the number of inputs (features).
        :param y: response matrix (numpy array) of dimension nxq, where q is the number of multivariate/functional
            responses.
        :param basis: matrix (numpy array) of basis functions of dimension qxk.
        :param newy: matrix (numpy array) of y projected onto basis, dimension kxn.
        :param y_mean: vector (numpy array) of length q with the mean if y was centered before obtaining newy.
        :param y_sd: vector (numpy array) of length q with the standard deviation if y was scaled before obtaining newy.
        :param trunc_error: numpy array of projection truncation errors (dimension qxn)
        :param ncores: number of threads to use when fitting independent BASS models (integer less than or equal to
            npc).
        :param kwargs: optional arguments to bass function.
        """
        self.basis = basis
        self.xx = xx
        self.y = y
        self.newy = newy
        self.y_mean = y_mean
        self.y_sd = y_sd
        self.trunc_error = trunc_error
        self.nbasis = len(basis[0])

        if ncores == 1:
            self.bm_list = list(map(lambda ii: bass(self.xx, self.newy[ii, :], **kwargs), list(range(self.nbasis))))
        else:
            #with Pool(ncores) as pool: # this approach for pathos.multiprocessing
            #    self.bm_list = list(
            #        pool.map(lambda ii: bass(self.xx, self.newy[ii, :], **kwargs), list(range(self.nbasis))))
            temp = PoolBass(self.xx, self.newy, **kwargs)
            self.bm_list = temp.fit(ncores, self.nbasis)
        return

    def predict(self, X, mcmc_use=None, nugget=False, trunc_error=False, ncores=1):
        """
        Predict the functional response at new inputs.

        :param X: matrix (numpy array) of predictors with dimension nxp, where n is the number of prediction points and
            p is the number of inputs (features). p must match the number of training inputs, and the order of the
            columns must also match.
        :param mcmc_use: which MCMC samples to use (list of integers of length m).  Defaults to all MCMC samples.
        :param nugget: whether to use the error variance when predicting.  If False, predictions are for mean function.
        :param trunc_error: whether to use truncation error when predicting.
        :param ncores: number of cores to use while predicting (integer).  In almost all cases, use ncores=1.
        :return: a numpy array of predictions with dimension mxnxq, with first dimension corresponding to MCMC samples,
            second dimension corresponding to prediction points, and third dimension corresponding to
            multivariate/functional response.
        """
        if ncores == 1:
            pred_coefs = list(map(lambda ii: self.bm_list[ii].predict(X, mcmc_use, nugget), list(range(self.nbasis))))
        else:
            #with Pool(ncores) as pool:
            #    pred_coefs = list(
            #        pool.map(lambda ii: self.bm_list[ii].predict(X, mcmc_use, nugget), list(range(self.nbasis))))
            temp = PoolBassPredict(X, mcmc_use, nugget, self.bm_list)
            pred_coefs = temp.predict(ncores, self.nbasis)
        out = np.dot(np.dstack(pred_coefs), self.basis.T)
        out2 = out * self.y_sd + self.y_mean
        if trunc_error:
            out2 += self.trunc_error[:, np.random.choice(np.arange(self.trunc_error.shape[1]), size=np.prod(out.shape[:2]), replace=True)].reshape(out.shape)
        return out2

    def plot(self):
        """
        Trace plots and predictions/residuals

        * top left - trace plot of number of basis functions (excluding burn-in and thinning) for each BASS model
        * top right - trace plot of residual variance for each BASS model
        * bottom left - training data against predictions
        * bottom right - histogram of residuals (posterior mean).
        """

        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        for i in range(self.nbasis):
            plt.plot(self.bm_list[i].samples.nbasis)
        plt.ylabel("number of basis functions")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 2)
        for i in range(self.nbasis):
            plt.plot(self.bm_list[i].samples.s2)
        plt.ylabel("error variance")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 3)
        yhat = self.predict(self.bm_list[0].data.xx_orig).mean(axis=0)  # posterior predictive mean
        plt.scatter(self.y, yhat)
        abline(1, 0)
        plt.xlabel("observed")
        plt.ylabel("posterior prediction")

        ax = fig.add_subplot(2, 2, 4)
        plt.hist((self.y - yhat).reshape(np.prod(yhat.shape)), color="skyblue", ec="white", density=True)
        plt.xlabel("residuals")
        plt.ylabel("density")

        fig.tight_layout()

        plt.show()

class BassPCAsetup:
    """
    Wrapper to get principal components that would be used for bassPCA.  Mainly used for checking how many PCs should be used.

    :param y: response matrix (numpy array) of dimension nxq, where n is the number of training examples and q is the number of multivariate/functional
        responses.
    :param npc: number of principal components to use (integer, optional if percVar is specified).
    :param percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
        (if npc=None).
    :param center: whether to center the responses before principal component decomposition (boolean).
    :param scale: whether to scale the responses before principal component decomposition (boolean).
    :return: object with plot method.
    """
    def __init__(self, y, center=True, scale=False):
        self.y = y
        self.y_mean = 0
        self.y_sd = 1
        if center:
            self.y_mean = np.mean(y, axis=0)
        if scale:
            self.y_sd = np.std(y, axis=0)
            self.y_sd[self.y_sd == 0] = 1
        self.y_scale = np.apply_along_axis(lambda row: (row - self.y_mean) / self.y_sd, 1, y)
        #decomp = np.linalg.svd(y_scale.T)
        U, s, V = np.linalg.svd(self.y_scale.T)
        self.evals = s ** 2
        self.basis = np.dot(U, np.diag(s))
        self.newy = V
        return

    def plot(self, npc=None, percVar=None):
        """
        Plot of principal components, eigenvalues

        * left - principal components; grey are excluded by setting of npc or percVar
        * right - eigenvalues (squared singular values), colored according to principal components
        """

        cs = np.cumsum(self.evals) / np.sum(self.evals) * 100.

        if npc == None and percVar == 100:
            npc = len(self.evals)
        if npc == None and percVar is not None:
            npc = np.where(cs >= percVar)[0][0] + 1
        if npc == None or npc > len(self.evals):
            npc = len(self.evals)

        fig = plt.figure()

        cmap = plt.get_cmap("tab10")

        ax = fig.add_subplot(1, 2, 1)
        if npc < len(self.evals):
            plt.plot(self.basis[:, npc:], color='grey')
        for i in range(npc):
            plt.plot(self.basis[:, i], color=cmap(i%10))
        plt.ylabel("principal components")
        plt.xlabel("multivariate/functional index")

        ax = fig.add_subplot(1, 2, 2)
        x = np.arange(len(self.evals)) + 1
        if npc < len(self.evals):
            plt.scatter(x[npc:], cs[npc:], facecolors='none', color='grey')
        for i in range(npc):
            plt.scatter(x[i], cs[i], facecolors='none', color=cmap(i%10))
        plt.axvline(npc)
        #if percVar is not None:
        #    plt.axhline(percVar)
        plt.ylabel("cumulative eigenvalues (percent variance)")
        plt.xlabel("index")

        fig.tight_layout()

        plt.show()

def bassPCA(xx, y, npc=None, percVar=99.9, ncores=1, center=True, scale=False, **kwargs):
    """
    Wrapper to get principal components and call BassBasis, which then calls bass function to fit the BASS model for
    functional (or multivariate) response data.

    :param xx: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and p is
        the number of inputs (features).
    :param y: response matrix (numpy array) of dimension nxq, where q is the number of multivariate/functional
        responses.
    :param npc: number of principal components to use (integer, optional if percVar is specified).
    :param percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
        (if npc=None).
    :param ncores: number of threads to use when fitting independent BASS models (integer less than or equal to npc).
    :param center: whether to center the responses before principal component decomposition (boolean).
    :param scale: whether to scale the responses before principal component decomposition (boolean).
    :param kwargs: optional arguments to bass function.
    :return: object of class BassBasis, with predict and plot functions.
    """

    setup = BassPCAsetup(y, center, scale)

    if npc == None:
        cs = np.cumsum(setup.evals) / np.sum(setup.evals) * 100.
        npc = np.where(cs > percVar)[0][0] + 1

    if ncores > npc:
        ncores = npc

    basis = setup.basis[:, :npc]
    newy = setup.newy[:npc, :]
    trunc_error = np.dot(basis, newy) - setup.y_scale.T

    print('\rStarting bassPCA with {:d} components, using {:d} cores.'.format(npc, ncores))

    return BassBasis(xx, y, basis, newy, setup.y_mean, setup.y_sd, trunc_error, ncores, **kwargs)


class sobolBasis:
    """
    Decomposes the variance of the BASS model into variance due to the main effects, two way
    interactions, and so on, similar to the ANOVA decoposition for linear models.

    Uses the Sobol' decomposition, which can be done analytically for MARS-type models. This is for
    the Basis class

    :param mod: BassBasis model

    :return: object with plot method.
    """
    def __init__(self, mod: BassBasis):
        self.mod = mod
        return
    
    def decomp(self, int_order, prior=None, mcmc_use=None, nind=None, ncores=1):
        """
        Perform Sobol Decomp

        :param int_order: an integer indicating the highest order of interactions to include in the Sobol decomposition.
        :param prior:  a list with the same number of elements as there are inputs to mod.
                       Each element specifies the prior for the particular input.  Each prior is specified as a
                       dictionary with elements (one of "normal", "student", or "uniform"), "trunc" (a vector of dimension 2
                       indicating the lower and upper truncation bounds, taken to be the data bounds if omitted), and for "normal"
                       or "student" priors, "mean" (scalar mean of the Normal/Student, or a vector of means for a mixture of
                       Normals or Students), "sd" (scalar standard deviation of the Normal/Student, or a vector of standard
                       deviations for a mixture of Normals or Students), "df" (scalar degrees of freedom of the Student,
                       or a vector of degrees of freedom for a mixture of Students), and "weights" (a vector of weights that
                       sum to one for the mixture components, or the scalar 1).  If unspecified, a uniform is assumed with the same
                       bounds as are represented in the input to mod.
        :param mcmc_use: an integer indicating which MCMC iteration to use for sensitivity analysis. Defaults to the last iteration.
        :param nind: number of Sobol indices to keep (will keep the largest nind).
        :param ncores: number of cores to use (default = 1)
        """
        self.int_order = int_order
        if mcmc_use == None:
            self.mcmc_use = self.mod.bm_list[0].samples.s2.shape[0]-1
        else:
            self.mcmc_use = mcmc_use
        self.nind = nind
        self.ncores = ncores

        bassMod = self.mod.bm_list[0]

        if prior == None:
            self.prior = []
        else:
            self.prior = prior 
        
        p = bassMod.data.p

        if len(self.prior) < p:
            for i in range(len(self.prior),p):
                tmp = {'dist':'uniform','trunc':None}
                self.prior.append(tmp)
        
        for i in range(len(self.prior)):
            if self.prior[i]['trunc'] == None:
                self.prior[i]['trunc'] = np.array([0,1])
            else:
                self.prior[i]['trunc'] = normalize(self.prior[i]['trunc'], bassMod.data.bounds[:,i])
            
            if self.prior[i]['dist'] == 'normal' or self.prior[i]['dist'] == 'student':
                self.prior[i]['mean'] = normalize(self.prior[i]['mean'], bassMod.data.bounds[:,i])
                self.prior[i]['sd'] = prior[i]['sd']/(bassMod.data.bounds[1,i]-bassMod.data.bounds[0,i])
                if self.prior[i]['dist'] == 'normal':
                    self.prior[i]['z'] = stats.norm.pdf((self.prior[i]['trunc'][1]-self.prior[i]['mean'])/self.prior[i]['sd']) -stats.norm.pdf((self.prior[i]['trunc'][0]-self.prior[i]['mean'])/self.prior[i]['sd'])
                else:
                    self.prior[i]['z'] = stats.t.pdf((self.prior[i]['trunc'][1]-self.prior[i]['mean'])/self.prior[i]['sd'], self.prior[i]['df']) - stats.t.pdf((self.prior[i]['trunc'][0]-self.prior[i]['mean'])/self.prior[i]['sd'], self.prior[i]['df'])
                
                cc = (self.prior[i]['weights']*self.prior[i]['z']).sum()
                self.prior[i]['weights'] = self.prior[i]['weights']/cc
            
        pc_mod = self.mod.bm_list
        pcs = self.mod.basis

        tic = time.perf_counter()
        print('Start\n')
        
        if int_order > p:
            self.int_order = p
            print('int_order > number of inputs, chnage to int_order = number of input\n')
        
        u_list = [list(itertools.combinations(range(0,p), x)) for x in range(1,int_order+1)]
        ncombs_vec = [len(x) for x in u_list]
        ncombs = sum(ncombs_vec)
        nxfunc = pcs.shape[0]

        n_pc = self.mod.nbasis

        w0 = np.zeros(n_pc)
        for i in range(n_pc):
            w0[i] = self.get_f0(pc_mod,i)
        
        f0r2 = (pcs@w0)**2

        tmp = [pc_mod[x].samples.nbasis[self.mcmc_use] for x in range(n_pc)]
        max_nbasis = max(tmp)

        C1Basis_array = np.zeros((n_pc,p,max_nbasis))
        for i in range(n_pc):
            nb = pc_mod[i].samples.nbasis[self.mcmc_use]
            mcmc_mod_usei = pc_mod[i].model_lookup[self.mcmc_use]
            for j in range(p):
                for k in range(nb):
                    C1Basis_array[i,j,k] = self.C1Basis(pc_mod, j+1, k, i, mcmc_mod_usei)
        
        u_list1 = []
        for i in range(int_order):
            u_list1.extend(u_list[i])
        
        toc = time.perf_counter()
        print('Integrating: %0.2fs\n' % (toc-tic))

        u_list_temp = u_list1
        u_list_temp.insert(0,list(np.arange(0,p)))

        if ncores > 1:
            # @todo write parallel version
            NameError('Parallel not Implemented\n')
        else:
             ints1_temp = [self.func_hat(x,pc_mod,pcs,mcmc_use,f0r2,C1Basis_array) for x in u_list_temp]
        
        V_tot = ints1_temp[0]
        ints1 = ints1_temp[1:]
        
        ints = []
        ints.append(np.zeros((ints1[0].shape[0], len(u_list[0]))))
        for i in range(len(u_list[0])):
            ints[0][:,i] = ints1[i]
        
        if int_order > 1:
            for i in range(2,int_order+1):
                idx = np.sum(ncombs_vec[0:(i-1)])+np.arange(0,len(u_list[i-1]))
                ints.append(np.zeros((ints1[0].shape[0], idx.shape[0])))
                cnt = 0
                for j in idx:
                    ints[i-1][:,cnt] = ints1[j]
                    cnt += 1
        
        sob = []
        sob.append(ints[0])
        toc = time.perf_counter()
        print('Shuffling: %0.2fs\n' % (toc-tic))

        if len(u_list) > 1:
            for i in range(1,len(u_list)):
                sob.append(np.zeros((nxfunc,ints[i].shape[1])))
                for j in range(len(u_list[i])):
                    cc = np.zeros(nxfunc)
                    for k in range(i):
                        ind = [np.all(np.in1d(x,u_list[i][j])) for x in u_list[k]]
                        cc += (-1)**(i-k)*np.sum(ints[k][:,ind],axis=1)
                    sob[i][:,j] = ints[i][:,j] + cc
        
        if nind is None:
            nind = ncombs
        
        sob_comb_var = np.concatenate(sob, axis=1)

        vv = np.mean(sob_comb_var,axis=0)
        ord = vv.argsort()[::-1]
        cutoff = vv[ord[nind-1]]
        if nind > ord.shape[0]:
            cutoff = vv.min()
        
        use = np.sort(np.where(vv>=cutoff)[0])

        V_other = V_tot - np.sum(sob_comb_var[:,use],axis=1)

        use = np.append(use, ncombs)

        sob_comb_var = np.hstack((sob_comb_var,V_other[:,np.newaxis])).T
        sob_comb = sob_comb_var/V_tot

        sob_comb_var = sob_comb_var[use,:]
        sob_comb = sob_comb[use,:]

        names_ind1 = []
        for i in range(len(u_list)):
            for j in range(len(u_list[i])):
                tmp = u_list[i][j]
                tmp1 = [x+1 for x in tmp]
                tmp1 = re.findall(r'\d+', str(tmp1))
                if len(tmp1) == 1:
                    names_ind1.append(tmp1[0])
                else:
                    separator = 'x'
                    names_ind1.append(separator.join(tmp1))
        
        names_ind1.append('other')
        names_ind2 = [names_ind1[x] for x in use]

        toc = time.perf_counter()
        print('Finish: %0.2fs\n' % (toc-tic))

        self.S = sob_comb
        self.S_var = sob_comb_var
        self.Var_tot = V_tot
        self.names_ind = names_ind2
        self.xx = np.linspace(0,1,nxfunc)
 
        return
    
    def plot(self, text=False, labels=[], col='Paired', time=[]):
        if len(time) == 0:
            time = self.xx
        
        if len(labels) == 0:
            labels1 = self.names_ind
        else:
            labels1 = self.names_ind
            for i in range(len(labels)):
                labels1[i] = labels[float(labels1[i])]
        
        map = cm.Paired(np.linspace(0,1,12))
        map = np.resize(map, (len(labels1),4))
        rgb = np.ones((map.shape[0]+1,4))
        rgb[0:map.shape[0],:] = map
        rgb[-1,0:3] = np.array([153,153,153])/255

        ord = time.argsort()
        x_mean = self.S
        sens = np.cumsum(x_mean,axis=0).T
        fig, axs = plt.subplots(1, 2)
        idx = np.where(np.sum(sens,axis=0)/sens.shape[0]>=.99999)[0][0]
        cnt = 0
        for i in range(idx+1):
            x2 = np.concatenate((time[ord], np.flip(time[ord])))
            if i == 0:
                inBetween = np.concatenate((np.zeros(time[ord].shape[0]), np.flip(sens[ord,i])))
            else:
                inBetween = np.concatenate((sens[ord,i-1], np.flip(sens[ord,i])))
            if (cnt % rgb.shape[0]+1) == 0:
                cnt = 0
            
            axs[0].fill(x2, inBetween, color=rgb[cnt,:])
            cnt += 1
        
        axs[0].set(xlabel="x", ylabel="proportion variance", title='Sensitivity', ylim=[0,1], xlim=[time.min(), time.max()])

        if text:
            lab_x = np.argmax(x_mean, axis=1)
            cs = np.zeros((sens.shape[1]+1, sens.shape[0]))
            cs[1:,:] = np.cumsum(x_mean,axis=0)
            cs_diff = np.zeros((x_mean.shape[0], x_mean.shape[1]))
            for i in range(x_mean.shape[1]):
                cs_diff[:,i] = np.diff(np.cumsum(np.concatenate((0, x_mean[:,0]))))
            tmp = np.concatenate((np.arange(0,lab_x.shape[0]), lab_x))
            ind = np.ravel_multi_index(np.concatenate((tmp[:,0], tmp[:,1])), dims=cs.shape, order='F')
            ind1 = np.ravel_multi_index(np.concatenate((tmp[:,0], tmp[:,1])), dims=cs_diff.shape, order='F')
            cs_diff2 = cs_diff/2
            plt.text(time[lab_x], cs[ind] + cs_diff2[ind1], self.names_ind)
        
        x_mean_var = self.S_var
        sens_var = np.cumsum(x_mean_var,axis=0).T
        cnt = 0
        for i in range(idx+1):
            x2 = np.concatenate((time[ord], np.flip(time[ord])))
            if i == 0:
                inBetween = np.concatenate((np.zeros(time[ord].shape[0]), np.flip(sens_var[ord,i])))
            else:
                inBetween = np.concatenate((sens_var[ord,i-1], np.flip(sens_var[ord,i])))
            if (cnt % rgb.shape[0]+1) == 0:
                cnt = 0
            
            axs[1].fill(x2, inBetween, color=rgb[cnt,:])
            cnt += 1
        
        axs[1].set(xlabel="x", ylabel="variance", title='Variance Decomposition', xlim=[time.min(), time.max()])

        if not text:
            plt.legend(labels1[0:(idx+1)],loc='upper left')

        fig.tight_layout()
        return
    
    def get_f0(self, pc_mod, pc):
        mcmc_mod_use = pc_mod[pc].model_lookup[self.mcmc_use]
        out = pc_mod[pc].samples.beta[self.mcmc_use,0]
        if (pc_mod[pc].samples.nbasis[self.mcmc_use] > 0):
            for m in range(pc_mod[pc].samples.nbasis[self.mcmc_use]):
                out1 = pc_mod[pc].samples.beta[self.mcmc_use, 1+m]
                for l in range(1,pc_mod[pc].data.p+1):
                    out1 = out1*self.C1Basis(pc_mod,l,m,pc,mcmc_mod_use)
                out += out1
        return out
    

    def C1Basis(self, pc_mod, l, m, pc, mcmc_mod_use):
        int_use_l = np.where(pc_mod[pc].samples.vs[mcmc_mod_use,m,:]==l)[0]
        if int_use_l.size == 0:
            out = 1
            return out

        s = pc_mod[pc].samples.signs[mcmc_mod_use,m,int_use_l]
        t = pc_mod[pc].samples.knots[mcmc_mod_use,m,int_use_l]
        q = 1
        
        if s == 0:
            out = 0
            return out
        
        cc = const(s, t)

        if s == 1:
            a = np.maximum(self.prior[l-1]['trunc'][0],t)
            b = self.prior[l]['trunc'][1]
            if b < t:
                out = 0
                return out
            out = self.intabq1(self.prior[l-1],a,b,t,q)/cc
        else:
            a = self.prior[l-1]['trunc'][0]
            b = np.minimum(self.prior[l-1]['trunc'][1],t)
            if t < a:
                out = 0
                return out
            out = self.intabq1(self.prior[l-1],a,b,t,q)*(-1)**q/cc
        
        return out
    
    def intabq1(self, prior, a, b, t, q):
        if prior['dist'] == 'normal':
            if q != 1:
                NameError('degree other than 1 not supported for normal priors')
            
            out = 0
            for k in range(len(prior['weights'])):
                zk = stats.norm.pdf(b, prior['mean'][k], prior['sd'][k]) - stats.norm.pdf(a, prior['mean'][k], prior['sd'][k])
                ast = (a-prior['mean'][k])/prior['sd'][k]
                bst = (b-prior['mean'][k])/prior['sd'][k]
                dnb = stats.norm.cdf(bst)
                dna = stats.norm.cdf(ast)
                tnorm_mean_zk = prior['mean'][k]*zk - prior['sd'][k]*(dnb-dna)
                out += prior['weights'][k] * (tnorm_mean_zk-t*zk)
        
        if prior['dist'] == 'student':
            if q != 1:
                NameError('degree other than 1 not supported for normal priors')
            
            out = 0
            for k in range(len(prior['weights'])):
                int = self.intx1Student(b, prior['mean'][k], prior['sd'][k], prior['df'][k],t) - self.intx1Student(a, prior['mean'][k], prior['sd'][k], prior['df'][k],t)
                out += prior['weights'][k] * int
        
        if prior['dist'] == 'uniform':
            out = 1/(q+1)*((b-t)**(q+1)-(a-t)**(q+1)) * 1/(prior['trunc'][1]-prior['trunc'][0])
        
        return out

    def intx1Student(self, x, m, s, v, t):
        temp = (s**2*v)/(m**2 + s**2*v - 2*m*x + x**2)
        out = -((v/(v + (m - x)**2/s^2))**(v/2) * 
              np.sqrt(temp) * 
              np.sqrt(1/temp) *
              (s**2*v* (np.sqrt(1/temp) - 
              (1/temp)**(v/2)) + 
              (t-m)*(-1 + v)*(-m + x) * 
              (1/temp)**(v/2) *
              self.robust2f1(1/2,(1 + v)/2,3/2,-(m - x)**2/(s**2 *v)) )) / (s *(-1 + v)* np.sqrt(v) * sp.special.beta(v/2, 1/2))
        
        return out
    
    def robust2f1(sself,a,b,c,x):
        if np.abs(x) < 1:
            z = sp.special.hyp2f1(a,b,c,np.array([0,x]))
            out = z[-1]
        else:
            z = sp.special.hyp2f1(a,c-b,c,0)
            out = z[-1]
        
        return(out)
    
    def func_hat(self,u,pc_mod,pcs,mcmc_use,f0r2,C1Basis_array):
        res = np.zeros(pcs.shape[0])
        n_pc = len(pc_mod)
        for i in range(n_pc):
            res += pcs[:,i]**2*self.Ccross(pc_mod,i,i,u,C1Basis_array)

            if (i+1) < n_pc:
                for j in range(i+1,n_pc):
                    res = res + 2 * pcs[:,i]*pcs[:,j] * self.Ccross(pc_mod,i,j,u,C1Basis_array)

        out = res - f0r2

        return out

    def Ccross(self,pc_mod,i,j,u,C1Basis_array):
        p = pc_mod[0].data.p
        mcmc_mod_usei = pc_mod[i].model_lookup[self.mcmc_use]
        mcmc_mod_usej = pc_mod[j].model_lookup[self.mcmc_use]

        Mi = pc_mod[i].samples.nbasis[self.mcmc_use]
        Mj = pc_mod[j].samples.nbasis[self.mcmc_use]

        a0i = pc_mod[i].samples.beta[self.mcmc_use,0]
        a0j = pc_mod[j].samples.beta[self.mcmc_use,0]
        f0i = self.get_f0(pc_mod,i)
        f0j = self.get_f0(pc_mod,j)

        out = a0i*a0j + a0i*(f0j-a0j) + a0j*(f0i-a0i)

        if (Mi > 0 and Mj > 0):
            ai = pc_mod[i].samples.beta[self.mcmc_use,1:(Mi+1)]
            aj = pc_mod[j].samples.beta[self.mcmc_use,1:(Mj+1)]
        
        for mi in range(Mi):
            for mj in range(Mj):
                temp1 = ai[mi]*aj[mj]
                temp2 = 1
                temp3 = 1
                idx = np.arange(0,p)
                idx2 = u
                idx = np.delete(idx,idx2)

                for l in idx:
                    temp2 = temp2 * C1Basis_array[i,l,mi]*C1Basis_array[j,l,mj]
                
                for l in idx2:
                    temp3 = temp3 * self.C2Basis(pc_mod,l+1,mi,mj,i,j,mcmc_mod_usei,mcmc_mod_usej)
                
                out += temp1*temp2*temp3
        
        return out
    

    def C2Basis(self,pc_mod,l,m1,m2,pc1,pc2,mcmc_mod_use1,mcmc_mod_use2):

        if (l <= pc_mod[pc1].data.p):
            int_use_l1 = np.where(pc_mod[pc1].samples.vs[mcmc_mod_use1,m1,:]==l)[0]
            int_use_l2 = np.where(pc_mod[pc2].samples.vs[mcmc_mod_use2,m2,:]==l)[0]

            if int_use_l1.size == 0 and int_use_l2.size == 0:
                out = 1
                return out
            
            if int_use_l1.size == 0:
                out = self.C1Basis(pc_mod,l,m2,pc2,mcmc_mod_use2)
                return out
            
            if int_use_l2.size == 0:
                out = self.C1Basis(pc_mod,l,m1,pc1,mcmc_mod_use1)
                return out
            
            q = 1
            s1 = pc_mod[pc1].samples.signs[mcmc_mod_use1,m1,int_use_l1]
            s2 = pc_mod[pc2].samples.signs[mcmc_mod_use2,m2,int_use_l2]
            t1 = pc_mod[pc1].samples.knots[mcmc_mod_use1,m1,int_use_l1]
            t2 = pc_mod[pc2].samples.knots[mcmc_mod_use2,m2,int_use_l2]

            if t2 < t1:
                temp = t1
                t1 = t2
                t2 = temp
                temp = s1
                s1 = s2
                s2 = temp
            
            out = self.C22Basis(self.prior[l-1],t1,t2,s1,s2,q)

        return out
    
    def C22Basis(self,prior,t1,t2,s1,s2,q):
        cc = const(np.array([s1,s2]), np.array([t1,t2]))
        out = 0
        if (s1*s2) == 0:
            out = 0
            return out
        
        if (s1 == 1):
            if (s2 == 1):
                out = self.intabq2(prior,t2,1,t1,t2,q)/cc
                return out
            else:
                out = self.intabq2(prior,t1,t2,t1,t2,q)*(-1)**q/cc
                return out
        else:
            if (s2 == 1):
                out = 0
                return out
            else:
                out = self.intabq2(prior,0,t1,t1,t2,q)/cc
                return out
            
        return out
    
    def intabq2(self, prior, a, b, t1, t2, q):
        if prior['dist'] == 'normal':
            if q != 1:
                NameError('degree other than 1 not supported for normal priors')
            
            out = 0
            for k in range(len(prior['weights'])):
                zk = stats.norm.pdf(b, prior['mean'][k], prior['sd'][k]) - stats.norm.pdf(a, prior['mean'][k], prior['sd'][k])
                if zk < np.finfo(float).eps:
                    continue
                ast = (a-prior['mean'][k])/prior['sd'][k]
                bst = (b-prior['mean'][k])/prior['sd'][k]
                dnb = stats.norm.cdf(bst)
                dna = stats.norm.cdf(ast)
                tnorm_mean_zk = prior['mean'][k]*zk - prior['sd'][k]*(dnb-dna)
                tnorm_var_zk = zk*prior['sd'][k]**2*(1 + (ast*dna-bst*dnb)/zk - ((dna-dnb)/zk)**2) + tnorm_mean_zk**2/zk
                out += prior['weights'][k] * (tnorm_var_zk - (t1+t2)*tnorm_mean_zk + t1*t2*zk)
                if (out < 0 and np.abs(out)< 1e-12):
                    out = 0
        
        if prior['dist'] == 'student':
            if q != 1:
                NameError('degree other than 1 not supported for normal priors')
            
            out = 0
            for k in range(len(prior['weights'])):
                int = self.intx2Student(b, prior['mean'][k], prior['sd'][k], prior['df'][k],t1,t2) - self.intx2Student(a, prior['mean'][k], prior['sd'][k], prior['df'][k],t1,t2)
                out += prior['weights'][k] * int
        
        if prior['dist'] == 'uniform':
            out = (np.sum(self.pCoef(np.arange(0,q+1),q)*(b-t1)**(q-np.arange(0,q+1))*(b-t2)**(q+1+np.arange(0,q+1))) - np.sum(self.pCoef(np.arange(0,q+1),q)*(a-t1)**(q-np.arange(0,q+1))*(a-t2)**(q+1+np.arange(0,q+1)))) * 1/(prior['trunc'][1]-prior['trunc'][0])
        
        return out
    
    def intx2Student(self,x,m,s,v,t1,t2):
        temp = (s**2*v)/(m**2 + s**2*v - 2*m*x + x**2)
        out = ((v/(v + (m - x)**2/s**2))**(v/2) *
            np.sqrt(temp) *
            np.sqrt(1/temp) *
            (-3*(-t1-t2+2*m)*s**2*v* (np.sqrt(1/temp) -
            (1/temp)**(v/2)) + 
            3*(-t1+m)*(-t2+m)*(-1 + v)*(-m + x) *
            (1/temp)**(v/2) * 
            self.robust2f1(1/2,(1 + v)/2,3/2,-(m - x)**2/(s**2 *v)) + 
            (-1+v)*(-m+x)**3*(1/temp)**(v/2) * 
            self.robust2f1(3/2,(1 + v)/2,5/2,-(m - x)**2/(s**2 *v)) )) / (3*s *(-1 + v)* np.sqrt(v) *sp.special.beta(v/2, 1/2))
        
        return out
    
    def pCoef(self,i,q):
        out = sp.special.factorial(q)**2*(-1)**i/(sp.special.factorial(q-i)*sp.special.factorial(q+1+i))
        return out

