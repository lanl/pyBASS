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

import time
from collections import namedtuple
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import pyBASS.utils as uf


class BassPrior:
    """Structure to store prior"""

    def __init__(
        self,
        maxInt,
        maxBasis,
        npart,
        g1,
        g2,
        s2_lower,
        h1,
        h2,
        a_tau,
        b_tau,
        w1,
        w2,
    ):
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
        self.xx = uf.normalize(self.xx_orig, self.bounds)
        return


Samples = namedtuple(
    "Samples", "s2 lam tau nbasis nbasis_models n_int signs vs knots beta"
)
Sample = namedtuple(
    "Sample", "s2 lam tau nbasis nbasis_models n_int signs vs knots beta"
)


class BassState:
    """
    The current state of the RJMCMC chain, with methods for getting the log
    posterior and for updating the state
    """

    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.s2 = 1.0
        self.nbasis = 0
        self.tau = 1.0
        self.s2_rate = 1.0
        self.R = 1
        self.lam = 1
        self.I_star = np.ones(prior.maxInt) * prior.w1
        self.I_vec = self.I_star / np.sum(self.I_star)
        self.z_star = np.ones(data.p) * prior.w2
        self.z_vec = self.z_star / np.sum(self.z_star)
        self.basis = np.ones([data.n, 1])
        self.nc = 1
        self.knots = np.zeros([prior.maxBasis, prior.maxInt])
        self.signs = np.zeros(
            [prior.maxBasis, prior.maxInt], dtype=int
        )  # could do "bool_", but would have to transform 0 to -1
        self.vs = np.zeros([prior.maxBasis, prior.maxInt], dtype=int)
        self.n_int = np.zeros([prior.maxBasis], dtype=int)
        self.Xty = np.zeros(prior.maxBasis + 2)
        self.Xty[0] = np.sum(data.y)
        self.XtX = np.zeros([prior.maxBasis + 2, prior.maxBasis + 2])
        self.XtX[0, 0] = data.n
        self.R = np.array([
            [np.sqrt(data.n)]
        ])  # np.linalg.cholesky(self.XtX[0, 0])
        self.R_inv_t = np.array([[1 / np.sqrt(data.n)]])
        self.bhat = np.mean(data.y)
        self.qf = pow(np.sqrt(data.n) * np.mean(data.y), 2)
        self.count = np.zeros(3)
        self.cmod = False  # has the state changed since the last write (i.e., has a birth, death, or change been accepted)?
        return

    def log_post(self):  # needs updating
        """get current log posterior"""
        lp = (
            -(self.s2_rate + self.prior.g2) / self.s2
            - (self.data.n / 2 + 1 + (self.nbasis + 1) / 2 + self.prior.g1)
            * np.log(self.s2)
            + np.sum(np.log(abs(np.diag(self.R))))  # .5*determinant of XtX
            + (self.prior.a_tau + (self.nbasis + 1) / 2 - 1) * np.log(self.tau)
            - self.prior.a_tau * self.tau
            - (self.nbasis + 1) / 2 * np.log(2 * np.pi)
            + (self.prior.h1 + self.nbasis - 1) * np.log(self.lam)
            - self.lam * (self.prior.h2 + 1)
        )  # curr$nbasis-1 because poisson prior is excluding intercept (for curr$nbasis instead of curr$nbasis+1)
        # -lfactorial(curr$nbasis) # added, but maybe cancels with prior
        self.lp = lp
        return

    def update(self):
        """
        Update the current state using a RJMCMC step (and Gibbs steps at
        the end of this function)
        """

        move_type = np.random.choice([1, 2, 3])

        if self.nbasis == 0:
            move_type = 1

        if self.nbasis == self.prior.maxBasis:
            move_type = np.random.choice(np.array([2, 3]))

        if move_type == 1:
            # BIRTH step

            cand = uf.genCandBasis(
                self.prior.maxInt,
                self.I_vec,
                self.z_vec,
                self.data.p,
                self.data.xx,
            )

            # if proposed basis function has too few non-zero entries,
            # dont change the state
            if (cand.basis > 0).sum() < self.prior.npart:
                return

            ata = np.dot(cand.basis, cand.basis)
            Xta = np.dot(self.basis.T, cand.basis)
            aty = np.dot(cand.basis, self.data.y)

            self.Xty[self.nc] = aty
            self.XtX[0 : self.nc, self.nc] = Xta
            self.XtX[self.nc, 0 : (self.nc)] = Xta
            self.XtX[self.nc, self.nc] = ata

            qf_cand = uf.getQf(
                self.XtX[0 : (self.nc + 1), 0 : (self.nc + 1)],
                self.Xty[0 : (self.nc + 1)],
            )

            fullRank = qf_cand is not None
            if not fullRank:
                return

            alpha = (
                0.5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau)
                + np.log(self.lam)
                - np.log(self.nc)
                + np.log(1 / 3)
                - np.log(1 / 3)
                - cand.lbmcmp
                + 0.5 * np.log(self.tau)
                - 0.5 * np.log(1 + self.tau)
            )

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                # note, XtX and Xty are already updated
                self.nbasis = self.nbasis + 1
                self.nc = self.nbasis + 1
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(
                    self.R, np.identity(self.nc)
                )
                self.count[0] = self.count[0] + 1
                self.n_int[self.nbasis - 1] = cand.n_int
                self.knots[self.nbasis - 1, 0 : (cand.n_int)] = cand.knots
                self.signs[self.nbasis - 1, 0 : (cand.n_int)] = cand.signs
                self.vs[self.nbasis - 1, 0 : (cand.n_int)] = cand.vs

                self.I_star[cand.n_int - 1] = self.I_star[cand.n_int - 1] + 1
                self.I_vec = self.I_star / sum(self.I_star)
                self.z_star[cand.vs] = self.z_star[cand.vs] + 1
                self.z_vec = self.z_star / sum(self.z_star)

                self.basis = np.append(
                    self.basis, cand.basis.reshape(self.data.n, 1), axis=1
                )

        elif move_type == 2:
            # DEATH step

            tokill_ind = np.random.choice(self.nbasis)
            ind = list(range(self.nc))
            del ind[tokill_ind + 1]

            qf_cand = uf.getQf(self.XtX[np.ix_(ind, ind)], self.Xty[ind])

            fullRank = qf_cand is not None
            if not fullRank:
                return

            I_star = self.I_star.copy()
            I_star[self.n_int[tokill_ind] - 1] = (
                I_star[self.n_int[tokill_ind] - 1] - 1
            )
            I_vec = I_star / sum(I_star)
            z_star = self.z_star.copy()
            z_star[self.vs[tokill_ind, 0 : self.n_int[tokill_ind]]] = (
                z_star[self.vs[tokill_ind, 0 : self.n_int[tokill_ind]]] - 1
            )

            z_vec = z_star / sum(z_star)

            lbmcmp = uf.logProbChangeMod(
                self.n_int[tokill_ind],
                self.vs[tokill_ind, 0 : self.n_int[tokill_ind]],
                I_vec,
                z_vec,
                self.data.p,
                self.prior.maxInt,
            )

            alpha = (
                0.5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau)
                - np.log(self.lam)
                + np.log(self.nbasis)
                + np.log(1 / 3)
                - np.log(1 / 3)
                + lbmcmp
                - 0.5 * np.log(self.tau)
                + 0.5 * np.log(1 + self.tau)
            )

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                self.nbasis = self.nbasis - 1
                self.nc = self.nbasis + 1
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(
                    self.R, np.identity(self.nc)
                )
                self.count[1] = self.count[1] + 1

                self.Xty[0 : self.nc] = self.Xty[ind]
                self.XtX[0 : self.nc, 0 : self.nc] = self.XtX[np.ix_(ind, ind)]

                temp = self.n_int[0 : (self.nbasis + 1)]
                temp = np.delete(temp, tokill_ind)
                self.n_int = self.n_int * 0
                self.n_int[0 : (self.nbasis)] = temp[:]

                temp = self.knots[0 : (self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.knots = self.knots * 0
                self.knots[0 : (self.nbasis), :] = temp[:]

                temp = self.signs[0 : (self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.signs = self.signs * 0
                self.signs[0 : (self.nbasis), :] = temp[:]

                temp = self.vs[0 : (self.nbasis + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.vs = self.vs * 0
                self.vs[0 : (self.nbasis), :] = temp[:]

                self.I_star = I_star[:]
                self.I_vec = I_vec[:]
                self.z_star = z_star[:]
                self.z_vec = z_vec[:]

                self.basis = np.delete(self.basis, tokill_ind + 1, 1)

        else:
            # CHANGE step

            tochange_basis = np.random.choice(self.nbasis)
            tochange_int = np.random.choice(self.n_int[tochange_basis])

            cand = uf.genBasisChange(
                self.knots[tochange_basis, 0 : self.n_int[tochange_basis]],
                self.signs[tochange_basis, 0 : self.n_int[tochange_basis]],
                self.vs[tochange_basis, 0 : self.n_int[tochange_basis]],
                tochange_int,
                self.data.xx,
            )

            # if proposed basis function has too few non-zero entries,
            # dont change the state
            if (cand.basis > 0).sum() < self.prior.npart:
                return

            ata = np.dot(cand.basis.T, cand.basis)
            Xta = np.dot(self.basis.T, cand.basis).reshape(self.nc)
            aty = np.dot(cand.basis.T, self.data.y)

            ind = list(range(self.nc))
            XtX_cand = self.XtX[np.ix_(ind, ind)].copy()
            XtX_cand[tochange_basis + 1, :] = Xta
            XtX_cand[:, tochange_basis + 1] = Xta
            XtX_cand[tochange_basis + 1, tochange_basis + 1] = ata.item()

            Xty_cand = self.Xty[0 : self.nc].copy()
            Xty_cand[tochange_basis + 1] = aty.item()

            qf_cand = uf.getQf(XtX_cand, Xty_cand)

            fullRank = qf_cand is not None
            if not fullRank:
                return

            alpha = 0.5 / self.s2 * (qf_cand.qf - self.qf) / (1 + self.tau)

            if np.log(np.random.rand()) < alpha:
                self.cmod = True
                self.qf = qf_cand.qf
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(
                    self.R, np.identity(self.nc)
                )  # check this
                self.count[2] = self.count[2] + 1

                self.Xty[0 : self.nc] = Xty_cand
                self.XtX[0 : self.nc, 0 : self.nc] = XtX_cand

                self.knots[tochange_basis, 0 : self.n_int[tochange_basis]] = (
                    cand.knots
                )
                self.signs[tochange_basis, 0 : self.n_int[tochange_basis]] = (
                    cand.signs
                )

                self.basis[:, tochange_basis + 1] = cand.basis.reshape(
                    self.data.n
                )

        a_s2 = self.prior.g1 + self.data.n / 2
        b_s2 = self.prior.g2 + 0.5 * (
            self.data.ssy
            - np.dot(self.bhat.T, self.Xty[0 : self.nc]) / (1 + self.tau)
        )
        if b_s2 < 0:
            self.prior.g2 = self.prior.g2 + 1.0e-10
            b_s2 = self.prior.g2 + 0.5 * (
                self.data.ssy
                - np.dot(self.bhat.T, self.Xty[0 : self.nc]) / (1 + self.tau)
            )
        self.s2 = 1 / np.random.gamma(a_s2, 1 / b_s2, size=1)

        self.beta = self.bhat / (1 + self.tau) + np.dot(
            self.R_inv_t, np.random.normal(size=self.nc)
        ) * np.sqrt(self.s2 / (1 + self.tau))

        a_lam = self.prior.h1 + self.nbasis
        b_lam = self.prior.h2 + 1
        self.lam = np.random.gamma(a_lam, 1 / b_lam, size=1)

        temp = np.dot(self.R, self.beta)
        qf2 = np.dot(temp, temp)
        a_tau = self.prior.a_tau + (self.nbasis + 1) / 2
        b_tau = self.prior.b_tau + 0.5 * qf2 / self.s2
        self.tau = np.random.gamma(a_tau, 1 / b_tau, size=1)


class BassModel:
    """
    The model structure, including the current RJMCMC state and previous
    saved states; with methods for saving the
    state, plotting MCMC traces, and predicting
    """

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
        signs = np.zeros(
            [nstore, self.prior.maxBasis, self.prior.maxInt], dtype=int
        )
        vs = np.zeros(
            [nstore, self.prior.maxBasis, self.prior.maxInt], dtype=int
        )
        knots = np.zeros([nstore, self.prior.maxBasis, self.prior.maxInt])
        beta = np.zeros([nstore, self.prior.maxBasis + 1])
        self.samples = Samples(
            s2, lam, tau, nbasis, nbasis_models, n_int, signs, vs, knots, beta
        )
        self.k = 0
        self.k_mod = -1
        self.model_lookup = np.zeros(nstore, dtype=int)
        return

    def writeState(self):
        """
        Take relevant parts of state and write to storage (only manipulates
        storage vectors created in init)
        """

        self.samples.s2[self.k] = self.state.s2.item()
        self.samples.lam[self.k] = self.state.lam.item()
        self.samples.tau[self.k] = self.state.tau.item()
        self.samples.beta[self.k, 0 : (self.state.nbasis + 1)] = self.state.beta
        self.samples.nbasis[self.k] = self.state.nbasis

        if self.state.cmod:  # basis part of state was changed
            self.k_mod = self.k_mod + 1
            self.samples.nbasis_models[self.k_mod] = self.state.nbasis
            self.samples.n_int[self.k_mod, 0 : self.state.nbasis] = (
                self.state.n_int[0 : self.state.nbasis]
            )
            self.samples.signs[self.k_mod, 0 : self.state.nbasis, :] = (
                self.state.signs[0 : self.state.nbasis, :]
            )
            self.samples.vs[self.k_mod, 0 : self.state.nbasis, :] = (
                self.state.vs[0 : self.state.nbasis, :]
            )
            self.samples.knots[self.k_mod, 0 : self.state.nbasis, :] = (
                self.state.knots[0 : self.state.nbasis, :]
            )
            self.state.cmod = False

        self.model_lookup[self.k] = self.k_mod
        self.k = self.k + 1

    def plot(self):
        """
        Trace plots and predictions/residuals

        * top left - trace plot of number of basis functions
                     (excluding burn-in and thinning)
        * top right - trace plot of residual variance
        * bottom left - training data against predictions
        * bottom right - histogram of residuals (posterior mean) with
                         assumed Gaussian overlaid.
        """
        fig = plt.figure()

        fig.add_subplot(2, 2, 1)
        plt.plot(self.samples.nbasis)
        plt.ylabel("number of basis functions")
        plt.xlabel("MCMC iteration (post-burn)")

        fig.add_subplot(2, 2, 2)
        plt.plot(self.samples.s2)
        plt.ylabel("error variance")
        plt.xlabel("MCMC iteration (post-burn)")

        fig.add_subplot(2, 2, 3)
        # posterior predictive mean
        yhat = self.predict(self.data.xx_orig).mean(axis=0)
        plt.scatter(self.data.y, yhat)
        uf.abline(1, 0)
        plt.xlabel("observed")
        plt.ylabel("posterior prediction")

        fig.add_subplot(2, 2, 4)
        plt.hist(self.data.y - yhat, color="skyblue", ec="white", density=True)
        axes = plt.gca()
        x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
        plt.plot(
            x,
            sp.stats.norm.pdf(x, scale=np.sqrt(self.samples.s2.mean())),
            color="red",
        )
        plt.xlabel("residuals")
        plt.ylabel("density")

        fig.tight_layout()

        plt.show()

    def makeBasisMatrix(self, model_ind, X):
        """Make basis matrix for model"""
        nb = self.samples.nbasis_models[model_ind]

        n = len(X)
        mat = np.zeros([n, nb + 1])
        mat[:, 0] = 1
        for m in range(nb):
            ind = list(range(self.samples.n_int[model_ind, m]))
            mat[:, m + 1] = uf.makeBasis(
                self.samples.signs[model_ind, m, ind],
                self.samples.vs[model_ind, m, ind],
                self.samples.knots[model_ind, m, ind],
                X,
            ).reshape(n)
        return mat

    def predict(self, X, mcmc_use=None, nugget=False):
        """
        BASS prediction using new inputs (after training).

        :param X: matrix (numpy array) of predictors with dimension nxp, where
                  n is the number of prediction points and
                  p is the number of inputs (features). p must match the
                  number of training inputs, and the order of the columns must
                  also match.
        :param mcmc_use: which MCMC samples to use (list of integers of length
                         m).  Defaults to all MCMC samples.
        :param nugget: whether to use the error variance when predicting.
                       If False, predictions are for mean function.
        :return: a matrix (numpy array) of predictions with dimension mxn,
                 with rows corresponding to MCMC samples and columns
                 corresponding to prediction points.
        """
        if X.ndim == 1:
            X = X[None, :]

        Xs = uf.normalize(X, self.data.bounds)
        if np.any(mcmc_use is None):
            mcmc_use = np.array(range(self.nstore))
        out = np.zeros([len(mcmc_use), len(Xs)])
        models = self.model_lookup[mcmc_use]
        umodels = set(models)
        k = 0
        for j in umodels:
            mcmc_use_j = mcmc_use[np.ix_(models == j)]
            nn = len(mcmc_use_j)
            out[range(k, nn + k), :] = np.dot(
                self.samples.beta[
                    mcmc_use_j, 0 : (self.samples.nbasis_models[j] + 1)
                ],
                self.makeBasisMatrix(j, Xs).T,
            )
            k += nn
        if nugget:
            out += np.random.normal(
                scale=np.sqrt(self.samples.s2[mcmc_use]),
                size=[len(Xs), len(mcmc_use)],
            ).T
        return out


def bass(
    xx,
    y,
    nmcmc=10000,
    nburn=9000,
    thin=1,
    w1=5,
    w2=5,
    maxInt=3,
    maxBasis=1000,
    npart=None,
    g1=0,
    g2=0,
    s2_lower=0,
    h1=10,
    h2=10,
    a_tau=0.5,
    b_tau=None,
    verbose=True,
):
    """
    **Bayesian Adaptive Spline Surfaces - model fitting**

    This function takes training data, priors, and algorithmic constants and
    fits a BASS model.  The result is a set of posterior samples of the model.
    The resulting object has a predict function to generate posterior
    predictive samples.  Default settings of priors and algorithmic parameters
    should only be changed by users who understand the model.

    :param xx: matrix (numpy array) of predictors of dimension nxp, where n is
               the number of training examples and p is the number of inputs
               (features).
    :param y: response vector (numpy array) of length n.
    :param nmcmc: total number of MCMC iterations (integer)
    :param nburn: number of MCMC iterations to throw away as burn-in (integer,
                  less than nmcmc).
    :param thin: number of MCMC iterations to thin (integer).
    :param w1: nominal weight for degree of interaction, used in generating
               candidate basis functions. Should be greater than 0.
    :param w2: nominal weight for variables, used in generating candidate
               basis functions. Should be greater than 0.
    :param maxInt: maximum degree of interaction for spline basis functions
                   (integer, less than p)
    :param maxBasis: maximum number of tensor product spline basis functions
                     (integer)
    :param npart: minimum number of non-zero points in a basis function. If
                  the response is functional, this refers only to the portion
                  of the basis function coming from the non-functional
                  predictors. Defaults to 20 or 0.1 times the number of
                  observations, whichever is smaller.
    :param g1: shape for IG prior on residual variance.
    :param g2: scale for IG prior on residual variance.
    :param s2_lower: lower bound for residual variance.
    :param h1: shape for gamma prior on mean number of basis functions.
    :param h2: scale for gamma prior on mean number of basis functions.
    :param a_tau: shape for gamma prior on 1/g in g-prior.
    :param b_tau: scale for gamma prior on 1/g in g-prior.
    :param verbose: boolean for printing progress
    :return: an object of class BassModel, which includes predict and plot
             functions.
    """

    t0 = time.time()
    if b_tau is None:
        b_tau = len(y) / 2
    if npart is None:
        npart = min(20, 0.1 * len(y))
    bd = BassData(xx, y)
    if bd.p < maxInt:
        maxInt = bd.p
    bp = BassPrior(
        maxInt, maxBasis, npart, g1, g2, s2_lower, h1, h2, a_tau, b_tau, w1, w2
    )
    nstore = int((nmcmc - nburn) / thin)
    bm = BassModel(
        bd, bp, nstore
    )  # if we add tempering, bm should have as many states as temperatures
    for i in range(nmcmc):  # rjmcmc loop
        bm.state.update()
        if i > (nburn - 1) and ((i - nburn + 1) % thin) == 0:
            bm.writeState()
        if verbose and i % 500 == 0:
            print("\rBASS MCMC {:.1%} Complete".format(i / nmcmc), end="")
            # print(str(datetime.now()) + ', nbasis: ' + str(bm.state.nbasis))
    t1 = time.time()
    print("\rBASS MCMC Complete. Time: {:f} seconds.".format(t1 - t0))
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
        return bass(self.x, self.y[i, :], **self.kw)

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
    """
    Structure for functional response BASS model using a basis
    decomposition, gets a list of BASS models
    """

    def __init__(
        self, xx, y, basis, newy, y_mean, y_sd, trunc_error, ncores=1, **kwargs
    ):
        """
        Fit BASS model with multivariate/functional response by projecting
        onto user specified basis.

        :param xx: matrix (numpy array) of predictors of dimension nxp, where
                   n is the number of training examples and p is the number of
                   inputs (features).
        :param y: response matrix (numpy array) of dimension nxq, where q is
                  the number of multivariate/functional responses.
        :param basis: matrix (numpy array) of basis functions of dimension qxk.
        :param newy: matrix (numpy array) of y projected onto basis, dimension
                     kxn.
        :param y_mean: vector (numpy array) of length q with the mean if y was
                       centered before obtaining newy.
        :param y_sd: vector (numpy array) of length q with the standard
                     deviation if y was scaled before obtaining newy.
        :param trunc_error: numpy array of projection truncation errors
                            (dimension qxn)
        :param ncores: number of threads to use when fitting independent BASS
                       models (integer less than or equal to npc).
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
            self.bm_list = list(
                map(
                    lambda ii: bass(self.xx, self.newy[ii, :], **kwargs),
                    list(range(self.nbasis)),
                )
            )
        else:
            temp = PoolBass(self.xx, self.newy, **kwargs)
            self.bm_list = temp.fit(ncores, self.nbasis)
        return

    def predict(
        self, X, mcmc_use=None, nugget=False, trunc_error=False, ncores=1
    ):
        """
        Predict the functional response at new inputs.

        :param X: matrix (numpy array) of predictors with dimension nxp, where
                  n is the number of prediction points and p is the number of
                  inputs (features). p must match the number of training
                  inputs, and the order of the columns must also match.
        :param mcmc_use: which MCMC samples to use (list of integers of length
                         m). Defaults to all MCMC samples.
        :param nugget: whether to use the error variance when predicting.
                       If False, predictions are for mean function.
        :param trunc_error: whether to use truncation error when predicting.
        :param ncores: number of cores to use while predicting (integer).
                       In almost all cases, use ncores=1.
        :return: a numpy array of predictions with dimension mxnxq, with first
                 dimension corresponding to MCMC samples, second dimension
                 corresponding to prediction points, and third dimension
                 corresponding to multivariate/functional response.
        """
        if ncores == 1:
            pred_coefs = list(
                map(
                    lambda ii: self.bm_list[ii].predict(X, mcmc_use, nugget),
                    list(range(self.nbasis)),
                )
            )
        else:
            temp = PoolBassPredict(X, mcmc_use, nugget, self.bm_list)
            pred_coefs = temp.predict(ncores, self.nbasis)
        out = np.dot(np.dstack(pred_coefs), self.basis.T)
        out2 = out * self.y_sd + self.y_mean
        if trunc_error:
            out2 += self.trunc_error[
                :,
                np.random.choice(
                    np.arange(self.trunc_error.shape[1]),
                    size=np.prod(out.shape[:2]),
                    replace=True,
                ),
            ].reshape(out.shape)
        return out2

    def plot(self):
        """
        Trace plots and predictions/residuals

        * top left - trace plot of number of basis functions
          (excluding burn-in and thinning) for each BASS model
        * top right - trace plot of residual variance for each BASS model
        * bottom left - training data against predictions
        * bottom right - histogram of residuals (posterior mean).
        """

        fig = plt.figure()

        fig.add_subplot(2, 2, 1)
        for i in range(self.nbasis):
            plt.plot(self.bm_list[i].samples.nbasis)
        plt.ylabel("number of basis functions")
        plt.xlabel("MCMC iteration (post-burn)")

        fig.add_subplot(2, 2, 2)
        for i in range(self.nbasis):
            plt.plot(self.bm_list[i].samples.s2)
        plt.ylabel("error variance")
        plt.xlabel("MCMC iteration (post-burn)")

        fig.add_subplot(2, 2, 3)
        yhat = self.predict(self.bm_list[0].data.xx_orig).mean(
            axis=0
        )  # posterior predictive mean
        plt.scatter(self.y, yhat)
        uf.abline(1, 0)
        plt.xlabel("observed")
        plt.ylabel("posterior prediction")

        fig.add_subplot(2, 2, 4)
        plt.hist(
            (self.y - yhat).reshape(np.prod(yhat.shape)),
            color="skyblue",
            ec="white",
            density=True,
        )
        plt.xlabel("residuals")
        plt.ylabel("density")

        fig.tight_layout()

        plt.show()


class BassPCAsetup:
    """
    Wrapper to get principal components that would be used for bassPCA.
    Mainly used for checking how many PCs should be used.

    :param y: response matrix (numpy array) of dimension nxq, where n is the
              number of training examples and q is the number of
              multivariate/functional responses.
    :param npc: number of principal components to use (integer, optional if
                percVar is specified).
    :param percVar: percent (between 0 and 100) of variation to explain when
                    choosing number of principal components (if npc=None).
    :param center: whether to center the responses before principal component
                   decomposition (boolean).
    :param scale: whether to scale the responses before principal component
                  decomposition (boolean).
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
        self.y_scale = np.apply_along_axis(
            lambda row: (row - self.y_mean) / self.y_sd, 1, y
        )
        # decomp = np.linalg.svd(y_scale.T)
        U, s, V = np.linalg.svd(self.y_scale.T, full_matrices=False)
        self.evals = s**2
        self.basis = np.dot(U, np.diag(s))
        self.newy = V
        return

    def plot(self, npc=None, percVar=None):
        """
        Plot of principal components, eigenvalues

        * left - principal components; grey are excluded by setting of npc or
                 percVar
        * right - eigenvalues (squared singular values), colored according to
                  principal components
        """

        cs = np.cumsum(self.evals) / np.sum(self.evals) * 100.0

        if npc is None and percVar == 100:
            npc = len(self.evals)
        if npc is None and percVar is not None:
            npc = np.where(cs >= percVar)[0][0] + 1
        if npc is None or npc > len(self.evals):
            npc = len(self.evals)

        fig = plt.figure()

        cmap = plt.get_cmap("tab10")

        fig.add_subplot(1, 2, 1)
        if npc < len(self.evals):
            plt.plot(self.basis[:, npc:], color="grey")
        for i in range(npc):
            plt.plot(self.basis[:, i], color=cmap(i % 10))
        plt.ylabel("principal components")
        plt.xlabel("multivariate/functional index")

        fig.add_subplot(1, 2, 2)
        x = np.arange(len(self.evals)) + 1
        if npc < len(self.evals):
            plt.scatter(x[npc:], cs[npc:], facecolors="none", color="grey")
        for i in range(npc):
            plt.scatter(x[i], cs[i], facecolors="none", color=cmap(i % 10))
        plt.axvline(npc)
        # if percVar is not None:
        #    plt.axhline(percVar)
        plt.ylabel("cumulative eigenvalues (percent variance)")
        plt.xlabel("index")

        fig.tight_layout()

        plt.show()


def bassPCA(
    xx, y, npc=None, percVar=99.9, ncores=1, center=True, scale=False, **kwargs
):
    """
    Wrapper to get principal components and call BassBasis, which then calls
    bass function to fit the BASS model for functional (or multivariate)
    response data.

    :param xx: matrix (numpy array) of predictors of dimension nxp, where n is
               the number of training examples and p is the number of inputs
               (features).
    :param y: response matrix (numpy array) of dimension nxq, where q is the
              number of multivariate/functional responses.
    :param npc: number of principal components to use (integer, optional if
                percVar is specified).
    :param percVar: percent (between 0 and 100) of variation to explain when
                    choosing number of principal components(if npc=None).
    :param ncores: number of threads to use when fitting independent BASS
                   models (integer less than or equal to npc).
    :param center: whether to center the responses before principal component
                   decomposition (boolean).
    :param scale: whether to scale the responses before principal component
                  decomposition (boolean).
    :param kwargs: optional arguments to bass function.
    :return: object of class BassBasis, with predict and plot functions.
    """

    setup = BassPCAsetup(y, center, scale)

    if npc is None:
        cs = np.cumsum(setup.evals) / np.sum(setup.evals) * 100.0
        npc = np.where(cs > percVar)[0][0] + 1

    if ncores > npc:
        ncores = npc

    basis = setup.basis[:, :npc]
    newy = setup.newy[:npc, :]
    trunc_error = np.dot(basis, newy) - setup.y_scale.T

    print(
        "\rStarting bassPCA with {:d} components, using {:d} cores.".format(
            npc, ncores
        )
    )

    return BassBasis(
        xx,
        y,
        basis,
        newy,
        setup.y_mean,
        setup.y_sd,
        trunc_error,
        ncores,
        **kwargs,
    )
