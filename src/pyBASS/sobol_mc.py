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

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyBASS import BassModel
from pyBASS.utils import normalize


def getCombs(mod, uniq_models, nmodels, max_basis, max_int_tot, func_var=None):
    """
    Get all the variable combinations used in the models, storing in proper structures.
    """
    des_labs = [i for i in range(mod.data.p)]
    labs = des_labs  # this is the order things end up in

    n_un = list()
    for i in uniq_models:
        for j in range(mod.samples.nbasis[uniq_models[i]]):
            temp = sorted(
                mod.samples.vs[i, j, : mod.samples.n_int[i, j]].tolist()
            )
            if temp not in n_un:
                n_un.append(temp)
    n_un = [sublist for sublist in n_un if sublist]

    int_lower = []  # lower order interactions
    for comb in n_un:
        if len(comb) > 1:
            for r in range(1, len(comb) + 1):
                int_lower.extend(list(c) for c in list(combinations(comb, r)))

    for el in int_lower:
        if el not in n_un:
            n_un.append(
                el
            )  # add lower order interactions if they are not there already

    ord_int_size = np.argsort(np.argsort([len(x) for x in n_un]))
    n_un = [item for value, item in sorted(zip(ord_int_size, n_un))]

    int_begin_ind = []
    ints_used = []
    for i in range(len(n_un)):
        temp = len(n_un[i])
        if temp not in ints_used:
            ints_used.append(temp)
            int_begin_ind.append(i)

    int_begin_ind.append(
        len(n_un)
    )  # need the top level to help with indexing later

    int_begin_ind = [x for x in int_begin_ind if x is not None]

    combs, names_ind, disp_combs, disp_names = [], [], [], []
    for i in ints_used:
        if int_begin_ind[i - 1] is not None:
            mat = np.array([
                n_un[j] for j in range(int_begin_ind[i - 1], int_begin_ind[i])
            ])
            mat = mat[np.lexsort(mat.T), :]
            combs.append(mat.T)
            names_ind.append(["x".join(map(str, x)) for x in mat])
            disp_combs = np.array([[labs[y] for y in x] for x in mat])
            disp_combs = np.apply_along_axis(
                lambda x: sorted(x, key=lambda y: (isinstance(y, int), y)),
                axis=1,
                arr=disp_combs,
            )
            if i == 1:
                disp_names.append(disp_combs)
            else:
                disp_names.append(["x".join(map(str, x)) for x in disp_combs])

    num_ind = [
        x.shape[1] for x in combs
    ]  # num_ind[i] is number of interactions of order i
    cs_num_ind = np.cumsum(num_ind)  # used for indexing

    return {
        "combs": combs,
        "names_ind": names_ind,
        "num_ind": num_ind,
        "cs_num_ind": cs_num_ind,
        "disp_combs": disp_combs,
        "disp_names": disp_names,
    }


def pos(vec):
    """Makes negative values 0"""
    return (np.abs(vec) + vec) / 2


def const(signs, knots, degree):
    """Largest value of basis function, assuming x's in [0,1], used for scaling"""
    cc = np.prod((signs + 1) / 2 - signs * knots) ** degree
    return 1 if cc == 0 else cc


def h(x, sign, knot):
    """Scaled hockey-stick function"""
    cc = const(sign, knot, 1)
    return pos(sign * (x - knot)) / cc


def H(x, signs, knots):
    """Product of multiple hockey-sticks"""
    cc = const(signs, knots, 1)
    out = np.ones(x.shape[0])
    for i in range(len(signs)):
        out *= pos(signs[i] * (x[:, i] - knots[i]))
    return out / cc


def C1(var, sign, knot, xx):
    return np.mean(h(xx[:, var], sign, knot))


def C2(var, s1, s2, k1, k2, xx):
    return np.mean(h(xx[:, var], s1, k1) * h(xx[:, var], s2, k2))


def C1sub(varsub, signs, knots, xx):
    return np.mean(H(xx[:, varsub], signs, knots))


def C2sub(varsub1, varsub2, s1s, s2s, k1s, k2s, xx):
    return np.mean(H(xx[:, varsub1], s1s, k1s) * H(xx[:, varsub2], s2s, k2s))


def get_Cmats(mod, mod_idx, mcmc_use, xx):
    tmp = np.where(mod.model_lookup == mod_idx)[0]
    mcmc_idx = tmp[np.isin(tmp, mcmc_use)]
    nb = mod.samples.nbasis[mcmc_idx[0]]
    C1mat = np.ones((nb, mod.data.p))
    varmat = np.zeros((nb, mod.data.p))

    for m in range(nb):
        nint = mod.samples.n_int[mod_idx, m]
        vars = mod.samples.vs[mod_idx, m, :nint]
        varmat[m, vars] = 1
        signs = mod.samples.signs[mod_idx, m, :nint]
        knots = mod.samples.knots[mod_idx, m, :nint]

        for i in range(nint):
            C1mat[m, vars[i]] = C1(vars[i], signs[i], knots[i], xx)

    C2arr = np.ones((nb, nb, mod.data.p))

    for v in range(mod.data.p):
        bases_v_idx = np.where(varmat[:, v] == 1)[0]
        if len(bases_v_idx) > 0:
            for m1 in bases_v_idx:
                nint1 = mod.samples.n_int[mod_idx, m1]
                vind1 = np.where(mod.samples.vs[mod_idx, m1, :nint1] == v)[0]
                sign1 = mod.samples.signs[mod_idx, m1, vind1]
                knot1 = mod.samples.knots[mod_idx, m1, vind1]

                for m2 in range(nb):
                    if m2 in bases_v_idx:
                        nint2 = mod.samples.n_int[mod_idx, m2]
                        vind2 = np.where(
                            mod.samples.vs[mod_idx, m2, :nint2] == v
                        )[0]
                        sign2 = mod.samples.signs[mod_idx, m2, vind2]
                        knot2 = mod.samples.knots[mod_idx, m2, vind2]
                        C2arr[m1, m2, v] = C2arr[m2, m1, v] = C2(
                            v, sign1, sign2, knot1, knot2, xx
                        )
                    else:
                        C2arr[m1, m2, v] = C2arr[m2, m1, v] = C1mat[m1, v]

    C1all = np.ones((nb, nb, mod.data.p))
    for i in range(mod.data.p):
        C1all[:, :, i] = np.outer(C1mat[:, i], C1mat[:, i])

    C1allprod = np.prod(C1all, axis=2)

    return {
        "mcmc_idx": mcmc_idx,
        "nb": nb,
        "C1mat": C1mat,
        "varmat": varmat,
        "C2arr": C2arr,
        "C1all": C1all,
        "C1allprod": C1allprod,
        "amat": mod.samples.beta[mcmc_idx, 1 : nb + 1],
    }


def get_Cmats_sub(mod, mod_idx, mcmc_use, xx, subsets):
    subvec = np.array([
        next(i for i, s in enumerate(subsets) if j in s)
        for j in range(mod.data.p)
    ])

    tmp = np.where(mod.model_lookup == mod_idx)[0]
    mcmc_idx = tmp[np.isin(tmp, mcmc_use)]
    nb = mod.samples.nbasis[mcmc_idx[0]]
    C1mat = np.ones((nb, len(subsets)))
    submat = np.zeros((nb, len(subsets)))
    varmat = np.zeros((nb, mod.data.p))

    for m in range(nb):
        nint = mod.samples.n_int[mod_idx, m]
        vars = mod.samples.vs[mod_idx, m, :nint]
        varmat[m, vars] = 1
        subs = np.unique(subvec[vars])
        submat[m, subs] = 1
        signs = mod.samples.signs[mod_idx, m, :nint]
        knots = mod.samples.knots[mod_idx, m, :nint]

        for i, sub in enumerate(subs):
            ind = np.isin(vars, subsets[sub])
            C1mat[m, sub] = C1sub(vars[ind], signs[ind], knots[ind], xx)

    C2arr = np.ones((nb, nb, len(subsets)))

    for u in range(len(subsets)):
        bases_u_idx = np.where(submat[:, u] == 1)[0]
        if len(bases_u_idx) > 0:
            for m1 in bases_u_idx:
                nint1 = mod.samples.n_int[mod_idx, m1]
                vind1 = np.where(
                    np.isin(mod.samples.vs[mod_idx, m1, :nint1], subsets[u])
                )[0]
                vars1 = mod.samples.vs[mod_idx, m1, vind1]
                sign1 = mod.samples.signs[mod_idx, m1, vind1]
                knot1 = mod.samples.knots[mod_idx, m1, vind1]

                for m2 in range(nb):
                    if m2 in bases_u_idx:
                        nint2 = mod.samples.n_int[mod_idx, m2]
                        vind2 = np.where(
                            np.isin(
                                mod.samples.vs[mod_idx, m2, :nint2], subsets[u]
                            )
                        )[0]
                        vars2 = mod.samples.vs[mod_idx, m2, vind2]
                        sign2 = mod.samples.signs[mod_idx, m2, vind2]
                        knot2 = mod.samples.knots[mod_idx, m2, vind2]
                        C2arr[m1, m2, u] = C2arr[m2, m1, u] = C2sub(
                            vars1, vars2, sign1, sign2, knot1, knot2, xx
                        )
                    else:
                        C2arr[m1, m2, u] = C2arr[m2, m1, u] = C1mat[m1, u]

    C1all = np.ones((nb, nb, len(subsets)))
    for i in range(len(subsets)):
        C1all[:, :, i] = np.outer(C1mat[:, i], C1mat[:, i])

    C1allprod = np.prod(C1all, axis=2)

    return {
        "mcmc_idx": mcmc_idx,
        "nb": nb,
        "C1mat": C1mat,
        "submat": submat,
        "C2arr": C2arr,
        "C1all": C1all,
        "C1allprod": C1allprod,
        "amat": mod.samples.beta[mcmc_idx, 1 : nb + 1],
    }


def vce(u, cmats):
    C1u = np.prod(cmats["C1all"][:, :, u], axis=2)
    C2u = np.prod(cmats["C2arr"][:, :, u], axis=2)
    mat = cmats["C1allprod"] * (C2u / C1u - 1)
    # mat[np.isnan(mat)] = 0
    out = np.array([x.T @ mat @ x for x in cmats["amat"]])
    return out


def vce_combine(u, combs, cs_num_ind, integrals):
    add = np.zeros(integrals.shape[0])
    len_u = len(u)
    for l in range(1, len_u + 1):
        low_ind = cs_num_ind[l - 2]
        if l - 2 < 0:
            low_ind = 0
        ind = np.arange(low_ind, cs_num_ind[l - 1])[
            np.all(np.isin(combs[l - 1], u), axis=0)
        ]
        add += (-1) ** (len_u - l) * np.sum(integrals[:, ind], axis=1)

    add[np.abs(add) < 1e-13] = 0
    return add


def get_sob(mcmc_use, mod, xx, ncores):
    models = mod.model_lookup[mcmc_use]
    uniq_models = np.unique(models)
    nmodels = len(uniq_models)
    maxInt_tot = np.sum(mod.prior.maxInt)
    maxBasis = np.max(mod.samples.nbasis)
    allCombs = getCombs(mod, uniq_models, nmodels, maxBasis, maxInt_tot)
    combs = allCombs["combs"]
    cs_num_ind = allCombs["cs_num_ind"]

    vces = np.zeros((len(mcmc_use), np.max(cs_num_ind)))
    tot_var = np.zeros(len(mcmc_use))

    for mod_idx in uniq_models:
        ind = np.where(models == mod_idx)[0]
        cmats = get_Cmats(mod=mod, mod_idx=mod_idx, mcmc_use=mcmc_use, xx=xx)
        k = 0
        for l in range(len(combs)):
            for i in range(combs[l].shape[1]):
                vces[ind, k] = vce(combs[l][:, i], cmats)
                k += 1
        tot_var[ind] = vce(np.arange(mod.data.p), cmats)

    vces_normed = vces.copy()
    k = combs[0].shape[1]
    if len(combs) > 1:
        for l in range(1, len(combs)):
            for i in range(combs[l].shape[1]):
                vces_normed[:, k] = vce_combine(
                    combs[l][:, i], combs, cs_num_ind, vces
                )
                k += 1

    out = pd.DataFrame(vces_normed / tot_var[:, np.newaxis])
    out.columns = ["x".join(map(str, c)) for comb in combs for c in comb.T]

    return {"S": out, "tot_var": tot_var}


def get_sob_sub(mcmc_use, mod, xx, subsets):
    """
    Get VCEs and normalize them into Sobol indices.
    """
    models = mod.model_lookup[
        mcmc_use
    ]  # only do the heavy lifting once for each model
    uniq_models = np.unique(models)
    nmodels = len(uniq_models)
    maxInt_tot = np.sum(mod.prior.maxInt)
    maxBasis = np.max(mod.samples.nbasis)
    allCombs = getCombs(mod, uniq_models, nmodels, maxBasis, maxInt_tot)
    combs = allCombs["combs"]  # which main effects and interactions included
    cs_num_ind = allCombs["cs_num_ind"]  # cumsum of num_ind

    # convert combs into subset version (subcombs)
    subcombs = []
    for i in range(len(combs)):
        for j in range(combs[i].shape[1]):
            subcombs.append([
                k
                for k, s in enumerate(subsets)
                if any(item in s for item in combs[i][:, j])
            ])
    subcombs_tuples = {tuple(sublist) for sublist in subcombs}
    subcombs = [list(tup) for tup in subcombs_tuples]
    # subcombs = np.unique(subcombs)
    subcombs_size = [len(x) for x in subcombs]
    subcombs = [
        np.column_stack([
            subcombs[j] for j in range(len(subcombs)) if len(subcombs[j]) == i
        ])
        for i in range(1, max(subcombs_size) + 1)
    ]
    subcs_num_ind = np.cumsum([x.shape[1] for x in subcombs])

    vces = np.zeros((len(mcmc_use), max(subcs_num_ind)))
    tot_var = np.zeros(len(mcmc_use))
    for mod_idx in uniq_models:
        ind = np.where(models == mod_idx)[0]
        cmats = get_Cmats_sub(
            mod=mod, mod_idx=mod_idx, mcmc_use=mcmc_use, xx=xx, subsets=subsets
        )
        k = 0
        for l in range(len(subcombs)):
            for i in range(subcombs[l].shape[1]):
                vces[ind, k] = vce(subcombs[l][:, i], cmats)
                k += 1
        tot_var[ind] = vce(np.arange(len(subsets)), cmats)

    # normalize vces
    vces_normed = vces.copy()
    k = subcombs[0].shape[1]
    if len(subcombs) > 1:
        for l in range(1, len(subcombs)):
            for i in range(subcombs[l].shape[1]):
                vces_normed[:, k] = vce_combine(
                    subcombs[l][:, i], subcombs, subcs_num_ind, vces
                )
                k += 1

    out = pd.DataFrame((vces_normed.T / tot_var).T)
    out.columns = [
        "x".join(map(str, subcombs[l][:, i]))
        for l in range(len(subcombs))
        for i in range(subcombs[l].shape[1])
    ]

    return {"S": out, "tot_var": tot_var}


def get_tot_MC(sob):
    """
    Get total sensitivity indices.
    """
    en = [list(map(int, x.split("x"))) for x in sob.columns]
    vars_use = np.unique(np.concatenate(en))
    tot = np.zeros((sob.shape[0], len(vars_use)))
    for i, v in enumerate(vars_use):
        ind = [j for j, x in enumerate(en) if v in x]
        tot[:, i] = sob.iloc[:, ind].sum(axis=1)
    return pd.DataFrame(tot, columns=vars_use)


class sobolMC:
    """
    **Bayesian Adaptive Spline Surfaces - Sobol or subset Sobol sensitivity via Monte Carlo**

    This class is initialized with a fitted BASS model and has methods for the Sobol
      decomposition (via Monte Carlo) and plotting.
    """

    def __init__(self, mod: BassModel):
        self.mod = mod
        return

    def decomp(self, xx=None, subsets=None, mcmc_use=None, ncores=1):
        """
        **BASS Sobol or subset Sobol sensitivity via Monte Carlo**

        This function uses BASS-specific Monte Carlo (for speed) to get the Sobol 
          decomposition for a BASS model (or ensemble of MCMC samples of BASS models).

        :xx: matrix (numpy.ndarray) of input variations. Defaults to the training 
          data for the BASS model.
        :subsets: for subset Sobol, a partition of the inputs represented through a
          list of lists. Defaults to no subsetting.
        :param mcmc_use: which MCMC samples to use (list of integers of length
                            m).  Defaults to all MCMC samples.
        :ncores: not yet used.
        :return: updates the object to have the following for each value of mcmc_use:
          self.S (Sobol indices of all orders), self.T (total indices), and self.tot_var
            (total variance).
        """
        if xx is None:
            xx = self.mod.data.xx
        else:
            xx = normalize(xx, self.mod.data.bounds)
        if mcmc_use is None:
            mcmc_use = np.arange(len(self.mod.samples.nbasis))

        if subsets is None:
            sob = get_sob(mcmc_use=mcmc_use, mod=self.mod, xx=xx, ncores=ncores)
        else:
            sob = get_sob_sub(mcmc_use=mcmc_use, mod=self.mod, xx=xx, subsets=subsets)
        
        ord = []
        en = [list(map(int, x.split("x"))) for x in sob["S"].columns]
        en_len = [len(x) for x in en]
        for i in range(1, max(en_len) + 1):
            tmp = np.array([x for x in en if len(x) == i])
            iord = np.lexsort(tmp.T)
            ord.extend([np.where(np.array(en_len) == i)[0][j] for j in iord])
        self.S = sob["S"].iloc[:, ord]
        self.T = get_tot_MC(sob["S"].iloc[:, ord])
        self.tot_var = sob["tot_var"]

        return
    
    def plot(self):
        """
        **BASS plots of Sobol or subset Sobol sensitivity via Monte Carlo**

        This function plots boxplots of Sobol indices of all orders and total indices.
          The boxplots are from BASS MCMC iterations. If subsets are specified, the 
          labels of effects are for the list of subsets.

        """
        fig = plt.figure()

        fig.add_subplot(1, 2, 1)
        self.S.boxplot(whis=[0, 100])
        plt.ylabel("proportion variance explained")
        plt.xlabel("effect")
        plt.xticks(rotation=90)

        fig.add_subplot(1, 2, 2)
        self.T.boxplot(whis=[0, 100])
        plt.ylabel("total effect")
        plt.xlabel("input")
        plt.xticks(rotation=90)
        
        fig.tight_layout()

        plt.show()
