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
Author: J. Derek Tucker
"""

import itertools
import math
import re
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats

import pyBASS.utils as uf
from pyBASS import BassBasis
from pyBASS.sobol_mc import sobolMC # gets exported from here


class sobolBasis:
    """
    Decomposes the variance of the BASS model into variance due to the main
    effects, two way interactions, and so on, similar to the ANOVA
    decoposition for linear models.

    Uses the Sobol' decomposition, which can be done analytically for
    MARS-type models. This is for the Basis class

    :param mod: BassBasis model

    :return: object with plot method.
    """

    def __init__(self, mod: BassBasis):
        self.mod = mod
        return

    def decomp(self, int_order, prior=None, mcmc_use=None, nind=None, ncores=1):
        """
        Perform Sobol Decomp

        :param int_order: an integer indicating the highest order of
                          interactions to include in the Sobol decomposition.
        :param prior:  a list with the same number of elements as there are
                       inputs to mod. Each element specifies the prior for the
                       particular input.  Each prior is specified as a
                       dictionary with elements (one of "normal", "student",
                       or "uniform"), "trunc" (a vector of dimension 2
                       indicating the lower and  upper truncation bounds,
                       taken to be the data bounds if omitted), and for
                       "normal" or "student" priors, "mean" (scalar mean of
                       the Normal/Student, or a vector of means for a mixture
                       of Normals or Students), "sd" (scalar standard deviation
                       of the Normal/Student, or a vector of standard
                       deviations for a mixture of Normals or Students), "df"
                       (scalar degrees of freedom of the Student, or a vector
                       of degrees of freedom for a mixture of Students), and
                       "weights" (a vector of weights that sum to one for the
                       mixture components, or the scalar 1).  If unspecified,
                       a uniform is assumed with the same bounds as are
                       represented in the input to mod.
        :param mcmc_use: an integer indicating which MCMC iteration to use for
                         sensitivity analysis. Defaults to the last iteration.
        :param nind: number of Sobol indices to keep
                     (will keep the largest nind).
        :param ncores: number of cores to use (default = 1)
        """
        self.int_order = int_order
        if mcmc_use is None:
            self.mcmc_use = self.mod.bm_list[0].nstore - 1
        else:
            self.mcmc_use = mcmc_use
        self.nind = nind
        self.ncores = ncores

        bassDat = self.mod.bm_list[0].data

        if prior is None:
            self.prior = []
        else:
            self.prior = prior

        p = bassDat.p

        if len(self.prior) < p:
            for i in range(len(self.prior), p):
                tmp = {"dist": "uniform", "trunc": None}
                self.prior.append(tmp)

        for i in range(len(self.prior)):
            if self.prior[i]["trunc"] is None:
                self.prior[i]["trunc"] = np.array([0, 1])
            else:
                self.prior[i]["trunc"] = uf.normalize(
                    self.prior[i]["trunc"], bassDat.bounds[:, i]
                )

            if (
                self.prior[i]["dist"] == "normal"
                or self.prior[i]["dist"] == "student"
            ):
                self.prior[i]["mean"] = uf.normalize(
                    self.prior[i]["mean"], bassDat.bounds[:, i]
                )
                self.prior[i]["sd"] = prior[i]["sd"] / (
                    bassDat.bounds[1, i] - bassDat.bounds[0, i]
                )
                if self.prior[i]["dist"] == "normal":
                    self.prior[i]["z"] = stats.norm.pdf(
                        (self.prior[i]["trunc"][1] - self.prior[i]["mean"])
                        / self.prior[i]["sd"]
                    ) - stats.norm.pdf(
                        (self.prior[i]["trunc"][0] - self.prior[i]["mean"])
                        / self.prior[i]["sd"]
                    )
                else:
                    self.prior[i]["z"] = stats.t.pdf(
                        (self.prior[i]["trunc"][1] - self.prior[i]["mean"])
                        / self.prior[i]["sd"],
                        self.prior[i]["df"],
                    ) - stats.t.pdf(
                        (self.prior[i]["trunc"][0] - self.prior[i]["mean"])
                        / self.prior[i]["sd"],
                        self.prior[i]["df"],
                    )

                cc = (self.prior[i]["weights"] * self.prior[i]["z"]).sum()
                self.prior[i]["weights"] = self.prior[i]["weights"] / cc

        pc_mod = self.mod.bm_list
        pcs = self.mod.basis

        tic = time.perf_counter()
        print("Start\n")

        if int_order > p:
            self.int_order = p
            print(
                "int_order > number of inputs, change to int_order = number of input\n"
            )

        u_list = [
            list(itertools.combinations(range(0, p), x))
            for x in range(1, int_order + 1)
        ]
        ncombs_vec = [len(x) for x in u_list]
        ncombs = sum(ncombs_vec)
        nxfunc = pcs.shape[0]

        n_pc = self.mod.nbasis

        w0 = np.zeros(n_pc)
        for i in range(n_pc):
            w0[i] = self.get_f0(pc_mod, i).item()

        f0r2 = (pcs @ w0) ** 2

        tmp = [pc_mod[x].samples.nbasis[self.mcmc_use] for x in range(n_pc)]
        max_nbasis = max(tmp)

        C1Basis_array = np.zeros((n_pc, p, max_nbasis))
        for i in range(n_pc):
            nb = pc_mod[i].samples.nbasis[self.mcmc_use]
            mcmc_mod_usei = pc_mod[i].model_lookup[self.mcmc_use]
            for j in range(p):
                for k in range(nb):
                    C1Basis_array[i, j, k] = self.C1Basis(
                        pc_mod, j, k, i, mcmc_mod_usei
                    )

        u_list1 = []
        for i in range(int_order):
            u_list1.extend(u_list[i])

        toc = time.perf_counter()
        print("Integrating: %0.2fs\n" % (toc - tic))

        u_list_temp = u_list1
        u_list_temp.insert(0, list(np.arange(0, p)))

        if ncores > 1:
            # @todo write parallel version
            NameError("Parallel not Implemented\n")
        else:
            ints1_temp = [
                self.func_hat(x, pc_mod, pcs, mcmc_use, f0r2, C1Basis_array)
                for x in u_list_temp
            ]

        V_tot = ints1_temp[0]
        ints1 = ints1_temp[1:]

        ints = []
        ints.append(np.zeros((ints1[0].shape[0], len(u_list[0]))))
        for i in range(len(u_list[0])):
            ints[0][:, i] = ints1[i]

        if int_order > 1:
            for i in range(2, int_order + 1):
                idx = np.sum(ncombs_vec[0 : (i - 1)]) + np.arange(
                    0, len(u_list[i - 1])
                )
                ints.append(np.zeros((ints1[0].shape[0], idx.shape[0])))
                cnt = 0
                for j in idx:
                    ints[i - 1][:, cnt] = ints1[j]
                    cnt += 1

        sob = []
        sob.append(ints[0])
        toc = time.perf_counter()
        print("Shuffling: %0.2fs\n" % (toc - tic))

        if len(u_list) > 1:
            for i in range(1, len(u_list)):
                sob.append(np.zeros((nxfunc, ints[i].shape[1])))
                for j in range(len(u_list[i])):
                    cc = np.zeros(nxfunc)
                    for k in range(i):
                        ind = [
                            np.all(np.in1d(x, u_list[i][j])) for x in u_list[k]
                        ]
                        cc += (-1) ** (i - k) * np.sum(ints[k][:, ind], axis=1)
                    sob[i][:, j] = ints[i][:, j] + cc

        if nind is None:
            nind = ncombs

        sob_comb_var = np.concatenate(sob, axis=1)

        vv = np.mean(sob_comb_var, axis=0)
        ord = vv.argsort()[::-1]
        cutoff = vv[ord[nind - 1]]
        if nind > ord.shape[0]:
            cutoff = vv.min()

        use = np.sort(np.where(vv >= cutoff)[0])

        V_other = V_tot - np.sum(sob_comb_var[:, use], axis=1)

        use = np.append(use, ncombs)

        sob_comb_var = np.hstack((sob_comb_var, V_other[:, np.newaxis])).T
        sob_comb = sob_comb_var / V_tot

        sob_comb_var = sob_comb_var[use, :]
        sob_comb = sob_comb[use, :]

        # Calculate "Total Sobol' Index"
        sob_comb_tot = np.zeros((p, nxfunc))
        idx = 0
        for i in range(int_order):
            for j in range(len(u_list[i])):
                sob_comb_tot[u_list[i][j], :] += sob_comb_var[idx]
                idx += 1

        names_ind1 = []
        for i in range(len(u_list)):
            for j in range(len(u_list[i])):
                tmp = u_list[i][j]
                tmp1 = [x + 1 for x in tmp]
                tmp1 = re.findall(r"\d+", str(tmp1))
                if len(tmp1) == 1:
                    names_ind1.append(tmp1[0])
                else:
                    separator = "x"
                    names_ind1.append(separator.join(tmp1))

        names_ind1.append("other")
        names_ind2 = [names_ind1[x] for x in use]

        toc = time.perf_counter()
        print("Finish: %0.2fs\n" % (toc - tic))

        self.S = sob_comb
        self.S_var = sob_comb_var
        self.T_var = sob_comb_tot
        self.Var_tot = V_tot
        self.names_ind = names_ind2
        self.xx = np.linspace(0, 1, nxfunc)

        return

    def plot(
        self,
        int_order=1,
        total_sobol=True,
        text=False,
        labels=[],
        col="Paired",
        time=[],
    ):
        if len(time) == 0:
            time = self.xx

        p = np.shape(self.mod.xx)[1]
        ncomb = np.sum([math.comb(p, k) for k in range(1, int_order + 1)])

        if len(labels) == 0:
            labels = self.names_ind[:ncomb] + [self.names_ind[-1]]

        map = cm.Paired(np.linspace(0, 1, 12))
        map = np.resize(map, (len(labels), 4))
        rgb = np.ones((map.shape[0] + 1, 4))
        rgb[0 : map.shape[0], :] = map
        rgb[-1, 0:3] = np.array([153, 153, 153]) / 255

        ord = time.argsort()
        x_mean = np.vstack([
            self.S[:ncomb, :],
            np.sum(self.S[ncomb:, :], axis=0, keepdims=True),
        ])
        sens = np.cumsum(x_mean, axis=0).T
        fig, axs = plt.subplots(1, 2 + total_sobol)
        cnt = 0
        for i in range(ncomb + 1):
            x2 = np.concatenate((time[ord], np.flip(time[ord])))
            if i == 0:
                inBetween = np.concatenate((
                    np.zeros(time[ord].shape[0]),
                    np.flip(sens[ord, i]),
                ))
            else:
                inBetween = np.concatenate((
                    sens[ord, i - 1],
                    np.flip(sens[ord, i]),
                ))
            if (cnt % rgb.shape[0] + 1) == 0:
                cnt = 0

            axs[0].fill(x2, inBetween, color=rgb[cnt, :])
            cnt += 1

        axs[0].set(
            xlabel="x",
            ylabel="proportion variance",
            title="Sensitivity",
            ylim=[0, 1],
            xlim=[time.min(), time.max()],
        )

        if text:
            lab_x = np.argmax(x_mean, axis=1)
            cs = np.zeros((sens.shape[1] + 1, sens.shape[0]))
            cs[1:, :] = np.cumsum(x_mean, axis=0)
            cs_diff = np.zeros((x_mean.shape[0], x_mean.shape[1]))
            for i in range(x_mean.shape[1]):
                cs_diff[:, i] = np.diff(
                    np.cumsum(np.concatenate((0, x_mean[:, 0])))
                )
            tmp = np.concatenate((np.arange(0, lab_x.shape[0]), lab_x))
            ind = np.ravel_multi_index(
                np.concatenate((tmp[:, 0], tmp[:, 1])), dims=cs.shape, order="F"
            )
            ind1 = np.ravel_multi_index(
                np.concatenate((tmp[:, 0], tmp[:, 1])),
                dims=cs_diff.shape,
                order="F",
            )
            cs_diff2 = cs_diff / 2
            plt.text(time[lab_x], cs[ind] + cs_diff2[ind1], self.names_ind)

        x_mean_var = np.vstack([
            self.S_var[:ncomb, :],
            np.sum(self.S_var[ncomb:, :], axis=0, keepdims=True),
        ])
        sens_var = np.cumsum(x_mean_var, axis=0).T
        cnt = 0
        for i in range(ncomb + 1):
            x2 = np.concatenate((time[ord], np.flip(time[ord])))
            if i == 0:
                inBetween = np.concatenate((
                    np.zeros(time[ord].shape[0]),
                    np.flip(sens_var[ord, i]),
                ))
            else:
                inBetween = np.concatenate((
                    sens_var[ord, i - 1],
                    np.flip(sens_var[ord, i]),
                ))
            if (cnt % rgb.shape[0] + 1) == 0:
                cnt = 0

            axs[1].fill(x2, inBetween, color=rgb[cnt, :])
            cnt += 1

        axs[1].set(
            xlabel="x",
            ylabel="variance",
            title="Variance Decomposition",
            xlim=[time.min(), time.max()],
            ylim=[0, inBetween.max() + 3],
        )

        if not text:
            axs[1].legend(labels, loc="upper left")

        if total_sobol:
            x_mean_tot = self.T_var
            sens_tot = np.cumsum(x_mean_tot, axis=0).T
            cnt = 0
            for i in range(p):
                x2 = np.concatenate((time[ord], np.flip(time[ord])))
                if i == 0:
                    inBetween = np.concatenate((
                        np.zeros(time[ord].shape[0]),
                        np.flip(sens_tot[ord, i]),
                    ))
                else:
                    inBetween = np.concatenate((
                        sens_tot[ord, i - 1],
                        np.flip(sens_tot[ord, i]),
                    ))
                if (cnt % rgb.shape[0] + 1) == 0:
                    cnt = 0

                axs[2].fill(x2, inBetween, color=rgb[cnt, :])
                cnt += 1

            axs[2].set(
                xlabel="x",
                ylabel="total variance",
                title="Total Sobol'",
                xlim=[time.min(), time.max()],
                ylim=[0, inBetween.max() + 3],
            )

        fig.tight_layout()
        return

    def get_f0(self, pc_mod, pc):
        mcmc_mod_use = pc_mod[pc].model_lookup[self.mcmc_use]
        out = pc_mod[pc].samples.beta[self.mcmc_use, 0]
        if pc_mod[pc].samples.nbasis[self.mcmc_use] > 0:
            for m in range(pc_mod[pc].samples.nbasis[self.mcmc_use]):
                out1 = pc_mod[pc].samples.beta[self.mcmc_use, 1 + m]
                for ell in range(pc_mod[pc].data.p):
                    out1 = out1 * self.C1Basis(pc_mod, ell, m, pc, mcmc_mod_use)
                out += out1
        return out

    def C1Basis(self, pc_mod, ell, m, pc, mcmc_mod_use):
        n_int = pc_mod[pc].samples.n_int[mcmc_mod_use, m]
        int_use_l = np.where(
            pc_mod[pc].samples.vs[mcmc_mod_use, m, :][:n_int] == ell
        )[0]

        if len(int_use_l) == 0:
            out = 1
            return out

        s = pc_mod[pc].samples.signs[mcmc_mod_use, m, int_use_l]
        t = pc_mod[pc].samples.knots[mcmc_mod_use, m, int_use_l]
        q = 1

        if s == 0:
            out = 0
            return out

        cc = uf.const(s, t)

        if s == 1:
            a = np.maximum(self.prior[ell]["trunc"][0], t)
            b = self.prior[ell]["trunc"][1]
            if b < t:
                out = 0
                return out
            out = self.intabq1(self.prior[ell], a, b, t, q) / cc
        else:
            a = self.prior[ell]["trunc"][0]
            b = np.minimum(self.prior[ell]["trunc"][1], t)
            if t < a:
                out = 0
                return out
            out = self.intabq1(self.prior[ell], a, b, t, q) * (-1) ** q / cc
        if isinstance(out, int) or isinstance(out, float):
            return out
        elif isinstance(out, np.ndarray):
            return out.item()
        else:
            raise TypeError("out is unexpected type: " + str(type(out)))

    def intabq1(self, prior, a, b, t, q):
        if prior["dist"] == "normal":
            if q != 1:
                NameError("degree other than 1 not supported for normal priors")

            out = 0
            for k in range(len(prior["weights"])):
                zk = stats.norm.pdf(
                    b, prior["mean"][k], prior["sd"][k]
                ) - stats.norm.pdf(a, prior["mean"][k], prior["sd"][k])
                ast = (a - prior["mean"][k]) / prior["sd"][k]
                bst = (b - prior["mean"][k]) / prior["sd"][k]
                dnb = stats.norm.cdf(bst)
                dna = stats.norm.cdf(ast)
                tnorm_mean_zk = prior["mean"][k] * zk - prior["sd"][k] * (
                    dnb - dna
                )
                out += prior["weights"][k] * (tnorm_mean_zk - t * zk)

        if prior["dist"] == "student":
            if q != 1:
                NameError("degree other than 1 not supported for normal priors")

            out = 0
            for k in range(len(prior["weights"])):
                int = self.intx1Student(
                    b, prior["mean"][k], prior["sd"][k], prior["df"][k], t
                ) - self.intx1Student(
                    a, prior["mean"][k], prior["sd"][k], prior["df"][k], t
                )
                out += prior["weights"][k] * int

        if prior["dist"] == "uniform":
            out = (
                1
                / (q + 1)
                * ((b - t) ** (q + 1) - (a - t) ** (q + 1))
                * 1
                / (prior["trunc"][1] - prior["trunc"][0])
            )

        return out

    def intx1Student(self, x, m, s, v, t):
        temp = (s**2 * v) / (m**2 + s**2 * v - 2 * m * x + x**2)
        out = -(
            (v / (v + (m - x) ** 2 / s ^ 2)) ** (v / 2)
            * np.sqrt(temp)
            * np.sqrt(1 / temp)
            * (
                s**2 * v * (np.sqrt(1 / temp) - (1 / temp) ** (v / 2))
                + (t - m)
                * (-1 + v)
                * (-m + x)
                * (1 / temp) ** (v / 2)
                * self.robust2f1(
                    1 / 2, (1 + v) / 2, 3 / 2, -((m - x) ** 2) / (s**2 * v)
                )
            )
        ) / (s * (-1 + v) * np.sqrt(v) * sp.special.beta(v / 2, 1 / 2))

        return out

    def robust2f1(self, a, b, c, x):
        if np.abs(x) < 1:
            z = sp.special.hyp2f1(a, b, c, np.array([0, x]))
            out = z[-1]
        else:
            z = sp.special.hyp2f1(a, c - b, c, 0)
            out = z[-1]

        return out

    def func_hat(self, u, pc_mod, pcs, mcmc_use, f0r2, C1Basis_array):
        res = np.zeros(pcs.shape[0])
        n_pc = len(pc_mod)
        for i in range(n_pc):
            res += pcs[:, i] ** 2 * self.Ccross(pc_mod, i, i, u, C1Basis_array)

            if (i + 1) < n_pc:
                for j in range(i + 1, n_pc):
                    res = res + 2 * pcs[:, i] * pcs[:, j] * self.Ccross(
                        pc_mod, i, j, u, C1Basis_array
                    )

        out = res - f0r2

        return out

    def Ccross(self, pc_mod, i, j, u, C1Basis_array):
        p = pc_mod[0].data.p
        mcmc_mod_usei = pc_mod[i].model_lookup[self.mcmc_use]
        mcmc_mod_usej = pc_mod[j].model_lookup[self.mcmc_use]

        Mi = pc_mod[i].samples.nbasis[self.mcmc_use]
        Mj = pc_mod[j].samples.nbasis[self.mcmc_use]

        a0i = pc_mod[i].samples.beta[self.mcmc_use, 0]
        a0j = pc_mod[j].samples.beta[self.mcmc_use, 0]
        f0i = self.get_f0(pc_mod, i)
        f0j = self.get_f0(pc_mod, j)

        out = a0i * a0j + a0i * (f0j - a0j) + a0j * (f0i - a0i)

        if Mi > 0 and Mj > 0:
            ai = pc_mod[i].samples.beta[self.mcmc_use, 1 : (Mi + 1)]
            aj = pc_mod[j].samples.beta[self.mcmc_use, 1 : (Mj + 1)]

        for mi in range(Mi):
            for mj in range(Mj):
                temp1 = ai[mi] * aj[mj]
                temp2 = 1
                temp3 = 1
                idx = np.arange(0, p)
                idx2 = u
                idx = np.delete(idx, idx2)

                for ell in idx:
                    temp2 = (
                        temp2
                        * C1Basis_array[i, ell, mi]
                        * C1Basis_array[j, ell, mj]
                    )

                for ell in idx2:
                    temp3 = temp3 * self.C2Basis(
                        pc_mod, ell, mi, mj, i, j, mcmc_mod_usei, mcmc_mod_usej
                    )

                out += temp1 * temp2 * temp3

        return out

    def C2Basis(
        self, pc_mod, ell, m1, m2, pc1, pc2, mcmc_mod_use1, mcmc_mod_use2
    ):
        if ell < pc_mod[pc1].data.p:
            n_int1 = pc_mod[pc1].samples.n_int[mcmc_mod_use1, m1]
            int_use_l1 = np.where(
                pc_mod[pc1].samples.vs[mcmc_mod_use1, m1, :][:n_int1] == ell
            )[0]
            n_int2 = pc_mod[pc2].samples.n_int[mcmc_mod_use2, m2]
            int_use_l2 = np.where(
                pc_mod[pc2].samples.vs[mcmc_mod_use2, m2, :][:n_int2] == ell
            )[0]

            if int_use_l1.size == 0 and int_use_l2.size == 0:
                out = 1
                return out

            if int_use_l1.size == 0:
                out = self.C1Basis(pc_mod, ell, m2, pc2, mcmc_mod_use2)
                return out

            if int_use_l2.size == 0:
                out = self.C1Basis(pc_mod, ell, m1, pc1, mcmc_mod_use1)
                return out

            q = 1
            s1 = pc_mod[pc1].samples.signs[mcmc_mod_use1, m1, int_use_l1]
            s2 = pc_mod[pc2].samples.signs[mcmc_mod_use2, m2, int_use_l2]
            t1 = pc_mod[pc1].samples.knots[mcmc_mod_use1, m1, int_use_l1]
            t2 = pc_mod[pc2].samples.knots[mcmc_mod_use2, m2, int_use_l2]

            if t2 < t1:
                t1, t2 = t2, t1
                s1, s2 = s2, s1

            out = self.C22Basis(self.prior[ell], t1, t2, s1, s2, q)

        return out

    def C22Basis(self, prior, t1, t2, s1, s2, q):
        cc = uf.const(np.array([s1, s2]), np.array([t1, t2]))
        out = 0
        if (s1 * s2) == 0:
            out = 0
            return out

        if s1 == 1:
            if s2 == 1:
                out = self.intabq2(prior, t2, 1, t1, t2, q) / cc
                return out
            else:
                out = self.intabq2(prior, t1, t2, t1, t2, q) * (-1) ** q / cc
                return out
        else:
            if s2 == 1:
                out = 0
                return out
            else:
                out = self.intabq2(prior, 0, t1, t1, t2, q) / cc
                return out

        return out

    def intabq2(self, prior, a, b, t1, t2, q):
        if prior["dist"] == "normal":
            if q != 1:
                NameError("degree other than 1 not supported for normal priors")

            out = 0
            for k in range(len(prior["weights"])):
                zk = stats.norm.pdf(
                    b, prior["mean"][k], prior["sd"][k]
                ) - stats.norm.pdf(a, prior["mean"][k], prior["sd"][k])
                if zk < np.finfo(float).eps:
                    continue
                ast = (a - prior["mean"][k]) / prior["sd"][k]
                bst = (b - prior["mean"][k]) / prior["sd"][k]
                dnb = stats.norm.cdf(bst)
                dna = stats.norm.cdf(ast)
                tnorm_mean_zk = prior["mean"][k] * zk - prior["sd"][k] * (
                    dnb - dna
                )
                tnorm_var_zk = (
                    zk
                    * prior["sd"][k] ** 2
                    * (
                        1
                        + (ast * dna - bst * dnb) / zk
                        - ((dna - dnb) / zk) ** 2
                    )
                    + tnorm_mean_zk**2 / zk
                )
                out += prior["weights"][k] * (
                    tnorm_var_zk - (t1 + t2) * tnorm_mean_zk + t1 * t2 * zk
                )
                if out < 0 and np.abs(out) < 1e-12:
                    out = 0

        if prior["dist"] == "student":
            if q != 1:
                NameError("degree other than 1 not supported for normal priors")

            out = 0
            for k in range(len(prior["weights"])):
                int = self.intx2Student(
                    b, prior["mean"][k], prior["sd"][k], prior["df"][k], t1, t2
                ) - self.intx2Student(
                    a, prior["mean"][k], prior["sd"][k], prior["df"][k], t1, t2
                )
                out += prior["weights"][k] * int

        if prior["dist"] == "uniform":
            out = (
                (
                    np.sum(
                        self.pCoef(np.arange(0, q + 1), q)
                        * (b - t1) ** (q - np.arange(0, q + 1))
                        * (b - t2) ** (q + 1 + np.arange(0, q + 1))
                    )
                    - np.sum(
                        self.pCoef(np.arange(0, q + 1), q)
                        * (a - t1) ** (q - np.arange(0, q + 1))
                        * (a - t2) ** (q + 1 + np.arange(0, q + 1))
                    )
                )
                * 1
                / (prior["trunc"][1] - prior["trunc"][0])
            )

        return out

    def intx2Student(self, x, m, s, v, t1, t2):
        temp = (s**2 * v) / (m**2 + s**2 * v - 2 * m * x + x**2)
        out = (
            (v / (v + (m - x) ** 2 / s**2)) ** (v / 2)
            * np.sqrt(temp)
            * np.sqrt(1 / temp)
            * (
                -3
                * (-t1 - t2 + 2 * m)
                * s**2
                * v
                * (np.sqrt(1 / temp) - (1 / temp) ** (v / 2))
                + 3
                * (-t1 + m)
                * (-t2 + m)
                * (-1 + v)
                * (-m + x)
                * (1 / temp) ** (v / 2)
                * self.robust2f1(
                    1 / 2, (1 + v) / 2, 3 / 2, -((m - x) ** 2) / (s**2 * v)
                )
                + (-1 + v)
                * (-m + x) ** 3
                * (1 / temp) ** (v / 2)
                * self.robust2f1(
                    3 / 2, (1 + v) / 2, 5 / 2, -((m - x) ** 2) / (s**2 * v)
                )
            )
        ) / (3 * s * (-1 + v) * np.sqrt(v) * sp.special.beta(v / 2, 1 / 2))

        return out

    def pCoef(self, i, q):
        out = (
            sp.special.factorial(q) ** 2
            * (-1) ** i
            / (sp.special.factorial(q - i) * sp.special.factorial(q + 1 + i))
        )
        return out
