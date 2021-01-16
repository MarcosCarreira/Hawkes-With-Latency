# %% Imports


import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from functools import partial
from numba import jit, njit
from numba import types
from numba.typed import Dict

from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc

# %% Define the time_function


def time_funclat(f, support, lat=0, steps=100,
              inter_mode=TimeFunction.InterLinear):
    t_values = np.linspace(0, support, steps + 1)
    y_values = f(t_values)
    if lat > 0:
        t_values_lat = np.linspace(0, lat, num=steps, endpoint=False)
        y_values_lat = np.zeros(steps)
        t_values_shifted = t_values + lat
        t_all = np.concatenate((t_values_lat, t_values_shifted))
        y_all = np.concatenate((y_values_lat, y_values))
    else:
        t_all = t_values.view()
        y_all = y_values.view()
    return TimeFunction(values=(t_all, y_all),
                        border_type=TimeFunction.Border0,
                        inter_mode=inter_mode)

# %% Log-likelihood without latency


@njit
def δts0(ts):
    return ts[-1] - ts[:-1]


@njit
def ll(θ, ts, Δts, δts):
    λ0 = θ[0]
    α = θ[1]
    β = θ[2]
    tn = ts[-1]
    ebdts = np.exp(-β * Δts)
    ri = np.zeros(len(ts))
    for i in range(1, len(ts)):
        ri[i] = ebdts[i - 1] * (1 + ri[i - 1])
    s1 = np.sum(1 - np.exp(-β * δts))
    s2 = np.sum(np.log(λ0 + α * ri))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


def findθ(tss, bounds, constraints, x0):
    results = []
    for path in tss:
        ts = path[0]
        Δts = np.diff(ts)
        δts = δts0(ts)
        optim0 = partial(ll, ts=ts, Δts=Δts, δts=δts)
        res0 = minimize(optim0, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        results = results + [res0.x]
    return np.array(results)


# %% Log-likelihood with latency


@njit
def findtlat(ts, τ, i):
    j = np.searchsorted(ts[i + 1:] - τ - ts[i], 0) + i + 1
    if j <= len(ts) - 1:
        return [j, ts[j] - τ - ts[i]]
    else:
        return [j, 0.]


# @njit
def τtsi(ts, τ):
    # dtsis = dict()
    dtsis = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:],
    )
    for i in range(0, len(ts) - 1):
        k, v = findtlat(ts, τ, i)
        if v > 0:
            if int(k) not in dtsis:
                # dtsis[int(k)] = [v]
                dtsis[int(k)] = np.array([v])
            else:
                # dtsis[k].extend([v])
                dtsis[int(k)] = np.append(dtsis[int(k)], v)
    for i in range(0, len(ts)):
        if i not in dtsis:
            dtsis[i] = np.array([np.inf])
    return dtsis
    # return [np.array(dtsis.get(i, np.inf)) for i in range(1, len(ts))]


def heav(ts, τ):
    return np.heaviside(ts[-1] - τ - ts[:-1], 0)


def δtsτ(ts, τ):
    return (ts[-1] - τ - ts[:-1]) * heav(ts, τ)


def Δtst(ts):
    return np.diff(ts)


@njit
def lllat(θ, ts, Δts, δts, τts):
    λ0 = θ[0]
    α = θ[1]
    β = θ[2]
    tn = ts[-1]
    ebdts = np.exp(-β * Δts)
    ri = np.zeros(len(ts))
    for i in range(1, len(ts)):
        # ri[i] = ebdts[i - 1] * ri[i - 1] + np.sum(np.exp(-β * τts[i - 1]))
        ri[i] = ebdts[i - 1] * ri[i - 1] + np.sum(np.exp(-β * τts[i]))
    s1 = np.sum(1 - np.exp(-β * δts))
    s2 = np.sum(np.log(λ0 + α * ri))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


def findθlat(tss, bounds, constraints, x0, τ=0):
    results = []
    for path in tss:
        ts = path[0]
        Δts = Δtst(ts)
        δts = δtsτ(ts, τ)
        τts = τtsi(ts, τ)
        optim0 = partial(lllat, ts=ts, Δts=Δts, δts=δts, τts=τts)
        res0 = minimize(optim0, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        results = results + [res0.x]
    return np.array(results)


# %% Multidimensional with latency


def lenms(ts):
    return np.array([len(ts[m]) for m in range(len(ts))])


def last_tn(ts):
    return np.array([ts[n][-1] for n in range(len(ts))])


def max_last_tn(ts):
    return max([ts[n][-1] for n in range(len(ts))])


def ΔtsM(ts):
    mxn = max(lenms(ts))
    M = len(ts)
    Δtsa = np.zeros((M, mxn))
    for m in range(M):
        new_row = np.diff(ts[m], prepend=0)
        Δtsa[m] = np.pad(new_row, (0, mxn - len(new_row)),
                         'constant', constant_values=np.inf)
    return Δtsa


def δtsτMM(ts, τ):
    mxn = max(lenms(ts))
    M = len(ts)
    δtsτa = np.zeros((M, M, mxn))
    for m in range(M):
        for n in range(M):
            new_row = np.abs((ts[m][-1] - τ - ts[n]) * 
                              np.heaviside(ts[m][-1] - τ - ts[n], 0))
            δtsτa[m, n] = np.pad(new_row, (mxn - len(new_row), 0),
                                 'constant', constant_values=0)
    return δtsτa


def τtsiMM(ts, τ):
    M = len(ts)
    mxn = max(lenms(ts))
    dtsis = dict()
    for m in range(M):
        for n in range(M):
            for i in range(0, len(ts[n])):
                k = np.searchsorted(ts[m] - τ - ts[n][i], 0)
                if k <= len(ts[m]) - 1:
                    key = str(m) + '-' + str(n) + '-' + str(k)
                    v = ts[m][k] - τ - ts[n][i]
                    if key not in dtsis:
                        dtsis[key] = [v]
                    else:
                        dtsis[key].extend([v])
    mxτ = max([len(x) for x in dtsis.values()])
    τtsia = np.zeros((M, M, mxn, mxτ))
    for m in range(M):
        for n in range(M):
            for i in range(mxn):
                key = str(m) + '-' + str(n) + '-' + str(i)
                new_row = dtsis.get(key, np.array([np.inf]))
                τtsia[m, n, i] = np.pad(new_row, (mxτ - len(new_row), 0),
                                        'constant', constant_values=np.inf)
    return τtsia


@njit
def lllatm(θ, m, nts, maxtnm, Δts, δts, τts):
    M = len(nts)
    nm = nts[m]
    λ0m = θ[:M]
    αmn = θ[M:M + M**2].reshape((M, M))
    βmn = θ[M + M**2:].reshape((M, M))
    s1 = np.sum(np.array(
        [np.sum((αmn[m, n] / βmn[m, n]) * (1 - np.exp(-βmn[m, n] * δts[m, n])))
         for n in range(M)]))
    rin = np.zeros((M, nm))
    for n in range(M):
        ebdts = np.exp(-βmn[m, n] * Δts[m])
        for i in range(1, nm):
            rin[n][i] = ebdts[i] * rin[n][i - 1] + np.sum(np.exp(-βmn[m, n] * τts[m, n, i]))
        rin[n] = αmn[m, n] * rin[n]
    s2 = np.sum(np.log(λ0m[m] + np.sum(rin, axis=0)))
    llms = -(maxtnm * (1 - λ0m[m]) - s1 + s2) #
    return llms


def slicex(x, M, m):
    λ0 = x[:M][m]
    αm = x[M:M + M**2].reshape((M, M))[m]
    βm = x[M + M**2:].reshape((M, M))[m]
    return np.array([λ0] + αm.tolist() + βm.tolist())


def findθlatm(ts, bounds, x0, τ=0, method='L-BFGS-B', print_m=False):
    nts = lenms(ts)
    M = len(nts)
    maxtnm = max_last_tn(ts)
    Δts = ΔtsM(ts)
    δts = δtsτMM(ts, τ)
    τts = τtsiMM(ts, τ)
    results = []
    for m in range(M):
        optim0 = partial(lllatm, m=m, nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
        res0 = minimize(optim0, x0, method=method, bounds=bounds)
        if print_m:
            print('m = ' + str(m))
            print(res0.message)
            print(res0.fun)
        results = results + [slicex(res0.x, M, m)]
    return np.array(results)


def findθlatms(tss, bounds, x0, τ=0, method='L-BFGS-B', print_m=False):
    results = []
    for ts in tss:
        results = results + [findθlatm(ts, bounds, x0, τ, method, print_m)]
    return results


def findθlatmde(ts, bounds, τ=0, maxiter=200, popsize=20, tol=1e-2,
        print_m=False, disp=False):
    nts = lenms(ts)
    M = len(nts)
    maxtnm = max_last_tn(ts)
    Δts = ΔtsM(ts)
    δts = δtsτMM(ts, τ)
    τts = τtsiMM(ts, τ)
    results = []
    for m in range(M):
        optim0 = partial(lllatm, m=m, nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
        res0 = differential_evolution(optim0, bounds=bounds,
            maxiter=maxiter, popsize=popsize, tol=tol, disp=disp)
        if print_m:
            print('m = ' + str(m))
            print(res0.message)
            print(res0.fun)
        results = results + [slicex(res0.x, M, m)]
    return np.array(results)


def findθlatmsde(tss, bounds, x0, τ=0, maxiter=200, popsize=20, tol=1e-2,
        print_m=False, disp=False):
    results = []
    for ts in tss:
        results = results + [findθlatmde(ts, bounds, τ,
            maxiter, popsize, tol, print_m, disp)]
    return results

# %% Multidimensional with latency - all


@njit
def lllatall(θ, nts, maxtnm, Δts, δts, τts):
    M = len(nts)
    λ0m = θ[:M]
    αmn = θ[M:M + M**2].reshape((M, M))
    βmn = θ[M + M**2:].reshape((M, M))
    llall = 0.
    for m in range(M):
        nm = nts[m]
        s1 = np.sum(np.array(
            [np.sum((αmn[m, n] / βmn[m, n]) * (1 - np.exp(-βmn[m, n] * δts[m, n])))
             for n in range(M)]))
        rin = np.zeros((M, nm))
        for n in range(M):
            ebdts = np.exp(-βmn[m, n] * Δts[m])
            for i in range(1, nm):
                rin[n][i] = ebdts[i] * rin[n][i - 1] + np.sum(np.exp(-βmn[m, n] * τts[m, n, i]))
            rin[n] = αmn[m, n] * rin[n]
        s2 = np.sum(np.log(λ0m[m] + np.sum(rin, axis=0)))
        llms = -(maxtnm * (1 - λ0m[m]) - s1 + s2)
        llall = llall + llms
    return llall


def findθlatall(ts, bounds, x0, τ=0, method='L-BFGS-B', print_msg=False):
    nts = lenms(ts)
    maxtnm = max_last_tn(ts)
    Δts = ΔtsM(ts)
    δts = δtsτMM(ts, τ)
    τts = τtsiMM(ts, τ)
    optim0 = partial(lllatall, nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
    res0 = minimize(optim0, x0, method=method, bounds=bounds)
    if print_msg:
        print(res0.message)
        print(res0.fun)
    results = res0.x
    return results


# def findθlatdeall(ts, bounds, τ=0, maxiter=200, popsize=20, tol=1e-2,
#         print_msg=False, disp=False):
#     nts = lenms(ts)
#     maxtnm = max_last_tn(ts)
#     Δts = ΔtsM(ts)
#     δts = δtsτMM(ts, τ)
#     τts = τtsiMM(ts, τ)
#     optim0 = partial(lllatall, nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
#     res0 = differential_evolution(optim0, bounds=bounds,
#             maxiter=maxiter, popsize=popsize, tol=tol, disp=disp)
#     if print_msg:
#         print(res0.message)
#         print(res0.fun)
#     results = res0.x
#     return results

# %% Multidimensional with latency - blocks

@njit
def exp_block(b):
    return np.array([
        [b[0], b[0], b[1], b[1]],
        [b[0], b[0], b[1], b[1]],
        [b[2], b[2], b[3], b[3]],
        [b[2], b[2], b[3], b[3]]])


@njit
def exp_diag(b):
    return np.array([
        [b[0], b[1], b[2], b[3]],
        [b[1], b[0], b[3], b[2]],
        [b[4], b[5], b[6], b[7]],
        [b[5], b[4], b[7], b[6]]])


@njit
def lllatblock(θ, bsz, mrng, nts, maxtnm, Δts, δts, τts):
    M = len(nts)
    λ0m = θ[:M]
    αmn = exp_diag(θ[M:M + 2 * bsz])
    βmn = exp_block(θ[M + 2 * bsz:])
    llall = 0.
    for m in mrng:
        nm = nts[m]
        s1 = np.sum(np.array(
            [np.sum((αmn[m, n] / βmn[m, n]) * (1 - np.exp(-βmn[m, n] * δts[m, n])))
             for n in range(M)]))
        rin = np.zeros((M, nm))
        for n in range(M):
            ebdts = np.exp(-βmn[m, n] * Δts[m])
            for i in range(1, nm):
                rin[n][i] = ebdts[i] * rin[n][i - 1] + np.sum(np.exp(-βmn[m, n] * τts[m, n, i]))
            rin[n] = αmn[m, n] * rin[n]
        s2 = np.sum(np.log(λ0m[m] + np.sum(rin, axis=0)))
        llms = -(maxtnm * (1 - λ0m[m]) - s1 + s2)
        llall = llall + llms
    return llall


def slicexblock(x, M, bsz, m):
    λ0 = x[:M][m]
    αm = exp_diag(x[M:M + 2 * bsz])[m]
    βm = exp_block(x[M + 2 * bsz:])[m]
    return np.array([λ0] + αm.tolist() + βm.tolist())


def findθlatblock(ts, bsz, mrngs, bounds, x0, τ=0,
                  method='Powell', print_m=False, disp=False):
    nts = lenms(ts)
    M = len(nts)
    maxtnm = max_last_tn(ts)
    Δts = ΔtsM(ts)
    δts = δtsτMM(ts, τ)
    τts = τtsiMM(ts, τ)
    results = []
    lls = []
    for mrng in mrngs:
        optim0 = partial(lllatblock, bsz=bsz, mrng=mrng,
                         nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
        res0 = minimize(optim0, x0, method=method, bounds=bounds,
                        options={'disp': disp})
        lls = lls + [res0.fun]
        if print_m:
            print('mrng = ' + str(mrng))
            print(res0.message)
            print(res0.fun)
        for m in mrng:
            results = results + [slicexblock(res0.x, M, bsz, m)]
    return [np.array(results), np.array(lls)]


def findθlatblock_de(ts, bsz, mrngs, bounds, τ=0,
                     maxiter=1000, popsize=20, atol=0., tol=1e-1,
                     print_m=False, disp=False):
    nts = lenms(ts)
    M = len(nts)
    maxtnm = max_last_tn(ts)
    Δts = ΔtsM(ts)
    δts = δtsτMM(ts, τ)
    τts = τtsiMM(ts, τ)
    results = []
    lls = []
    for mrng in mrngs:
        optim0 = partial(lllatblock, bsz=bsz, mrng=mrng,
                         nts=nts, maxtnm=maxtnm, Δts=Δts, δts=δts, τts=τts)
        res0 = differential_evolution(optim0, bounds=bounds, strategy='best1bin',
                mutation=(0.5, 1.5), recombination=0.3, disp=disp,
                maxiter=maxiter, popsize=popsize, tol=tol, atol=atol)
        lls = lls + [res0.fun]
        if print_m:
            print('mrng = ' + str(mrng))
            print(res0.message)
            print(res0.fun)
        for m in mrng:
            results = results + [slicexblock(res0.x, M, bsz, m)]
    return [np.array(results), np.array(lls)]

# %% One dimension - no recursion (benchmarking)


# # No recursion
# def lllatnr(θ, ts, dtsns, lat=0):
#     λ0 = θ[0]
#     α = θ[1]
#     β = θ[2]
#     tn = ts[-1]
#     ri = np.zeros(len(ts))
#     for i in range(1, len(ts)):
#         ri[i] = np.sum(np.exp(-β * (ts[i] - lat - ts[:i])) * np.heaviside(tn - lat - ts[:i], 0))
#     s1 = np.sum(1 - np.exp(-β * dtsns))
#     s2 = np.sum(np.log(λ0 + α * ri))
#     return -(tn * (1 - λ0) - (α / β) * s1 + s2)


# # No recursion
# def findθlatnr(tss, lat=0, bounds=def_bounds, x0=def_x0):
#     results = []
#     for path in tss:
#         ts = path[0]
#         dtsns = dtsn(ts, lat)
#         optim0 = partial(lllatnr, ts=ts, dtsns=dtsns, lat=lat)
#         res0 = minimize(optim0, x0, method='SLSQP', bounds=def_bounds)
#         results = results + [res0.x]
#     return np.array(results)