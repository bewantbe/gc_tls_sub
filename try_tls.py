# test TLS in time series

# also test singular spectrum analysis (SSA)

import os
import sys
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# use JAX
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap

from tqdm import tqdm

def GenToeplitz(x, od):
    n1 = len(x) - od
    # X: [[xt, xt-1, ..., xt-od], ...].T
    X = np.zeros((od + 1, n1))
    for j in range(od+1):
        X[j, :] = x[od-j:n1+od-j]
    return X

def GenToeplitzJNP(x, od):
    """Version suitable for JAX"""
    n1 = len(x) - od
    # X: [[xt, xt-1, ..., xt-od], ...].T
    X = jnp.zeros((od + 1, n1))
    for j in range(od+1):
        X = X.at[j, :].set(x[od-j:n1+od-j])
    return X

def SSA(X, ext_mode = 'toep'):
    assert X.shape[0] < X.shape[1], 'X should be a fat matrix'
    u, s, vh = svd(X, full_matrices=False)
    #print(np.linalg.norm(X - np.dot(u, np.dot(np.diag(s), vh))))

    od = len(s) - 1
    n = X.shape[1]
    if ext_mode == 'toep_full':
        h_ts = np.zeros((od+1, n+od))
        w = np.zeros((od + 1, n+od))
        nw = np.hstack([np.arange(1,od+1), np.ones(n-od) * (od+1), np.arange(od, 0, -1)])
        for k in range(od+1):
            w[:,:] = 0
            for j in range(od+1):
                w[j, od-j:od+n-j] = u[j, k] * vh[k, :]
            h_ts[k, :] = np.sum(w, axis=0) / nw
    elif ext_mode == 'toep':
        h_ts = np.zeros_like(X)
        w = np.zeros((od + 1, n))
        nw = np.hstack([np.ones(n-od) * (od+1), np.arange(od, 0, -1)])
        for k in range(od+1):
            w[:,:] = 0
            for j in range(od+1):
                w[j, :n-j] = u[j, k] * vh[k, j:]
            h_ts[k, :] = np.sum(w, axis=0) / nw
    elif ext_mode == 'pca':
        h_ts = (u[0:1,:].T * (u.T @ X)) / s[:, None]

    return s, h_ts

def ShowTSComponents(x, s, h_ts):
    od = h_ts.shape[0] - 1
    n = len(x)

    re_x = s @ h_ts
    assert np.linalg.norm(x - re_x) / np.linalg.norm(x) < 1e-10, 're_x is not equal to x'

    plt.figure()
    plt.plot(x, 'b', label='original')
    #plt.plot(re_x, 'r', label='sum')
    for k in range(od+1):
        plt.plot(s[k] * h_ts[k, :], label='h_ts[%d]' % k)
    plt.legend()
    plt.show()

def TSSVDCost(X, eta, rank_weight):
    E = GenToeplitzJNP(eta, od)
    X_eta = X - E
    #l = jnp.linalg.eigvalsh(X_eta.dot(X_eta.T))
    l = jnp.linalg.svd(X_eta, compute_uv=False)
    l = l * l
    # minimize min(l), ideally min(l) = 0
    # for simultaneously minimize eta.T @ eta
    return jnp.sum(eta * eta) + rank_weight * jnp.min(l)

def FindTSSVD(x_orig, od):
    x = x_orig[od:]
    X = GenToeplitz(x_orig, od)
    # find a eta so that X is rank-deficient
    # eta is a JAX variable in the size of x_orig
    init_mode = 'ssa'
    if init_mode == 'zeros':
        eta = np.zeros_like(x_orig)
    elif init_mode == 'randn':
        eta = np.random.randn(len(x_orig))
    elif init_mode == 'ssa':
        s, h_ts = SSA(X, ext_mode='toep_full')
        eta = s[-1] * h_ts[-1, :]
    else:
        raise ValueError('unknown init')
    # use JAX to minimize the objective function
    # parameters:
    max_iter = 500
    learning_rate = 0.0001
    rank_weight = 1000.0
    n_reheat = 4
    reheat_ratio = 0.1  # 0.125
    rank_weight_multiplier = 1.0 * np.ones(n_reheat)
    # derivative of TSSVDCost w.r.t. eta
    d_TSSVDCost = grad(TSSVDCost, 1)
    log_loss = []
    log_vareta = []
    last_eta = eta
    # minimize the cost function
    for i_reheat in range(n_reheat):
        for i in tqdm(range(max_iter), desc='iter'):
            eta = eta - learning_rate * d_TSSVDCost(X, eta, rank_weight)
            log_loss.append(TSSVDCost(X, eta, rank_weight))
            log_vareta.append(jnp.sum(eta*eta))
        last_eta = eta
        rank_weight *= rank_weight_multiplier[i_reheat]
        learning_rate /= rank_weight_multiplier[i_reheat]
        r = np.random.randn(len(x_orig))
        r = r / np.linalg.norm(r) * np.sqrt(log_vareta[-1]) * reheat_ratio
        eta = eta + r
    
    print(f'var(eta) = {log_vareta[-1]}, std(eta) = {np.sqrt(log_vareta[-1])}')
    print(f'loss = {log_loss[-1]}')
    print(f'eig_min_margin = {log_loss[-1] - log_vareta[-1]}')
    print(f'svd_s_orig = {jnp.linalg.svd(X, compute_uv=False)}')
    E = GenToeplitzJNP(last_eta, od)
    X_eta = X - E
    print(f'svd_s_opti = {jnp.linalg.svd(X_eta, compute_uv=False)}')
    
    info = SimpleNamespace(log_loss=log_loss, log_vareta=log_vareta)
    return last_eta, info

if __name__ == '__main__':
    # fix random seed of np.randn
    np.random.seed(0)
    n = 1000
    od = 3   # fitting order
    x_orig = np.random.randn(n + od)
    # filter x_orig by a moving average filter
    x_orig = np.convolve(x_orig, np.ones(5)/5, mode='same')
    x = x_orig[od:]
    X = GenToeplitz(x_orig, od)

    #s, h_ts = SSA(X)
    #ShowTSComponents(x, s, h_ts)

    if 0:
        s, h_ts = SSA(X, ext_mode='toep_full')
        ShowTSComponents(x_orig, s, h_ts)
        #print(h_ts @ h_ts.T)

        # test SSA for rank reduction
        x_reduce = x_orig - s[-1] * h_ts[-1, :]

        print('SSA rank reduction')
        s_orig = svd(GenToeplitz(x_orig, od), compute_uv=False)
        print('X_trim s_orig    = ', s_orig)

        s_ssa = svd(GenToeplitz(x_reduce, od), compute_uv=False)
        print('X_trim s_SSA[-1] = ', s_ssa)

    if 0: # non-full matrix cmp
        s, h_ts = SSA(X, ext_mode='pca')
        ShowTSComponents(x, s, h_ts)
        #print(h_ts @ h_ts.T)

        # test SSA for rank reduction
        x_reduce = x - s[-1] * h_ts[-1, :]

        print('SSA rank reduction')
        s_orig = svd(GenToeplitz(x, od), compute_uv=False)
        print('X_trim s_orig    = ', s_orig)

        s_ssa  = svd(GenToeplitz(x_reduce, od), compute_uv=False)
        print('X_trim s_SSA[-1] = ', s_ssa)

    if 1:
        eta, info = FindTSSVD(x_orig, od)
    
        plt.figure()
        plt.plot(info.log_loss, label='loss')
        plt.plot(info.log_vareta, label='vareta')
        plt.legend()

        plt.figure()
        plt.plot(x_orig, 'b', label='original')
        plt.plot(eta, 'g', label='eta')
        plt.plot(x_orig - eta, 'r', label='residual')
        plt.legend()
        plt.show()

        # check rank reduction
        x_reduce = x_orig - eta
        s_reduce = svd(GenToeplitz(x_reduce, od), compute_uv=False)
        print('X_trim s_opt-1 = ', s_reduce)
