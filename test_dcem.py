#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
from dcem import dcem

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)

def test_dcem():
    n_batch = 2
    n_sample = 100
    N = 2

    torch.manual_seed(0)
    Q = torch.eye(N).unsqueeze(0).repeat(n_batch, 1, 1)
    p = 0.1*torch.randn(n_batch, N)

    Q_sample = Q.unsqueeze(1).repeat(1, n_sample, 1, 1).view(n_batch*n_sample, N, N)
    p_sample = p.unsqueeze(1).repeat(1, n_sample, 1).view(n_batch*n_sample, N)

    def f(X):
        assert X.size() == (n_batch, n_sample, N)
        X = X.view(n_batch*n_sample, N)
        obj = 0.5*(bmv(Q_sample, X)*X).sum(dim=1) + (p_sample*X).sum(dim=1)
        obj = obj.view(n_batch, n_sample)
        return obj

    def iter_cb(i, X, fX, I, X_I, mu, sigma):
        print(fX.mean(dim=1))

    xhat = dcem(
        f, nx = N, n_batch = n_batch, n_sample = n_sample,
        n_elite = 50, n_iter = 40,
        temp = 1.,
        normalize = True,
        # temp = np.infty,
        init_sigma=1.,
        iter_cb = iter_cb,
        # lb = -5., ub = 5.,
    )

    Q_LU = torch.lu(Q)
    xstar = -torch.lu_solve(p, *Q_LU)
    assert (xhat-xstar).abs().max() < 1e-4


if __name__ == '__main__':
    test_dcem()
