# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal

from lml import LML

def dcem(
    f,
    nx,
    n_batch=1,
    n_sample=20,
    n_elite=10,
    n_iter=10,
    temp=1.,
    lb=None,
    ub=None,
    init_mu=None,
    init_sigma=None,
    device=None,
    iter_cb=None,
    proj_iterate_cb=None,
    lml_verbose=0,
    lml_eps=1e-3,
    normalize=True,
    iter_eps=1e-4,
):
    if init_mu is None:
        init_mu = 0.

    size = (n_batch, nx)

    if isinstance(init_mu, torch.Tensor):
        mu = init_mu.clone()
    elif isinstance(init_mu, float):
        mu = init_mu * torch.ones(size, requires_grad=True, device=device)
    else:
        assert False

    # TODO: Check if init_mu is in the domain

    if init_sigma is None:
        init_sigma = 1.

    if isinstance(init_sigma, torch.Tensor):
        sigma = init_sigma.clone()
    elif isinstance(init_sigma, float):
        sigma = init_sigma * torch.ones(
            size, requires_grad=True, device=device)
    else:
        assert False

    assert mu.size() == size
    assert sigma.size() == size

    if lb is not None:
        assert isinstance(lb, float)

    if ub is not None:
        assert isinstance(ub, float)
        assert ub > lb

    for i in range(n_iter):
        X = Normal(mu, sigma).rsample((n_sample,)).transpose(0, 1).to(device)
        X = X.contiguous()
        if lb is not None or ub is not None:
            X = torch.clamp(X, lb, ub)

        if proj_iterate_cb is not None:
            X = proj_iterate_cb(X)

        fX = f(X)
        X, fX = X.view(n_batch, n_sample, -1), fX.view(n_batch, n_sample)

        if temp is not None and temp < np.infty:
            if normalize:
                fX_mu = fX.mean(dim=1).unsqueeze(1)
                fX_sigma = fX.std(dim=1).unsqueeze(1)
                _fX = (fX - fX_mu) / (fX_sigma + 1e-6)
            else:
                _fX = fX

            if n_elite == 1:
                # I = LML(N=n_elite, verbose=lml_verbose, eps=lml_eps)(-_fX*temp)
                I = torch.softmax(-_fX * temp, dim=1)
            else:
                I = LML(N=n_elite, verbose=lml_verbose,
                        eps=lml_eps)(-_fX * temp)
            I = I.unsqueeze(2)
        else:
            I_vals = fX.argsort(dim=1)[:, :n_elite]
            # TODO: A scatter would be more efficient here.
            I = torch.zeros(n_batch, n_sample, device=device)
            for j in range(n_batch):
                for v in I_vals[j]:
                    I[j, v] = 1.
            I = I.unsqueeze(2)

        assert I.shape[:2] == X.shape[:2]
        X_I = I * X
        old_mu = mu.clone()
        mu = torch.sum(X_I, dim=1) / n_elite
        if (mu - old_mu).norm() < iter_eps:
            break
        sigma = ((I * (X - mu.unsqueeze(1))**2).sum(dim=1) / n_elite).sqrt()

        if iter_cb is not None:
            iter_cb(i, X, fX, I, X_I, mu, sigma)

    if lb is not None or ub is not None:
        mu = torch.clamp(mu, lb, ub)

    return mu
