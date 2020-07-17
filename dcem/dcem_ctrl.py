# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from dcem import dcem

def dcem_ctrl(
    obs,
    plan_horizon,
    init_mu,
    init_sigma,
    n_sample,
    n_elite,
    n_iter,
    n_ctrl,
    temp,
    rollout_cost,
    lb=None,
    ub=None,
    iter_cb=None,
    dcem_iter_eps=1e-3,
    dcem_normalize=True,
):
    squeeze = obs.ndimension() == 1
    if squeeze:
        obs = obs.unsqueeze(0)

    assert obs.ndimension() == 2
    n_batch, nx = obs.size()

    cem_u_dim = plan_horizon*n_ctrl

    obs_sample = obs.unsqueeze(1).repeat(1, n_sample, 1)
    obs_sample = obs_sample.view(n_batch*n_sample, -1)

    def f(u):
        assert u.ndimension() == 3
        assert u.size(0) == n_batch
        assert u.size(2) == cem_u_dim
        N = u.size(1)

        assert u.ndimension() == 3
        assert u.size() == (n_batch, N, plan_horizon*n_ctrl)
        u = u.view(n_batch*N, plan_horizon, n_ctrl)
        u = u.transpose(0, 1)
        assert u.size() == (plan_horizon, n_batch*N, n_ctrl)

        if N == 1:
            xt = obs.clone()
        elif N == n_sample:
            xt = obs_sample.clone()
        else:
            assert False

        cost = rollout_cost(xt, u)
        assert cost.ndim == 1 and cost.size(0) == n_batch*N

        cost.view(n_batch, -1)
        return cost

    if init_mu is not None and not isinstance(init_mu, float):
        assert init_mu.ndimension() == 2
        assert init_mu.size() == (n_batch, cem_u_dim)

    u = dcem(
        f, cem_u_dim,
        n_batch=n_batch,
        n_sample=n_sample,
        n_elite=n_elite, n_iter=n_iter,
        temp=temp,
        lb=lb, ub=ub,
        normalize=dcem_normalize,
        init_mu=init_mu, init_sigma=init_sigma,
        iter_eps=dcem_iter_eps,
        device=obs.device,
        iter_cb=iter_cb,
    )
    cost = f(u.unsqueeze(1))

    out = {}

    assert u.size() == (n_batch, plan_horizon*n_ctrl)
    u = u.view(n_batch, plan_horizon, n_ctrl)
    u = u.transpose(0, 1)
    assert u.size() == (plan_horizon, n_batch, n_ctrl)

    if squeeze:
        u = u.squeeze(1)

    out['u'] = u
    out['init_u'] = u[0]
    out['cost'] = cost
    out['f'] = f

    return out
