# Copyright (c) Facebook, Inc. and its affiliates.

import torch

import sys
sys.path.append('..')
from dcem import dcem

def dcem_ctrl(
    obs,
    plan_horizon,
    init_mu,
    init_sigma,
    n_sample,
    n_elite,
    n_iter,
    rew_step,
    n_ctrl,
    temp,
    unroll_dx,
    discount=1.,
    slew_rate_penalty=None,
    Q=None,
    Q_reduce=None,
    u_decode=None,
    u_latent_dim=None,
    lb=-1.,
    ub=1.,
    dcem_iter_eps=1e-3,
    dcem_normalize=True,
    dcem_verbose=False,
):
    squeeze = obs.ndimension() == 1
    if squeeze:
        obs = obs.unsqueeze(0)

    assert obs.ndimension() == 2
    n_batch, nx = obs.size()

    if u_latent_dim is not None:
        assert u_decode is not None
        cem_u_dim = u_latent_dim
    else:
        cem_u_dim = plan_horizon*n_ctrl

    obs_sample = obs.unsqueeze(1).repeat(1, n_sample, 1)
    obs_sample = obs_sample.view(n_batch*n_sample, -1)

    # @profile
    # TODO: It could be cleaner to pull this out
    def f(u):
        # assert u.size() == (n_batch, n_sample, cem_u_dim)
        assert u.ndimension() == 3
        assert u.size(0) == n_batch
        assert u.size(2) == cem_u_dim
        N = u.size(1)

        if u_decode:
            u = u_decode(u.reshape(-1, cem_u_dim))
            assert u.size() == (plan_horizon, n_batch*N, n_ctrl)
        else:
            assert u.ndimension() == 3
            assert u.size() == (n_batch, N, plan_horizon*n_ctrl)
            u = u.view(n_batch*N, plan_horizon, n_ctrl)
            u = u.transpose(0, 1)
            assert u.size() == (plan_horizon, n_batch*N, n_ctrl)


        if plan_horizon > 1 or (plan_horizon == 1 and Q is None):
            if N == 1:
                xt = obs.clone()
            elif N == n_sample:
                xt = obs_sample.clone()
            else:
                assert False

            # TODO: Think more about the interface here
            xts, rews = unroll_dx(xt, u)
            assert rews.size(0) == n_batch*N

            # rews = rew_step(xts.view(-1, nx), u.contiguous().view(-1, n_ctrl))
            # rews = rews.view(plan_horizon, n_batch, N)

            # if Q is not None:
            #     rews = rews[:-1,:,:]
            # gammas = torch.Tensor(
            #     [discount**t for t in range(rews.shape[0])]
            # ).unsqueeze(1).unsqueeze(2).to(obs.device)
            # rews *= gammas
            # rews = rews.sum(dim=0)
            # rews = rews.view(n_batch*N)

            # if dx_rew is not None:
            #     assert dx_rew.ndimension() == 1
            #     assert dx_rew.size(0) == n_batch*N
            #     rews += dx_rew

            # TODO
            # rews -= 1.*pred_dists.sigma.mean(dim=0).mean(dim=1).mean(dim=1)
        else:
            xts = [obs.clone() if N == 1 else obs_sample]
            rews = 0.

        if Q is not None:
            gamma_T = discount**(plan_horizon-1)
            xT = xts[-1]
            uT = u[-1]

            Q1, Q2, _ = Q(xT, uT)
            QTs = torch.cat((Q1, Q2), dim=1)
            if Q_reduce == 'mean':
                QT = QTs.mean(dim=1)
            elif Q_reduce == 'max':
                QT = QTs.max(dim=1).values
            elif Q_reduce == 'min':
                QT = QTs.max(dim=1).values
            else:
                assert False
            rews += gamma_T*QT

        obj = -rews
        if plan_horizon > 1 and slew_rate_penalty is not None \
                and slew_rate_penalty > 0.:
            obj += slew_rate_penalty * \
                (u[:-1]-u[1:]).pow(2).mean(dim=0).mean(dim=1)
        obj.view(n_batch, -1)
        return obj


    if dcem_verbose:
        def iter_cb(i, X, fX, I, X_I, mu, sigma):
            assert fX.ndimension() == 2
            I = I.view(fX.shape)
            print(f'  + {i}: {(fX*I).sum()/I.sum():.2f}')
    else:
        iter_cb = None


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
        iter_cb = iter_cb,
    )
    cost = f(u.unsqueeze(1))

    # TODO: Consider using another return object here
    out = {}

    if u_decode is not None:
        out['z'] = u
        u = u_decode(u)
        assert u.size() == (plan_horizon, n_batch, n_ctrl)
    else:
        assert u.size() == (n_batch, plan_horizon*n_ctrl)
        u = u.view(n_batch, plan_horizon, n_ctrl)
        u = u.transpose(0, 1)
        assert u.size() == (plan_horizon, n_batch, n_ctrl)

    if squeeze:
        u = u.squeeze(1)

    out['u_nominal'] = u
    out['init_u'] = u[0]
    out['cost'] = cost
    out['f'] = f

    return out
