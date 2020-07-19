#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import numpy.random as npr

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad, Function
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import os
import math
import gym
import sys
import random
import pickle as pkl
import time
import copy

import itertools
import operator

import pandas as pd

import numdifftools as nd

from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('bmh')

import logging
log = logging.getLogger(__name__)

import higher
import hydra

from mpc.env_dx.cartpole import CartpoleDx
from dcem import dcem, dcem_ctrl

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

get_params = lambda *models: itertools.chain.from_iterable(
    map(operator.methodcaller('parameters'), models))

@hydra.main(config_path='cartpole_emb.yaml')
def main(cfg):
    from cartpole_emb import CartpoleEmbExp, Decode, DCEM, GD
    exp = CartpoleEmbExp(cfg)
    # exp.estimate_full_cost()
    exp.run()


class CartpoleEmbExp():
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = os.getcwd()
        log.info(f'Saving to {self.exp_dir}')
        self.device = cfg.device
        self.dx = CartpoleDx()
        self.n_ctrl = self.dx.n_ctrl

        torch.manual_seed(cfg.seed)
        kwargs = dict(cfg.method.params)
        clz = hydra.utils.get_class(cfg.method['class'])
        self.method = clz(
            n_ctrl=self.n_ctrl,
            dx=self.dx, plan_horizon=cfg.plan_horizon,
            device=self.device,
            solve_full=self.solve_full, # Somewhat messy to do this.
            **kwargs
        )
        self.lr_sched = ReduceLROnPlateau(
            self.method.opt, 'min', patience=20, factor=0.5, verbose=True)

        torch.manual_seed(0)
        self.xinit_val = sample_init(n_batch=100).to(self.device)


    def dump(self, tag='latest'):
        fname = os.path.join(self.exp_dir, f'{tag}.pkl')
        pkl.dump(self, open(fname, 'wb'))

    # def __getstate__(self):
    #     d = copy.copy(self.__dict__)
    #     del d['x_train']
    #     del d['y_train']
    #     return d

    # def __setstate__(self, d):
    #     self.__dict__ = d
    #     self.load_data()


    def load_data(self):
        self.x_train = torch.linspace(0., 2.*np.pi, steps=self.cfg.n_samples).to(self.device)
        self.y_train = self.x_train*torch.sin(self.x_train)


    def solve_full(self, xinits):
        def unroll_dx(x0, us):
            rews, xs = rew_nominal(self.dx, x0, us)
            return xs, rews

        kwargs = copy.deepcopy(self.cfg.full_ctrl_opts)
        kwargs['init_sigma'] = kwargs['init_sigma_scale']*(self.dx.upper-self.dx.lower)
        del kwargs['init_sigma_scale']

        assert False, "TODO: Update to new dcem_ctrl interface"
        out = ctrl(
            xinits, plan_horizon=self.cfg.plan_horizon, init_mu=0.,
            lb=self.dx.lower, ub=self.dx.upper,
            rew_step=rew_step,
            n_ctrl=self.n_ctrl, unroll_dx=unroll_dx,
            dcem_verbose=False,
            **kwargs
        )
        return out


    def estimate_full_cost(self):
        costs = []

        for i in range(100):
            xinits = sample_init(n_batch=self.cfg.n_batch).to(self.device)
            out = self.solve_full(xinits)
            costs.append(out['cost'].item())
            print(f'{np.mean(costs):.2f} +/- {np.std(costs):.2f}')


    def run(self):
        hist = []
        self.best_val_loss = None
        for i in range(self.cfg.n_train_iter):
            xinits = sample_init(n_batch=self.cfg.n_batch).to(self.device)
            loss = self.method.train_step(xinits)

            hist.append({
                'iter': i,
                'loss': loss,
                'mode': 'train',
            })
            log.info(f'iter {i} loss {loss:.2f}')

            if i % self.cfg.save_interval == 0:
                u_emb, emb_cost = self.method.solve(self.xinit_val)
                val_loss = emb_cost.mean().item()
                self.lr_sched.step(val_loss)
                hist.append({
                    'iter': i,
                    'loss': val_loss,
                    'mode': 'val',
                })
                log.info(f'iter {i} val loss {loss:.2f}')
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.dump(tag='best')

                self.dump()
                # TODO: Pretty inefficient
                pd.DataFrame.from_records(hist, index='iter').to_csv(
                    f'{self.exp_dir}/losses.csv')


def rew_step(xt, ut):
    assert xt.ndimension() == 2
    assert ut.ndimension() == 2
    n_batch, n_ctrl = ut.shape
    _, nx = xt.shape
    assert xt.size(0) == n_batch

    up_rew = xt[:,2]
    dist_pen = -0.1*xt[:,0].pow(2)
    ctrl_pen = -1e-5*ut.pow(2).sum(dim=1)
    return up_rew + dist_pen + ctrl_pen


# @profile
def rew_nominal(dx, xinit, u):
    assert u.ndimension() == 3
    assert xinit.ndimension() == 2
    T, n_batch, n_ctrl = u.shape
    _, nx = xinit.shape
    assert xinit.size(0) == n_batch

    rews = 0.
    xt = xinit.clone()
    xs = [xt]
    for t in range(T):
        ut = u[t]
        rew = rew_step(xt, ut)
        rews += rew

        xt = dx(xt, ut)
        xs.append(xt)
    return rews, torch.stack(xs)


class DCEM():
    def __init__(
        self, latent_size, ctrl_opts, n_hidden, lr, n_ctrl, dx, plan_horizon, device,
        solve_full
    ):
        self.latent_size = latent_size
        self.ctrl_opts = ctrl_opts
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_ctrl = n_ctrl
        self.dx = dx
        self.plan_horizon = plan_horizon
        self.device = device

        self.decode = Decode(
            latent_size, n_hidden,
            plan_horizon, n_ctrl, dx.lower, dx.upper
        ).to(self.device)
        self.opt = optim.Adam(self.decode.parameters(), lr=lr)


    def get_cost_f(self, xinit):
        assert xinit.ndimension() == 2
        nbatch = xinit.size(0)

        def f_emb(u_emb):
            u = self.decode(u_emb.view(-1, self.latent_size))
            nsample = u.size(1)//nbatch
            xinit_sample = xinit.unsqueeze(1).repeat(1, nsample, 1)
            xinit_sample = xinit_sample.view(nbatch*nsample, -1)
            cost = -rew_nominal(self.dx, xinit_sample, u)[0]
            return cost
        return f_emb


    def solve(self, xinit, iter_cb=None):
        if xinit.ndimension() == 1:
            xinit = xinit.unsqueeze(0)
        assert xinit.ndimension() == 2
        nbatch = xinit.size(0)

        if iter_cb is None:
            def iter_cb(i, X, fX, I, X_I, mu, sigma):
                assert fX.ndimension() == 2
                I = I.view(fX.shape)
                print(f'  + {i}: {(fX*I).sum()/I.sum():.2f}')
                print(f'    + {I.min().item():.2f}/{I.max().item():.2f}')

        # torch.manual_seed(0)
        f_emb = self.get_cost_f(xinit)
        u_emb = dcem(
            f_emb, self.latent_size, init_mu=None,
            n_batch=nbatch, device=self.device, iter_cb=iter_cb,
            **self.ctrl_opts
        )
        emb_cost = f_emb(u_emb)
        return u_emb, emb_cost


    def train_step(self, xinit):
        assert xinit.ndimension() == 2 and xinit.size(0) == 1
        xinit = xinit.squeeze()

        u_emb, emb_cost = self.solve(xinit)
        loss = emb_cost.sum()

        self.opt.zero_grad()
        loss.backward()
        # print('---')
        # util.print_grad(self.decode)
        nn.utils.clip_grad_norm_(self.decode.parameters(), 1.0)
        # util.print_grad(self.decode)
        self.opt.step()

        return loss.item()


class GD():
    def __init__(
        self, latent_size, n_hidden, lr,
        inner_optim_opts,
        n_ctrl, dx, plan_horizon, device,
        solve_full,
    ):
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.lr = lr
        self.inner_optim_opts = inner_optim_opts
        self.n_ctrl = n_ctrl
        self.dx = dx
        self.plan_horizon = plan_horizon
        self.device = device

        self.decode = Decode(
            latent_size, n_hidden,
            plan_horizon, n_ctrl, dx.lower, dx.upper
        ).to(self.device)
        self.opt = optim.Adam(self.decode.parameters(), lr=lr)


    def get_cost_f(self, xinit):
        def f_emb(u_emb):
            u = self.decode(u_emb.view(-1, self.latent_size))
            # xinit_batch = xinit.repeat(u.size(1), 1) # TODO: Inefficient
            cost = -rew_nominal(self.dx, xinit, u)[0]
            return cost
        return f_emb


    def solve(self, xinit):
        assert xinit.ndimension() == 2

        nbatch = xinit.size(0)
        z = torch.zeros(nbatch, self.latent_size,
                        device=xinit.device, requires_grad=True)

        inner_opt = higher.get_diff_optim(
            torch.optim.SGD([z], lr=self.inner_optim_opts.lr),
            [z], device=xinit.device
        )

        f_emb = self.get_cost_f(xinit)
        for _ in range(self.inner_optim_opts.n_iter):
            cost = f_emb(z)
            z, = inner_opt.step(cost.sum(), params=[z])

        us = self.decode(z)
        rews, xs = rew_nominal(self.dx, xinit, us)
        cost = -rews
        return z, cost


    def train_step(self, xinit):
        assert xinit.ndimension() == 2
        _, cost = self.solve(xinit)
        loss = cost.mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decode.parameters(), 1.0)
        self.opt.step()

        return loss.item()


class Decode(nn.Module):
    def __init__(self, n_in, n_hidden, plan_horizon, n_ctrl, lb, ub):
        super().__init__()
        self.plan_horizon = plan_horizon
        self.n_ctrl = n_ctrl
        self.lb = lb
        self.ub = ub

        act = nn.ELU
        bias = True
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=bias),
            act(),
            nn.Linear(n_hidden, n_hidden, bias=bias),
            act(),
            nn.Linear(n_hidden, n_hidden, bias=bias),
            act(),
            nn.Linear(n_hidden, plan_horizon*n_ctrl, bias=bias),
        )

        def init(m):
            if isinstance(m, nn.Linear):
                fan_out, fan_in = m.weight.size()
                m.weight.data.normal_(0.0, 2./np.sqrt(fan_in+fan_out))
                if bias:
                    m.bias.data.zero_()

        self.net.apply(init)

    def forward(self, x):
        assert x.ndimension() == 2
        nbatch = x.size(0)
        u = self.net(x)
        r = self.ub-self.lb
        u = r*torch.sigmoid(u)+self.lb
        u = u.t().view(self.plan_horizon, self.n_ctrl, nbatch).transpose(1,2)
        return u


def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low


def sample_init(n_batch=1):
    pos = uniform(n_batch, -0.5, 0.5)
    dpos = uniform(n_batch, -0.5, 0.5)
    th = uniform(n_batch, -np.pi, np.pi)
    dth = uniform(n_batch, -1., 1.)
    xinit = torch.stack((pos, dpos, torch.cos(th), torch.sin(th), dth), dim=1)
    return xinit


def plot(fname, title=None):
    x = np.linspace(-1.0, 1.0, 20)
    y = np.linspace(-1.0, 1.0, 20)
    X, Y = np.meshgrid(x, y)
    Xflat = X.reshape(-1)
    Yflat = Y.reshape(-1)
    XYflat = np.stack((Xflat, Yflat), axis=1)
    Zflat = f_emb_np(XYflat)
    Z = Zflat.reshape(X.shape)

    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z, cmap=cm.Blues)
    fig.colorbar(CS)
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(fname)


if __name__ == '__main__':
    main()
