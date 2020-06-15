#!/usr/bin/env python3

import argparse
import typing

import copy

import numpy as np
import numpy.random as npr

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


import higher
import csv
import os

import pickle as pkl

from dcem import dcem

import hydra

from setproctitle import setproctitle
setproctitle('regression')

@hydra.main(config_path="regression-conf.yaml", strict=True)
def main(cfg):
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    print('Current dir: ', os.getcwd())

    from regression import RegressionExp, UnrollEnergyGD, UnrollEnergyCEM
    exp = RegressionExp(cfg)
    exp.run()


class RegressionExp():
    def __init__(self, cfg):
        self.cfg = cfg

        self.exp_dir = os.getcwd()
        self.model_dir = os.path.join(self.exp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        torch.manual_seed(cfg.seed)
        npr.seed(cfg.seed)

        self.device = torch.device("cuda")
        self.Enet = EnergyNet(n_in=1, n_out=1, n_hidden=cfg.n_hidden).to(self.device)
        self.model = hydra.utils.instantiate(cfg.model, self.Enet)
        self.load_data()


    def dump(self, tag='latest'):
        fname = os.path.join(self.exp_dir, f'{tag}.pkl')
        pkl.dump(self, open(fname, 'wb'))

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['x_train']
        del d['y_train']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.load_data()

    def load_data(self):
        self.x_train = torch.linspace(0., 2.*np.pi, steps=self.cfg.n_samples).to(self.device)
        self.y_train = self.x_train*torch.sin(self.x_train)

    def run(self):
        # opt = optim.SGD(self.Enet.parameters(), lr=1e-1)
        opt = optim.Adam(self.Enet.parameters(), lr=1e-3)
        lr_sched = ReduceLROnPlateau(opt, 'min', patience=20, factor=0.5, verbose=True)

        fieldnames = ['iter', 'loss']
        f = open(os.path.join(self.exp_dir, 'loss.csv'), 'w')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        step = 0
        while step < self.cfg.n_update:
            if (step in list(range(100)) or step % 10000 == 0):
                self.dump(f'{step:07d}')

            j = npr.randint(self.cfg.n_samples)
            for i in range(self.cfg.n_inner_update):
                y_preds = self.model(self.x_train[j].view(1)).squeeze()
                loss = F.mse_loss(input=y_preds, target=self.y_train[j])
                opt.zero_grad()
                loss.backward(retain_graph=True)
                if self.cfg.clip_norm:
                    nn.utils.clip_grad_norm_(self.Enet.parameters(), 1.0)
                opt.step()
                step += 1

            if step % 100 == 0:
                y_preds = self.model(self.x_train.view(-1, 1)).squeeze()
                loss = F.mse_loss(input=y_preds, target=self.y_train)
                lr_sched.step(loss)
                print(f'{step}: {loss:.2f}')
                writer.writerow({
                    'iter': step,
                    'loss': loss.item(),
                })
                f.flush()
                exp_dir = os.getcwd()
                fieldnames = ['iter', 'loss', 'lr']
                self.dump('latest')


class EnergyNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden: int = 256):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.E_net = nn.Sequential(
            nn.Linear(n_in+n_out, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, y):
        z = torch.cat((x, y), dim=-1)
        E = self.E_net(z)
        return E


class UnrollEnergyGD(nn.Module):
    def __init__(self, Enet: EnergyNet, n_inner_iter, inner_lr):
        super().__init__()
        self.Enet = Enet
        self.n_inner_iter = n_inner_iter
        self.inner_lr = inner_lr

    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        y = torch.zeros(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)

        inner_opt = higher.get_diff_optim(
            torch.optim.SGD([y], lr=self.inner_lr),
            [y], device=x.device
        )

        for _ in range(self.n_inner_iter):
            E = self.Enet(x, y)
            y, = inner_opt.step(E.sum(), params=[y])

        return y


class UnrollEnergyCEM(nn.Module):
    def __init__(self, Enet: EnergyNet, n_sample, n_elite, n_iter, init_sigma, temp, normalize):
        super().__init__()
        self.Enet = Enet
        self.n_sample = n_sample
        self.n_elite = n_elite
        self.n_iter = n_iter
        self.init_sigma = init_sigma
        self.temp = temp
        self.normalize = normalize


    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        def f(y):
            _x = x.unsqueeze(1).repeat(1, y.size(1), 1)
            Es = self.Enet(_x.view(-1, 1), y.view(-1, 1)).view(y.size(0), y.size(1))
            return Es

        yhat = dcem(
            # f, n_batch=nbatch, nx = 1, n_sample = 20, n_elite = 5, n_iter = 10,
            f, n_batch=nbatch, nx = 1, n_sample = self.n_sample, n_elite = self.n_elite,
            n_iter = self.n_iter, init_sigma=self.init_sigma, temp = self.temp,
            device=x.device, normalize=self.temp,
            #, lb = -1., ub = 1.,
        )

        return yhat


if __name__ == '__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    main()
