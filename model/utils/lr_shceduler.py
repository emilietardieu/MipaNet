##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

__all__ = ['LR_Scheduler', 'LR_Scheduler_Head']

class LR_Scheduler(object):
    """
    Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

        :param str mode: learning rate scheduler mode, one of `cos`, `poly`, `step`
        :param float base_lr: base learning rate
        :param int num_epochs: number of epochs to train
        :param int iters_per_epoch: number of iterations per epoch, used to calculate the
            total number of iterations for the scheduler
        :param int lr_step: step size for step mode, ignored in other modes
        :param int warmup_epochs: number of warmup epochs, during which the learning rate
            increases linearly from 0 to the base learning rate
        :param bool quiet: if True, suppresses output messages about the scheduler
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False):
        self.mode = mode
        self.quiet = quiet
        # if not quiet:
        #     print('Utilisation de {} LR scheduler avec {} époques de'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        """
        Appelée pour ajuster le taux d'apprentissage de l'optimiseur
            :param optimizer: l'optimiseur dont le taux d'apprentissage doit être ajusté
            :param int i: l'itération actuelle dans l'époque
            :param int epoch: le numéro de l'époque actuelle
            :param float best_pred: la meilleure prédiction actuelle, utilisée pour décider si l'époque
        """
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

class LR_Scheduler_Head(LR_Scheduler):
    """
    Learning Rate Scheduler for the head of the model
    """
    def _adjust_learning_rate(self, optimizer, lr):
        """
        Adjust the learning rate of the optimizer.
            :param optimizer: the optimizer whose learning rate needs to be adjusted
            :param float lr: the new learning rate to set
        """
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10