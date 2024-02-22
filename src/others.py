import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.meta_prox import MetaProx


class MetaProxMetaSGD(MetaProx):
    def __init__(self, args):
        super(MetaProxMetaSGD, self).__init__(args)

        self.log_step_size = OrderedDict()
        for name, param in self.model.meta_named_parameters():
            self.log_step_size[name] = nn.Parameter(torch.ones_like(param) * np.log(self.args.task_lr))

    def meta_optimizer(self):
        if self.args.dataset.lower() == 'miniimagenet':
            optimizer = torch.optim.SGD(list(self.model.meta_parameters()) +
                                        list(self.proximal_net.parameters()) +
                                        list(self.log_step_size.values()),
                                        lr=self.args.meta_lr,
                                        momentum=0.9,
                                        weight_decay=1e-4,
                                        nesterov=True)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000)
        else:
            optimizer = torch.optim.Adam(list(self.model.meta_parameters()) +
                                         list(self.proximal_net.parameters()) +
                                         list(self.log_step_size.values()),
                                         lr=self.args.meta_lr)
            lr_scheduler = None

        return optimizer, lr_scheduler

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'proximal_net': self.proximal_net.state_dict(),
                    'log_step_size': self.log_step_size},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.proximal_net.load_state_dict(state_dicts['proximal_net'])
        self.log_step_size = OrderedDict(state_dicts['log_step_size'])

    def adaptation(self, x_supp, y_supp, first_order):
        params_init = OrderedDict(self.model.meta_named_parameters())
        params = OrderedDict(self.model.meta_named_parameters())
        params_shift = OrderedDict({name: torch.zeros_like(param)
                                    for name, param in params.items()})
        step_size = OrderedDict({name: param.exp()
                                 for name, param in self.log_step_size.items()})

        for task_idx in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            grads = torch.autograd.grad(task_loss,
                                        params.values(),
                                        create_graph=not first_order)

            for (name, param), grad in zip(params_shift.items(), grads):
                params_shift[name] = param - grad

            if self.args.share_param:
                task_idx = None
            params_shift = self.proximal_net(params_shift, task_idx)

            for name, param in params.items():
                params[name] = params_init[name] + step_size[name] * params_shift[name]

        return params
