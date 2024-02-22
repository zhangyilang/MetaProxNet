import os
import torch
import torch.nn as nn
from collections import OrderedDict
from src.meta_alg_base import MetaLearningAlgBase
from src.utils import elementwise_Softshrink


class MetaPGDGaussian(MetaLearningAlgBase):
    def __init__(self, args):
        super(MetaPGDGaussian, self).__init__(args)

        self.lambd = OrderedDict()
        for name, param in self.model.meta_named_parameters():
            self.lambd[name] = nn.Parameter(torch.ones_like(param))

    def meta_optimizer(self):
        return torch.optim.Adam(list(self.model.meta_parameters()) +
                                list(self.lambd.values()),
                                lr=self.args.meta_lr)

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'lambd': self.lambd},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.lambd = OrderedDict(state_dicts['lambd'])

    def adaptation(self, x_supp, y_supp, first_order):
        params_init = OrderedDict(self.model.meta_named_parameters())
        params = OrderedDict(self.model.meta_named_parameters())
        params_shift = OrderedDict({name: torch.zeros_like(param)
                                    for name, param in params.items()})

        for task_idx in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            grads = torch.autograd.grad(task_loss,
                                        params.values(),
                                        create_graph=not first_order)

            for (name, param), grad in zip(params_shift.items(), grads):
                params_shift[name] = (param - grad) / (1 + self.args.task_lr * self.lambd[name].abs())

            for name, param in params.items():
                params[name] = params_init[name] + self.args.task_lr * params_shift[name]

        return params


class MetaPGDSparse(MetaPGDGaussian):
    def __init__(self, args):
        super(MetaPGDSparse, self).__init__(args)

        for name, param in self.model.meta_named_parameters():
            self.lambd[name] = nn.Parameter(torch.ones_like(param) * 1e-2)

    def adaptation(self, x_supp, y_supp, first_order):
        params_init = OrderedDict(self.model.meta_named_parameters())
        params = OrderedDict(self.model.meta_named_parameters())
        params_shift = OrderedDict({name: torch.zeros_like(param)
                                    for name, param in params.items()})

        for task_idx in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            grads = torch.autograd.grad(task_loss,
                                        params.values(),
                                        create_graph=not first_order)

            for (name, param), grad in zip(params_shift.items(), grads):
                params_shift[name] = elementwise_Softshrink(param - grad, self.lambd[name].abs())

            for name, param in params.items():
                params[name] = params_init[name] + self.args.task_lr * params_shift[name]

        return params
