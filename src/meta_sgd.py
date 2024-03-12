import os
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torchmeta.utils import gradient_update_parameters

from src.meta_alg_base import MetaLearningAlgBase


class MetaSGD(MetaLearningAlgBase):
    def __init__(self, args):
        super(MetaSGD, self).__init__(args)

        self.log_step_size = OrderedDict()
        for name, param in self.model.meta_named_parameters():
            self.log_step_size[name] = nn.Parameter(torch.zeros_like(param) + math.log(self.args.task_lr))

    def meta_optimizer(self):
        return torch.optim.Adam(list(self.model.meta_parameters()) +
                                list(self.log_step_size.values()),
                                lr=self.args.meta_lr), None

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'log_step_size': self.log_step_size},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.log_step_size = OrderedDict(state_dicts['log_step_size'])

    def adaptation(self, x_supp, y_supp, first_order):
        params = None
        step_size = OrderedDict({name: param.exp() for name, param in self.log_step_size.items()})

        for _ in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            params = gradient_update_parameters(self.model,
                                                task_loss,
                                                params=params,
                                                step_size=step_size,
                                                first_order=first_order)

        return params
