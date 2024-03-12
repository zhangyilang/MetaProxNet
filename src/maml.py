import os

import torch
from torchmeta.utils import gradient_update_parameters

from src.meta_alg_base import MetaLearningAlgBase


class MAML(MetaLearningAlgBase):
    def __init__(self, args):
        super(MAML, self).__init__(args)

    def meta_optimizer(self):
        return torch.optim.Adam(self.model.meta_parameters(), lr=self.args.meta_lr), None

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, file_name)))

    def adaptation(self, x_supp, y_supp, first_order):
        params = None

        for _ in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            params = gradient_update_parameters(self.model,
                                                task_loss,
                                                params=params,
                                                step_size=self.args.task_lr,
                                                first_order=first_order)

        return params
