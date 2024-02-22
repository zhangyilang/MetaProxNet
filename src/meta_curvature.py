import os
import torch
import torch.nn as nn
from collections import OrderedDict
from src.meta_alg_base import MetaLearningAlgBase


class TensorModeProd(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.mc_weight = nn.ParameterList()

        for param in params:
            param_size = param.size()
            param_dim = param.dim()

            if param_dim == 1:      # Conv2d().bias / Linear().bias
                self.mc_weight.append(nn.Parameter(torch.ones_like(param)))
            else:                   # Linear().weight / Conv2d().weight
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[0])))
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[1])))
                if param_dim == 4:  # Conv2d().weight
                    self.mc_weight.append(nn.Parameter(torch.eye(param_size[2] * param_size[3])))

    def forward(self, input_grads):
        output_grads = list()
        pointer = 0

        for input_grad in input_grads:
            param_dim = input_grad.dim()

            if param_dim == 1:  # Conv2d().bias / Linear().bias
                output_grad = self.mc_weight[pointer] * input_grad
                pointer += 1
            elif param_dim == 2:  # Linear().weight
                output_grad = self.mc_weight[pointer] @ input_grad @ self.mc_weight[pointer+1]
                pointer += 2
            elif param_dim == 4:  # Conv2d().weight
                output_grad = torch.einsum('ijk,il->ljk',
                                           input_grad.flatten(start_dim=2),
                                           self.mc_weight[pointer])
                output_grad = self.mc_weight[pointer+1] @ output_grad @ self.mc_weight[pointer+2]
                pointer += 3
            else:
                raise NotImplementedError

            output_grads.append(output_grad.view_as(input_grad))

        return output_grads


class MetaCurvature(MetaLearningAlgBase):
    def __init__(self, args):
        super().__init__(args)
        self.mc = TensorModeProd(self.model.meta_parameters()).to(self.args.device)

    def meta_optimizer(self):
        return torch.optim.Adam([{'params': self.model.meta_parameters()},
                                 {'params': self.mc.parameters(), 'lr': self.args.meta_lr * .1}],
                                lr=self.args.meta_lr), None

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'mc': self.mc.state_dict()},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.mc.load_state_dict(state_dicts['mc'])

    def adaptation(self, x_supp, y_supp, first_order):
        params = OrderedDict(self.model.meta_named_parameters())

        for _ in range(self.args.task_iter):
            y_pred = self.model(x_supp, params=params)
            task_loss = self.args.loss_fn(y_pred, y_supp)
            grads = torch.autograd.grad(task_loss,
                                        params.values(),
                                        create_graph=not first_order)
            grads = self.mc(grads)

            for (name, param), grad in zip(params.items(), grads):
                params[name] = param - self.args.task_lr * grad

        return params
