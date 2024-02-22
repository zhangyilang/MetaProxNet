import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.meta_alg_base import MetaLearningAlgBase
from src.meta_curvature import TensorModeProd


class PLFNet(nn.Module):
    def __init__(self, base_learner_params_fn, num_blk=None, num_pcs=5, pcs_range=2.):
        """
        Proximal network using piecewise linear functions
        All params are scaled by 1/task_lr for numerical stability
        :param base_learner_params_fn: function that returns base-learner parameters
        :param num_blk: task-level optimization steps, which is also the number of
        unrolling blocks; if None, params will be shared for each block of the
        unrolling NN
        :param num_pcs: number of pieces in PLF
        :param pcs_range: input axis of control points in PLF are fixed to uniformly
        partition the range [-pcs_range, pcs_range]
        """
        super().__init__()
        self.num_layers = len(list(base_learner_params_fn()))
        self.num_pcs = num_pcs
        self.spacing = 2 * pcs_range / num_pcs
        self.pcs_range = pcs_range

        PLF_init = torch.arange(-pcs_range,
                                pcs_range + self.spacing / 2,
                                self.spacing)
        self.ctrl_points = nn.ParameterList()
        if num_blk is None:
            num_blk = 1
        for _ in range(num_blk):
            for param in base_learner_params_fn():
                self.ctrl_points.append(nn.Parameter(PLF_init.repeat([*param.size(), 1])))

    def forward(self, params, blk=None):
        updated_params = OrderedDict()
        group_idx = blk * self.num_layers if blk is not None else 0

        for (key, param), ctrl_points in zip(params.items(),
                                             self.ctrl_points[group_idx:group_idx+self.num_layers]):
            with torch.no_grad():
                left_pcs_idx = torch.floor(param / self.spacing + self.num_pcs / 2).long()
                left_pcs_idx = torch.clamp(left_pcs_idx, 0, self.num_pcs-1)
                left_ctrl_mask = F.one_hot(left_pcs_idx, self.num_pcs+1)
                right_ctrl_mask = F.one_hot(left_pcs_idx+1, self.num_pcs+1)

            relative_weight = ((param + self.pcs_range) / self.spacing - left_pcs_idx).unsqueeze(-1)
            ctrl_mask = (1 - relative_weight) * left_ctrl_mask + relative_weight * right_ctrl_mask
            updated_params[key] = (ctrl_mask * ctrl_points).sum(dim=-1)

        return updated_params


class MetaProx(MetaLearningAlgBase):
    def __init__(self, args):
        super().__init__(args)

        num_blk = self.args.task_iter if not args.share_param else None
        self.proximal_net = PLFNet(self.model.meta_parameters, num_blk).to(self.args.device)

    def meta_optimizer(self):
        if self.args.dataset.lower() == 'tieredimagenet':
            optimizer = torch.optim.Adam([{'params': self.model.meta_parameters()},
                                         {'params': self.proximal_net.parameters()}],
                                         lr=self.args.meta_lr)
            lr_scheduler = None
        else:
            optimizer = torch.optim.SGD([{'params': self.model.meta_parameters()},
                                         {'params': self.proximal_net.parameters()}],
                                        lr=self.args.meta_lr,
                                        momentum=0.9,
                                        weight_decay=1e-4,
                                        nesterov=True)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000)

        return optimizer, lr_scheduler

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'proximal_net': self.proximal_net.state_dict()},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.proximal_net.load_state_dict(state_dicts['proximal_net'])

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
                params_shift[name] = param - grad

            if self.args.share_param:
                task_idx = None
            params_shift = self.proximal_net(params_shift, task_idx)

            for name, param in params.items():
                params[name] = params_init[name] + self.args.task_lr * params_shift[name]

        return params


class MetaProxMC(MetaLearningAlgBase):
    def __init__(self, args):
        super().__init__(args)

        num_blk = self.args.task_iter if not args.share_param else None
        self.proximal_net = PLFNet(self.model.meta_parameters, num_blk).to(self.args.device)
        self.mc = TensorModeProd(self.model.meta_parameters()).to(self.args.device)

    def meta_optimizer(self):
        if self.args.dataset.lower() == 'tieredimagenet':
            optimizer = torch.optim.Adam([{'params': self.model.meta_parameters()},
                                         {'params': self.proximal_net.parameters()},
                                         {'params': self.mc.parameters(), 'lr': self.args.meta_lr * .1}],
                                         lr=self.args.meta_lr)
            lr_scheduler = None
        else:
            optimizer = torch.optim.SGD([{'params': self.model.meta_parameters()},
                                         {'params': self.proximal_net.parameters()},
                                         {'params': self.mc.parameters(), 'lr': self.args.meta_lr * .1}],
                                        lr=self.args.meta_lr,
                                        momentum=0.9,
                                        weight_decay=1e-4,
                                        nesterov=True)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000)

        return optimizer, lr_scheduler

    def save_model(self, file_name):
        torch.save({'model': self.model.state_dict(),
                    'proximal_net': self.proximal_net.state_dict(),
                    'mc': self.mc.state_dict()},
                   os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        self.model.load_state_dict(state_dicts['model'])
        self.proximal_net.load_state_dict(state_dicts['proximal_net'])
        self.mc.load_state_dict(state_dicts['mc'])

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
            grads = self.mc(grads)

            for (name, param), grad in zip(params_shift.items(), grads):
                params_shift[name] = param - grad

            if self.args.share_param:
                task_idx = None
            params_shift = self.proximal_net(params_shift, task_idx)

            for name, param in params.items():
                params[name] = params_init[name] + self.args.task_lr * params_shift[name]

        return params


class LayerwisePLFNet(nn.Module):
    def __init__(self, num_layers, num_blk=None, num_pcs=5, pcs_range=1.):
        """
        Layerwise proximal network using piecewise linear functions
        :param num_layers: number of layers in base-learner
        :param num_blk: task-level optimization steps, which is also the number of
        unrolling blocks; if None, params will be shared for each block of the
        unrolling NN
        :param num_pcs: number of pieces in PLF
        :param pcs_range: input axis of control points in PLF are fixed to uniformly
        partition the range [-pcs_range, pcs_range]
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_pcs = num_pcs
        self.spacing = 2 * pcs_range / num_pcs
        self.pcs_range = pcs_range

        ctrl_point_groups = num_layers * num_blk if num_blk is not None else num_layers
        self.layerwise_ctrl_points = nn.ParameterList([
            nn.Parameter(torch.arange(-pcs_range,
                                      pcs_range + self.spacing / 2,
                                      self.spacing))
            for _ in range(ctrl_point_groups)
        ])

    def forward(self, params, blk=None):
        updated_params = OrderedDict()
        group_idx = blk * self.num_layers if blk else 0

        for (key, param), ctrl_points in zip(params.items(),
                                             self.layerwise_ctrl_points[group_idx:group_idx+self.num_layers]):
            with torch.no_grad():
                left_pcs_idx = torch.floor(param / self.spacing + self.num_pcs / 2).long()
                left_pcs_idx = torch.clamp(left_pcs_idx, 0, self.num_pcs-1)

            relative_weight = (param + self.pcs_range) / self.spacing - left_pcs_idx
            updated_params[key] = (1 - relative_weight) * ctrl_points[left_pcs_idx] \
                                  + relative_weight * ctrl_points[left_pcs_idx+1]

        return updated_params


class LayerwiseMetaProx(MetaProx):
    def __init__(self, args):
        MetaLearningAlgBase.__init__(self, args)

        num_layer = len(list(self.model.meta_parameters()))
        num_blk = self.args.task_iter if not args.share_param else None
        self.proximal_net = LayerwisePLFNet(num_layer, num_blk).to(self.args.device)
