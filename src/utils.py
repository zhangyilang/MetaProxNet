import torch


def split_task_batch(task_batch, device):
    x_supp_batch, y_supp_batch = task_batch['train']
    x_qry_batch, y_qry_batch = task_batch['test']

    x_supp_batch = x_supp_batch.to(device=device)
    y_supp_batch = y_supp_batch.to(device=device)
    x_qry_batch = x_qry_batch.to(device=device)
    y_qry_batch = y_qry_batch.to(device=device)

    return x_supp_batch, y_supp_batch, x_qry_batch, y_qry_batch


def elementwise_Softshrink(param, lambd):
    ind_pos = param > lambd
    ind_neg = param < -lambd

    param_new = torch.zeros_like(param)
    param_new[ind_pos] -= lambd[ind_pos]
    param_new[ind_neg] += lambd[ind_neg]

    return param_new


class Checkpointer:
    def __init__(self, save_fn, alg_name):
        self.save_fn = save_fn
        self.alg_name = alg_name
        self.counter = 0
        self.best_acc = 0

    def update(self, acc):
        self.counter += 1
        self.save_fn(self.alg_name + '_{0:02d}.ct'.format(self.counter))

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_fn(self.alg_name + '_final.ct'.format(self.counter))
