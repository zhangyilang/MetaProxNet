from abc import ABC, abstractmethod
from src.models import FourBlkCNN
from src.utils import split_task_batch, Checkpointer
import numpy as np
import torch


class MetaLearningAlgBase(ABC):
    @abstractmethod
    def __init__(self, args):
        self.args = args
        model = FourBlkCNN(args.num_way, hidden_size=args.num_filter, num_feat=25*args.num_filter)
        self.model = model.to(args.device)

    @abstractmethod
    def meta_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, file_name):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, file_name):
        raise NotImplementedError

    @abstractmethod
    def adaptation(self, x_supp, y_supp, first_order):
        raise NotImplementedError

    def train(self, train_dataloader, val_dataloader):
        print('Training starts ...')

        meta_optimizer, lr_scheduler = self.meta_optimizer()
        check_pointer = Checkpointer(self.save_model, self.args.algorithm.lower())

        running_loss = 0.
        running_acc = 0.

        # training loop
        for meta_idx, task_batch in enumerate(train_dataloader):
            if meta_idx >= self.args.meta_iter:
                break
            self.model.train()
            meta_optimizer.zero_grad()

            for x_supp, y_supp, x_qry, y_qry in zip(*split_task_batch(task_batch, self.args.device)):
                params = self.adaptation(x_supp, y_supp, first_order=self.args.first_order)
                y_pred = self.model(x_qry, params=params)
                meta_loss = self.args.loss_fn(y_pred, y_qry)
                meta_loss.backward()

                with torch.no_grad():
                    running_loss += meta_loss.detach().item()
                    running_acc += (y_pred.argmax(dim=1) == y_qry).detach().float().mean().item()

            meta_optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # meta-validation
            if (meta_idx + 1) % self.args.log_iter == 0:
                val_loss, val_acc = self.evaluate(val_dataloader, self.args.num_log_tasks)
                print('Meta-iter {0:d}: '
                      'train loss = {1:.3f}, train acc = {2:.2f}%, '
                      'val loss = {3:.3f}, val acc = {4:.1f}%'
                      .format(meta_idx + 1,
                              running_loss / (self.args.log_iter * self.args.batch_size),
                              running_acc / (self.args.log_iter * self.args.batch_size) * 100,
                              val_loss, val_acc * 100
                              )
                      )

                running_loss = 0.
                running_acc = 0.

            # save
            if (meta_idx + 1) % self.args.save_iter == 0:
                val_loss, val_acc = self.evaluate(val_dataloader, self.args.num_val_tasks)
                check_pointer.update(val_acc)
                print('Checkpoint {0:d}: '
                      'val loss = {1:.4f}, val acc = {2:.2f}%'
                      .format(check_pointer.counter, val_loss, val_acc * 100)
                      )

    def test(self, test_dataloader):
        print('Testing starts ...')

        loss_mean, loss_std, acc_mean, acc_std = self.evaluate(test_dataloader, self.args.num_ts_tasks, return_std=True)
        print('Test: nll = {0:.4f} +/- {1:.4f}, '
              'acc = {2:.2f}% +/- {3:.2f}%'
              .format(loss_mean, 1.96 * loss_std / np.sqrt(self.args.num_ts_tasks),
                      acc_mean * 100, 196 * acc_std / np.sqrt(self.args.num_ts_tasks))
              )

    def evaluate(self, dataloader, num_tasks, return_std=False):
        self.model.eval()   # this has no effect on BN layers since track_running_status=False (transductive setting)
        loss_list = list()
        acc_list = list()

        for eval_idx, task_batch in enumerate(dataloader):
            if eval_idx >= num_tasks:
                break

            for x_supp, y_supp, x_qry, y_qry in zip(*split_task_batch(task_batch, self.args.device)):
                params = self.adaptation(x_supp, y_supp, first_order=True)
                with torch.no_grad():
                    y_pred = self.model(x_qry, params=params)
                    loss_list.append(self.args.loss_fn(y_pred, y_qry).item())
                    acc_list.append((y_pred.argmax(dim=1) == y_qry).float().mean().item())

        if return_std:
            return np.mean(loss_list), np.std(loss_list), np.mean(acc_list), np.std(acc_list)
        else:
            return np.mean(loss_list), np.mean(acc_list)
