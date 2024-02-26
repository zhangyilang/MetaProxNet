import os
import argparse
import random
import torch
import numpy as np
from torchmeta.datasets import MiniImagenet, TieredImagenet, Omniglot, CUB
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from src.maml import MAML
from src.meta_prox import MetaProx, MetaProxMC
from src.meta_pgd import MetaPGDGaussian
from src.meta_sgd import MetaSGD
from src.meta_curvature import MetaCurvature
from src.others import MetaProxMetaSGD


def main(args):
    suffix = '-' + str(args.num_way) + 'way' + str(args.num_supp) + 'shot'
    args.model_dir = os.path.join(args.model_dir, args.dataset.lower(), args.algorithm.lower() + suffix)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"   # for CUDA >= 10.2
    torch.use_deterministic_algorithms(True)

    transform = ToTensor()
    class_aug = None

    dataset = args.dataset.lower()
    if dataset == 'miniimagenet':
        dataset = MiniImagenet
    elif dataset == 'tieredimagenet':
        dataset = TieredImagenet
    elif dataset == 'omniglot':
        dataset = Omniglot
        transform = Compose([Resize([28, 28]), ToTensor()])
        class_aug = [Rotation([90, 180, 270])]
    elif dataset == 'cub':
        dataset = CUB
        transform = Compose([Resize([84, 84]), ToTensor()])
    else:
        raise NotImplementedError

    class_splitter_train = ClassSplitter(shuffle=True, num_train_per_class=args.num_supp,
                                         num_test_per_class=args.num_qry)
    class_splitter_eval = ClassSplitter(shuffle=True, num_train_per_class=args.num_supp,
                                        num_test_per_class=args.num_supp)

    train_dataset = dataset(args.data_dir,
                            num_classes_per_task=args.num_way,
                            transform=transform,
                            target_transform=Categorical(num_classes=args.num_way),
                            dataset_transform=class_splitter_train,
                            class_augmentations=class_aug,
                            meta_train=True,
                            download=args.download)
    val_dataset = dataset(args.data_dir,
                          num_classes_per_task=args.num_way,
                          transform=transform,
                          target_transform=Categorical(num_classes=args.num_way),
                          dataset_transform=class_splitter_eval,
                          class_augmentations=class_aug,
                          meta_val=True,
                          download=args.download)
    test_dataset = dataset(args.data_dir,
                           num_classes_per_task=args.num_way,
                           transform=transform,
                           target_transform=Categorical(num_classes=args.num_way),
                           dataset_transform=class_splitter_eval,
                           class_augmentations=class_aug,
                           meta_test=True,
                           download=args.download)

    train_dataset.seed(args.seed)
    val_dataset.seed(args.seed)
    test_dataset.seed(args.seed)

    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=1, num_workers=1)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=1)

    for k, v in args.__dict__.items():
        print('%s: %s' % (k, v))

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    args.loss_fn = torch.nn.CrossEntropyLoss()

    alg = args.algorithm.lower()
    if alg == 'metaprox' or alg == 'metaproxmaml':
        alg = MetaProx(args)
    elif alg == 'maml':
        alg = MAML(args)
    elif alg == 'metapgd-gaussian':
        alg = MetaPGDGaussian(args)
    elif alg == 'metasgd':
        alg = MetaSGD(args)
    elif alg == 'metaproxmetasgd':
        alg = MetaProxMetaSGD(args)
    elif alg == 'metacurvature':
        alg = MetaCurvature(args)
    elif alg == 'metaproxmc':
        alg = MetaProxMC(args)
    else:
        raise NotImplementedError

    alg.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    alg.load_model(args.algorithm.lower() + '_final.ct')
    alg.test(test_dataloader=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup variables')

    # dir
    parser.add_argument('--data-dir', type=str, default='./datasets/', help='Dataset directory')
    parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='Dataset')
    parser.add_argument('--download', type=bool, default=False, help='Whether to download the dataset')
    parser.add_argument('--num-way', type=int, default=5, help='Number of classes per task')
    parser.add_argument('--num-supp', type=int, default=5, help='Number of data per class (aka. shot) in support set')
    parser.add_argument('--num-qry', type=int, default=15, help='Number of data per class in query set')
    parser.add_argument('--num-val-tasks', type=int, default=1000, help='Number of tasks for meta-validation')
    parser.add_argument('--num-ts-tasks', type=int, default=1000, help='Number of tasks for meta-test')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')

    # algorithm
    parser.add_argument('--algorithm', type=str, default='MetaProxMC', help='Few-shot learning methods')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use cuda')

    # meta training params
    parser.add_argument('--first-order', type=bool, default=False, help='Whether to use first-order approximation')
    parser.add_argument('--meta-iter', type=int, default=60000, help='Number of epochs for meta training')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (episodes) of tasks to update meta-param')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--log-iter', type=int, default=200, help='Log iter')
    parser.add_argument('--num-log-tasks', type=int, default=100, help='Log tasks')
    parser.add_argument('--save-iter', type=int, default=2000, help='Save iter')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-updates')

    # task training params
    parser.add_argument('--task-iter', type=int, default=5, help='Adaptation steps during meta-training')
    parser.add_argument('--task-lr', type=float, default=1e-2, help='Learning rate for task-updates')
    parser.add_argument('--share-param', type=bool, default=False, help='Whether to share parameters for different '
                                                                        'blocks of unrolling NN')
    parser.add_argument('--num-filter', type=int, default=128, help='Number of filters per layer in CNN')

    args = parser.parse_args()

    main(args)
