#!/bin/bash

# 5-class(way) miniImageNet (change num-supp to 5 for 5-shot tests)
# MetaProx + MAML
python main.py --algorithm MetaProxNet --dataset miniImageNet --num-supp 1 --filter-size 64 \
      > log/miniimagenet/MetaProxNet-5way1shot.log
# MetaProx + MetaCurvature (note: MetaCurvature increases #channels to 128)
python main.py --algorithm MetaProxNetMC --dataset miniImageNet --num-supp 1 --filter-size 128 \
      > log/miniimagenet/MetaProxNetMC-5way1shot.log

# 5-class(way) TieredImageNet
# MetaProx + MAML
python main.py --algorithm MetaProxNet --dataset TieredImageNet --num-supp 1 --filter-size 64 \
      > log/tieredimagenet/MetaProxNet-5way1shot.log
# MetaProx + MetaCurvature
python main.py --algorithm MetaProxNetMC --dataset TieredImageNet --num-supp 1 --filter-size 128 \
      > log/tieredimagenet/MetaProNetxMC-5way1shot.log
