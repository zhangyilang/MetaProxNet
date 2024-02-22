#!/bin/bash

# 5-class(way) miniImageNet (change num-supp to 5 for 5-shot tests)
# MetaProx + MAML
python main.py --algorithm MetaProx --dataset miniImageNet --num-supp 1 --filter-size 64 \
      > log/miniimagenet/MetaProx-5way1shot.log
# MetaProx + MetaCurvature (note: MetaCurvature increases #channels to 128)
python main.py --algorithm MetaProxMC --dataset miniImageNet --num-supp 1 --filter-size 128 \
      > log/miniimagenet/MetaProxMC-5way1shot.log

# 5-class(way) TieredImageNet
# MetaProx + MAML
python main.py --algorithm MetaProx --dataset TieredImageNet --num-supp 1 --filter-size 64 \
      > log/tieredimagenet/MetaProx-5way1shot.log
# MetaProx + MetaCurvature
python main.py --algorithm MetaProxMC --dataset TieredImageNet --num-supp 1 --filter-size 128 \
      > log/tieredimagenet/MetaProxMC-5way1shot.log
