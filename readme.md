# MetaProx

Implementation of paper “[Meta-Learning Priors Using Unrolled Proximal Networks](https://openreview.net/forum?id=b3Cu426njo)” (accepted by ICLR 2024). 

## Preparation

Codes tested under the following environment:

---

- PyTorch 1.9.1
- CUDA 10.2
- CUDNN 7.6.5
- torchvision 1.10.1
- torch-utils 0.1.2
- [torchmeta](https://github.com/tristandeleu/pytorch-meta) 1.8.0

---

One can use the following commands to setup the enviroment:

```shell
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
pip install torchmeta torch-utils
```

To prepare the datasets, follow the instructions of [Torchmeta](https://github.com/tristandeleu/pytorch-meta).

## Experiments

Default experimental setups can be found in `main.py`. To carry out the numerical test, use the commands

```shell
python main.py "--arguments" "values"
```

where `arguments` and `values` are the algorithm parameters that you want to alter.

To reproduce the results reported in our paper, please use the scripts provided in `bash_scripts.sh`. 

## Citation

> Y. Zhang, and G. B. Giannakis, "Meta-Learning Priors Using Unrolled Proximal Networks," in *Proceedings of International Conference on Learning Representations*, Vienna, Austria, Map 7-11, 2024.

```tex
@inproceedings{iBaML, 
  author={Zhang, Yilang and Giannakis, Georgios B.}, 
  title={Meta-Learning Priors Using Unrolled Proximal Networks}, 
  booktitle={International Conference on Learning Representations}, 
  year={2024}, 
  url={https://openreview.net/forum?id=b3Cu426njo},
}
```