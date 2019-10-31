# Rethinking Binarized Neural Network Optimization

[![arXiv:1906.02107](https://img.shields.io/badge/cs.LG-arXiv%3A1906.02107-b31b1b.svg)](https://arxiv.org/abs/1906.02107) [![License: Apache 2.0](https://img.shields.io/github/license/plumerai/rethinking-bnn-optimization.svg)](https://github.com/plumerai/rethinking-bnn-optimization/blob/master/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Implementation for paper "[Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://arxiv.org/abs/1906.02107)"

**Note**: [Bop is now added to Larq](https://larq.dev/api/optimizers/#bop), the open source training library for BNNs. We recommend using the Larq implementation of Bop: it is compatible with more versions of TensorFlow and will be more actively maintained.

## Requirements

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.14+` or `2.0.0`
- [Larq](https://github.com/plumerai/larq) version `0.2.0`
- [Zookeeper](https://github.com/plumerai/zookeeper) version `0.1.1`

You can also check out one of our prebuilt [docker images](https://hub.docker.com/r/plumerai/deep-learning/tags).

## Installation

This is a complete Python module. To install it in your local Python environment, `cd` into the folder containing `setup.py` and run:

```
pip install -e .
```

## Train

To train a model locally, you can use the cli:

```
bnno train binarynet --dataset cifar10
```

## Reproduce Paper Experiments

### Hyperparameter Analysis (section 5.1)

To reproduce the runs exploring various hyperparameters, run:

```
bnno train binarynet \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop \
    --hparams threshold=1e-6,gamma=1e-3
```

where you use the appropriate values for threshold and gamma.

### CIFAR-10 (section 5.2)

To achieve the accuracy in the paper of 91.3%, run:

```
bnno train binarynet \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop_sec52 \
```

### ImageNet (section 5.3)

To reproduce the reported results on ImageNet, run:

```
bnno train alexnet --dataset imagenet2012 --hparams-set bop
bnno train xnornet --dataset imagenet2012 --hparams-set bop
bnno train birealnet --dataset imagenet2012 --hparams-set bop
```

This should give the following results:

| Network | Bop - top-1 accuracy |
|----|----|
|Binary Alexnet | 41.1% |
|XNOR-Net | 45.9% |
|Bi-Real Net | 56.6% |
