# rethinking-bnn-optimization

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


Implementation for paper "Latent Weights Do Not Exist: Rethinking Binary Neural Network Optimization"

## Requirements

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.13+` or `2.0.0`
- [Larq](https://github.com/plumerai/larq) version `0.2.0`
- [Zookeeper](https://github.com/plumerai/zookeeper) version `0.1.0`

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

### ImageNet (section 5.3)

