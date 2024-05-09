# MedIMeta for PyTorch

## Medical Imaging Meta Dataset

We introduce the Medical Imaging Meta-Dataset (MedIMeta), a novel multi-domain, multi-task meta-dataset designed to facilitate the development and standardised evaluation of ML models and cross-domain few-shot learning algorithms for medical image classification. MedIMeta contains 19 medical imaging datasets spanning 10 different domains and encompassing 54 distinct medical tasks, offering opportunities for both single-task and multi-task training. All tasks are standardised to the same format and readily usable in PyTorch or other ML frameworks. All datasets have been previously published with an open license that allows redistribution or we obtained an explicit permission to do so.

Each dataset within the MedIMeta dataset is standardized to a size of 224 × 224 pixels which matches image size commonly used in pre-trained models. Furthermore, the dataset comes with pre-made splits to ensure ease of use and standardized benchmarking.

You can see details about the MedIMeta dataset as well as download the data from
[Zenodo](https://zenodo.org/records/7884735).

## PyTorch Library

This library allows easy access to all tasks in the MedIMeta dataset as PyTorch datasets.
It provides a unified interface to the data and allows for easy usage in PyTorch.
The MedIMeta library integrates with
[TorchCross](https://www.github.com/StefanoWoerner/torchcross), a PyTorch library for
cross-domain learning, few-shot learning and meta-learning. It is therefore easy to
use the MedIMeta dataset in conjunction with TorchCross to perform cross-domain learning,
few-shot learning or meta-learning experiments.

**This library is still in alpha. The API is potentially subject to change. Any feedback
is welcome.**

## Installation

The toolbox can be installed via pip:

```bash
pip install medimeta
```

## Basic Usage

The MedIMeta dataset can be accessed via the `medimeta.MedIMeta` class, which extends the 
`torch.utils.data.Dataset` class. See the basic example below:

```python
from medimeta import MedIMeta

# Create the dataset for the Disease task of the OCT dataset, assuming
# the data is stored in the "data/MedIMeta" directory
dataset = MedIMeta("data/MedIMeta", "oct", "Disease")

# Get the first sample
sample = dataset[0]

print(sample[0].shape)
print(sample[1])
```

This will print the following:

```bash
torch.Size([1, 224, 224])
0
```


## Advanced Usage

MedIMeta builds on top of [TorchCross](https://www.github.com/StefanoWoerner/torchcross),
a library for cross-domain learning, few-shot learning and meta-learning in PyTorch.
MedIMeta can be used in conjunction with TorchCross to easily create cross-domain learning
or few-shot learning experiments. To this end, MedIMeta provides two convenience classes
for generating batches from multiple MedIMeta tasks and for generating few-shot insttances
of multiple MedIMeta tasks.

### Examples

See the [examples](examples) directory for examples on how to use MedIMeta in conjunction
with [TorchCross](https://www.github.com/StefanoWoerner/torchcross).
- [`imagenet_pretrained.py`](examples/imagenet_pretrained.py) shows how you can test
  pre-trained models on a few-shot instance of a MedIMeta task.
- [`cross_domain_pretraining.py`](examples/cross_domain_pretraining.py) shows how you
  can perform cross-domain pre-training on different MedIMeta tasks and then test the
  pre-trained model on a few-shot instance of a MedIMeta task.
- [`cross_domain_maml.py`](examples/cross_domain_maml.py) shows how you can perform
  cross-domain meta-learning with [MAML](https://arxiv.org/abs/1703.03400) on different
  MedIMeta tasks and then test the meta-learned model on multiple few-shot instances of a
  MedIMeta task.
- [`fully_supervised.py`](examples/fully_supervised.py) shows how you can perform
  fully-supervised learning on MedIMeta tasks by using the TorchCross `SimpleClassifier`.
