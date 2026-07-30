# PyTorch Series

Notes and hands-on code from my PyTorch learning journey — tensors, operations, autograd, and building neural networks from scratch.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

## What this is

A running log of notebooks as I go from raw tensors to actual neural networks in PyTorch. Each notebook builds on the last — starting with everything done by hand, then swapping in PyTorch's built-in tools once the manual version makes sense.

## Contents

| # | Notebook | What's in it |
|---|----------|---------------|
| 01 | [`intro_and_tensors`](01_intro_and_tensors.ipynb) | Tensor creation, shapes, dtypes, math ops, indexing, reshaping, broadcasting |
| 02 | [`autograd`](02_autograd.ipynb) | Computational graphs, leaf vs non-leaf tensors, manual vs auto gradients, gradient accumulation, disabling tracking |
| 03 | [`manual_training_pipeline`](03_manual_training_pipeline.ipynb) | Full training loop on the Breast Cancer dataset, built from raw tensors — manual forward pass, manual BCE loss, manual gradient descent |
| 04 | [`nn_module`](04_nn_module.ipynb) | Same problem, three model depths (single neuron → 1 hidden layer → 2 hidden layers) using `nn.Module` / `nn.Sequential`, inspected with `torchinfo` |
| 05 | [`training_pipeline_using_nn_module`](05_training_pipeline_using_nn_module.ipynb) | Notebook 03's pipeline rebuilt with `nn.Linear`, `nn.BCELoss`, `torch.optim.SGD` |
| 06 | [`dataset_and_dataloader`](06_dataset_and_dataloader.ipynb) | Custom `Dataset` class, batching and shuffling with `DataLoader` |
| 07 | [`training_pipeline_nn_module_with_dataloader`](07_training_pipeline_nn_module_with_dataloader.ipynb) | Notebook 05's pipeline updated to pull data through a `DataLoader` |
| 08 | [`ann_build_fashion_mnist`](08_ann_build_fashion_mnist.ipynb) | Full ANN trained on Fashion MNIST (`fmnist_small.csv`) — raw pixels to a trained classifier |

## Dataset

`fmnist_small.csv` — a smaller version of Fashion MNIST used in notebook 08.

## Running locally

\`\`\`bash
git clone https://github.com/Ali-Hamza-developer/pytorch.git
cd pytorch
pip install torch torchvision torchinfo pandas scikit-learn jupyter
jupyter notebook
\`\`\`

## Author

**Ali Hamza**
[LinkedIn](https://www.linkedin.com/in/alihamzastudent/)
