# PyTorch Learning Series

![PyTorch Learning Series Banner](download.jpg)

Notes and hands-on code from my PyTorch learning journey—from tensors and autograd to building, optimizing, and training deep learning models with PyTorch.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-success)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blueviolet)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## What this is

A collection of notebooks documenting my PyTorch learning journey. Starting from tensor fundamentals, the series gradually introduces automatic differentiation, neural network construction, efficient training pipelines, GPU acceleration, hyperparameter optimization, convolutional neural networks, transfer learning, and sequence modeling with LSTMs.

Each notebook builds on concepts introduced in previous notebooks, making the repository suitable for anyone learning PyTorch step by step.

## Contents

| #  | Notebook                                                                                                    | What's in it                                                                                |
| -- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 01 | [`intro_and_tensors`](01_intro_and_tensors.ipynb)                                                           | Tensor creation, indexing, reshaping, broadcasting, tensor operations, dtypes               |
| 02 | [`autograd`](02_autograd.ipynb)                                                                             | Computational graphs, automatic differentiation, gradient accumulation, disabling gradients |
| 03 | [`manual_training_pipeline`](03_manual_training_pipeline.ipynb)                                             | Manual forward pass, BCE loss, backpropagation and gradient descent from scratch            |
| 04 | [`nn_module`](04_nn_module.ipynb)                                                                           | Building neural networks with `nn.Module` and `nn.Sequential`                               |
| 05 | [`training_pipeline_using_nn_module`](05_training_pipeline_using_nn_module.ipynb)                           | Training models using `nn.Linear`, optimizers and loss functions                            |
| 06 | [`dataset_and_dataloader`](06_dataset_and_dataloader.ipynb)                                                 | Creating custom datasets, batching and shuffling with `DataLoader`                          |
| 07 | [`training_pipeline_nn_module_with_dataloader`](07_training_pipeline_nn_module_with_dataloader.ipynb)       | Complete training pipeline using `Dataset` and `DataLoader`                                 |
| 08 | [`ann_build_fashion_mnist`](08_ann_build_fashion_mnist.ipynb)                                               | Building an ANN for Fashion MNIST image classification                                      |
| 09 | [`ann_fashion_mnist_pytorch_gpu`](09_ann_fashion_mnist_pytorch_gpu.ipynb)                                   | Training the ANN on GPU using CUDA                                                          |
| 10 | [`ann_fashion_mnist_pytorch_gpu_optimized`](10_ann_fashion_mnist_pytorch_gpu_optimized.ipynb)               | Optimizing model architecture and training performance                                      |
| 11 | [`ann_fashion_mnist_pytorch_gpu_optimized_optuna`](11_ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb) | Hyperparameter tuning using Optuna                                                          |
| 12 | [`cnn_fashion_mnist_pytorch_gpu`](12_cnn_fashion_mnist_pytorch_gpu.ipynb)                                   | Image classification using Convolutional Neural Networks                                    |
| 13 | [`transfer_learning_fashion_mnist_pytorch_gpu`](13_transfer_learning_fashion_mnist_pytorch_gpu.ipynb)       | Transfer Learning with pretrained CNN models                                                |
| 14 | [`pytorch_lstm_next_word_predictor`](14_pytorch_lstm_next_word_predictor.ipynb)                             | Building an LSTM-based next-word prediction model                                           |

## Datasets

* **fmnist_small.csv** — Reduced Fashion MNIST dataset used for ANN and CNN notebooks.
* **100_Unique_QA_Dataset.csv** — Dataset used for the LSTM next-word prediction notebook.

## Running locally

```bash
git clone https://github.com/Ali-Hamza-developer/pytorch.git
cd pytorch

pip install torch torchvision torchinfo optuna pandas numpy scikit-learn matplotlib jupyter

jupyter notebook
```

## Learning Progress

* ✅ PyTorch Fundamentals
* ✅ Autograd
* ✅ Manual Training Pipeline
* ✅ Neural Networks
* ✅ Dataset & DataLoader
* ✅ ANN
* ✅ GPU Training
* ✅ Model Optimization
* ✅ Optuna Hyperparameter Tuning
* ✅ CNN
* ✅ Transfer Learning
* ✅ LSTM

## Author

**Ali Hamza**

Machine Learning • Deep Learning • PyTorch

* GitHub: https://github.com/Ali-Hamza-developer
* LinkedIn: https://www.linkedin.com/in/alihamzastudent/

⭐ If you find this repository useful, consider giving it a star!
