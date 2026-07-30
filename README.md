# PyTorch Series
Notes and hands-on code from my PyTorch learning journey — tensors, operations, autograd, and building neural networks from scratch.

## Contents
- `01_intro_and_tensors.ipynb` — Introduction to PyTorch, tensor creation, shapes, dtypes, math operations, indexing, reshaping, and broadcasting.
- `02_autograd.ipynb` — Autograd deep dive: computational graphs, leaf vs non-leaf tensors, manual vs automatic gradients, gradient accumulation, and disabling gradient tracking.
- `03_manual_training_pipeline.ipynb` — Full training pipeline on the Breast Cancer dataset built from scratch: data loading, scaling, encoding, a single neuron defined with raw tensors, manual forward pass, manual Binary Cross-Entropy loss, and manual gradient descent (no `nn.Module`, no built-in loss).
- `04_nn_module.ipynb` — Building neural networks with `torch.nn`: three progressively deeper models (single neuron → one hidden layer → two hidden layers) using `nn.Module` and `nn.Sequential`, plus inspecting architectures with `torchinfo`.
- `05_training_pipeline_using_nn_module.ipynb` — The same Breast Cancer training pipeline rebuilt with PyTorch's built-in tools: `nn.Linear`, `nn.BCELoss`, and `torch.optim.SGD`, replacing the manual math from notebook 03.
- `06_dataset_and_dataloader.ipynb` — Working with PyTorch's `Dataset` and `DataLoader` classes: wrapping raw data into a custom `Dataset`, batching, shuffling, and iterating through data the proper PyTorch way instead of manual loops.
- `07_training_pipeline_nn_module_with_dataloader.ipynb` — The nn.Module training pipeline from notebook 05, updated to pull data through a `DataLoader` instead of feeding the whole dataset in one go.
- `08_ann_build_fashion_mnist.ipynb` — Building and training an artificial neural network on the Fashion MNIST dataset (`fmnist_small.csv`), covering the full pipeline from raw pixel data to a trained classifier.

## Author
**Ali Hamza**
[LinkedIn](https://www.linkedin.com/in/alihamzastudent/)
