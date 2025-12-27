# IN3063 Coursework — Code (Task 1: NumPy, Task 2: PyTorch)

This repo contains:
- **Task 1 (NumPy):** Fully-connected neural network from scratch on **CIFAR-10** (layers, backprop, dropout, SGD + momentum, evaluation scripts + plots).
- **Task 2 (PyTorch):** CNN experiments on **MNIST** (baseline + 3-layer CNN, CSV logging + plotting).

## Structure
```
layers/ (linear.py, relu.py, sigmoid.py, softmax.py, dropout.py)
dataset.py, NeuralNetwork.py, CrossEntropyLoss.py, optimisers.py
test_nn.py, run_activation_compare.py, run_part_g.py, optimiser_evaluation.py
ConvolutionalNeuralNetwork.py, 3layerCNN.py, CSVtoGraph.py
```

## Install
Python 3.9+
```bash
pip install numpy matplotlib pandas torch torchvision
```

## Datasets
### CIFAR-10 (Task 1)
Place the **CIFAR-10 python version** folder here:
```
./cifar-10-batches-py/
```
If a script can’t find CIFAR-10, set in that script:
```python
cifar_root = "cifar-10-batches-py"
```

### MNIST (Task 2)
Downloads automatically to `./data/` when you run the PyTorch scripts.

## Run Task 1 (NumPy)
Sanity checks:
```bash
python test_nn.py
```

ReLU vs Sigmoid (saves `activation_*.png`):
```bash
python run_activation_compare.py
```

Capacity + dropout (saves `capacity_*.png`):
```bash
python run_part_g.py
```

Optimiser comparison (shows plots / prints accuracies):
```bash
python optimiser_evaluation.py
```

## Run Task 2 (PyTorch)
Baseline CNN (writes CSV):
```bash
python ConvolutionalNeuralNetwork.py
```

3-layer CNN (writes CSV):
```bash
python 3layerCNN.py
```

Plot Task 2 CSV comparisons:
```bash
python CSVtoGraph.py
```
> If `CSVtoGraph.py` can’t find files, rename your CSVs or edit the filenames at the top of the script.

## Outputs
Task 1 plots (`activation_*.png`, `capacity_*.png`) are saved in the project root.  
Task 2 writes CSV logs (`*.csv`) which `CSVtoGraph.py` reads to plot comparisons.
## Authors
Coursework Group 1 Members:
- Ayesha
- Mohammed
- Yosias
- Jonathan
