# CaMo: Capturing the Modularity by End-to-End Models for Symbolic Regression

Pytorch implementation for the paper "CaMo: Capturing the Modularity by End-to-End Models for Symbolic Regression", submitted to the journal Knowledge-Based Systems, 2024.

## Installation
Create a new python environment by conda via:
```
conda create -n CaMo python=3.8.19
```
To activate this environment, use:
```
conda activate CaMo
```
or,
```
source activate CaMo
```
Then install the dependency package via:
```
pip install -r requirements.txt
```

## Training
Run the following command will conduct experiment on benchmark "nguyen2":
```
python main.py --benchmark nguyen2
```
If you want to conduct experiments on other benchmarks, please modify the benchmark_name:
```
python main.py --benchmark benchmark_name
```
Conduct this command may take a few hours to run the procedure. In each iteration, the best expression will be reported along with its reward, used module. After training, the mse, rmse, r2, and complexity of the searched best expression will be displayed. Note that in the first iteration, no module is used, thus the reported module is empty. After few iterations, better expressions with modules will be reported.
