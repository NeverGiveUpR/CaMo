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
