# CaMo: Capturing the Modularity by End-to-End Models for Symbolic Regression

Pytorch implementation for the paper "CaMo: Capturing the Modularity by End-to-End Models for Symbolic Regression", submitted to the journal Knowledge-Based Systems, 2024.

## Installation
Please install Python 3.8.19 and install the environment via:
```
pip install -r requirements.txt
```

## Training
The config.json document contains all the hyperparameter settings.
Below is an example of running CaMo with benchmark Nguyen1.
First, modify the "expr_name" configuration as "nguyen1" (default setting), then run the command:
```
python main.py
```
