operators = ['+', '-', '*', '/', '^', 'sin', 'cos', 'exp', 'log', 'arcsin', 'tanh', 'tan', 'arccos', 'sqrt', 'square', 'cube', 'quartic', 'quintic','neg']
variables = ['x1','x2','x3','x4','x5','x6', 'x7', 'x8', 'x9']
constants = ['1', '2', '3', '4', '5', 'pi', 'c']

arity_dict = {'+':2, '-':2, '*':2, '/':2, '^':2, 
                'sin':1, 'cos':1, 'exp':1, 'log':1, 'arcsin':1, 'tanh':1, 'tan':1, 'arccos':1,
                'sqrt':1, 'square':1, 'cube':1, 'quartic':1, 'quintic':1, 'neg':1,
                'AND':2, 'OR':2, 'NOT':1, 'NOR':2, 'NAND':2, # logit
                'x1':0, 'x2':0, 'x3':0, 'x4':0, 'x5':0, 'x6':0, 'x7':0, 'x8':0, 'x9':0,
                '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '8':0, 'pi':0, '0.5':0, '0.6':0, '1.5':0, '0.25':0, 'var':0, 'c':0,
                'var_x1':0, 'var_x2':0, 'var_x3':0, 'var_x4':0, 'var_x5':0, 'var_x6':0, 'var_x7':0, 'var_x8':0, 'var_x9':0, 'var_x10':0, 'var_x11':0,
                'var_x':0}

arity_dict_ = {
    '\+':2, '\-':2, '\*':2, '\/':2, '\^':2, 
                'sin':1, 'cos':1, 'exp':1, 'log':1, 'tanh':1, 'tan':1, 
                'sqrt':1, 'square':1
}

from types import SimpleNamespace

def recursively_convert_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{key: recursively_convert_to_namespace(value) for key, value in d.items()})
    return d

def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    return obj

from sympy import simplify, preorder_traversal

def expr_complex(expr):
    try:
        expr = simplify(expr)
        c=0
        for arg in preorder_traversal(expr):
            c += 1
        return c
    except:
        c = 100000
        return c
    
def simplicity_in_srbench(expr):
    try:
        expr = simplify(expr)
        num_components=0
        for arg in preorder_traversal(expr):
            num_components += 1
    except:
        num_components = 1000
    simplicity = -np.round(np.log(num_components)/np.log(5), 1)
    return simplicity
    
    
import re

def expr_complex_(expr):
    expr = str(expr)
    expr = expr.replace('**', '^')
    # print("replaced expr:", expr)
    cmplx = 0
    for key in arity_dict_.keys():
        # print("key:", key)
        # print(re.findall(str(key), expr))
        
        cmplx += len(re.findall(key, expr))*arity_dict_[key]
    return cmplx

import numpy as np
from numpy import cos, sin, tan
def RE(y_pred, y):
    return np.mean(np.abs((y-y_pred)/y))

# rewrite the sqrt function, protected
def sqrt(x):
    return np.sqrt(np.abs(x)+1e-5)

def log(x):
    return np.log(np.abs(x)+1e-5)

def exp(x):
    return np.clip(np.exp(x), 0, 1e5)

def div(x1, x2):
    return x1/(x2+1e-5)

def mul(x1, x2):
    return x1*x2

def sub(x1, x2):
    return x1-x2

def add(x1, x2):
    return x1+x2

def Abs(x):
    return np.abs(x)

def square(x):
    return x**2

from sklearn.metrics import r2_score
import pdb

def calculate_metric(expr, X, y):
    simplicity = simplicity_in_srbench(expr)
    complexity = expr_complex_(expr)

    # pdb.set_trace()
    expr = expr.replace(' ', '')
    expr = expr.replace('^', '**')
    expr = expr.replace('ln', 'log')

    y_pred = []                
    for x_i, y_i in zip(X, y):
        # substitute the value of X
        expTmp = expr
        for j in range(X.shape[1]):
            # print("j:", j)
            expTmp = expTmp.replace('x{}'.format(j+1), str(x_i[j]))
        
        try:
            exp_Tmp_y = eval(expTmp)
            y_pred.append(exp_Tmp_y)
        except:
            y_pred.append(10000)

    y_pred = np.array(y_pred)
    y_pred = np.nan_to_num(y_pred, 10000)

    try:
        r2 = r2_score(y_pred, y)
    except:
        r2 = 0
    
    mse = np.mean(np.square(y_pred-y))
    rmse = np.sqrt(mse)

    return mse, rmse, r2, simplicity, complexity
