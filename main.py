from train_self_update import train
import numpy as np
import pandas as pd
import torch
import OtherEquations as OE
from sklearn.model_selection import train_test_split
from module import UpdateModule
import json
from utils import recursively_convert_to_namespace, namespace_to_dict
import copy

np.random.seed(2025)

def get_data(expr_name, dataset=None):
    if dataset == 'OE':
        X, y = OE.func_dict[expr_name]()
    elif dataset == 'blackbox':
        path = '/home/liujingyi/CaMo/blackbox/datasets/{}.tsv.gz'.format(expr_name)
        data = pd.read_csv(path, sep='\t', compression='gzip').values
        X = np.array(data[:, :-1], dtype=float)
        y = np.array(data[:, -1], dtype=float)
    else:
        print("dataset name error!")
        return 

    return X, y

def get_all_expr_name(dataset):
    if dataset == 'OE':
        expr_names = OE.func_dict.keys()
    elif dataset == 'blackbox':
        path = '/home/liujingyi/CaMo/blackbox/Penn Machine Learning Benchmarks.csv'
        expr_names = pd.read_csv(path, header=0, index_col=0)
        expr_names = list(expr_names.index)
    else:
        print("dataset name error!")
        return 
    return expr_names

def execute(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        var_num=None,
        config=None
):
    # initialize operator list
    # pdb.set_trace()
    operator_list = copy.deepcopy(config.training.operator_list)
    print("config.module.module_number:", config.module.module_number)
    modules = UpdateModule(config.module.module_number, config.module.max_length, config.module.max_arity) 
    modules.modules = {'M'+str(i+1): {'traversal': [], 'arity':0} for i in range(config.module.module_number)}

    var_string = []
    for i in range(var_num):
        var_string.append('var_x'+str(i+1))
    operator_list.extend(var_string)

    results = train(X_constants, y_constants, X_rnn, y_rnn, modules, operator_list, config)
    del modules, operator_list

    return {'best_reward': results[0],
            'best_expression': results[1],
            'best_sequence': results[2],
            'best_modules': results[3],
            'best_modules_string': results[4],
            'time_cost': results[5],
            'epoches': results[6]}



if __name__=='__main__':

    # load config.json
    with open('./config.json', encoding='utf-8') as f:
        config_dict = json.load(f)
        config = recursively_convert_to_namespace(config_dict)

    expr_name = config.dataset.expr_name

    print("-----start training expr {}...".format(expr_name))
    dataset = config.dataset
    print("dataset.dataset_name:", dataset.dataset_name)
    # split train and test
    if dataset.dataset_name == 'blackbox':
        test_size = 0.25
        X, y = get_data(expr_name, dataset=dataset.dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        X_constants, X_rnn = torch.Tensor(X_train), torch.Tensor(X_train)
        y_constants, y_rnn = torch.Tensor(y_train), torch.Tensor(y_train)
    else:
        X_train, y_train = get_data(expr_name, dataset=dataset.dataset_name)
        X_test, y_test = get_data(expr_name, dataset=dataset.dataset_name)
    
        # in the train set, use 20% data points to optimize constants
        X_constants, X_rnn = torch.Tensor(X_train[:int(0.2*X_train.shape[0])]), torch.Tensor(X_train[int(0.2*X_train.shape[0]):])
        y_constants, y_rnn = torch.Tensor(y_train[:int(0.2*X_train.shape[0])]), torch.Tensor(y_train[int(0.2*X_train.shape[0]):])
        # print("X_constants:", X_constants.shape, "X_rnn:", X_rnn.shape, "y_constants:", y_constants.shape, "y_rnn:", y_rnn.shape)

    
    results = execute(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        var_num = X_constants.shape[1],
        config=config
    )
    best_expr = results['best_expression']
    time_cost = results['time_cost']
    end_epoch = results['epoches']
    best_sequence = results['best_sequence']
    best_modules = results['best_modules']
    best_modules_string = results['best_modules_string']

    # print("*********************************************")
    # print("best_expression:", best_expr)
    # print("best_sequence:", best_sequence)
    # print("best_modules_string:", best_modules_string)
    # print("time_cost:", time_cost)
    # print("*********************************************")
