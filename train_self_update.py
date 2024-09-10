import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from operators import Operators
from rnn import DSRRNN_MULTI
from expression_utils import *
from collections import Counter
import copy
from utils import recursively_convert_to_namespace, namespace_to_dict
from module import UpdateModule 
# from GP_controller import GPController
# from program import Program
import time
from sklearn.metrics import r2_score
import pdb

###############################################################################
# Main Training loop
###############################################################################

def train(X_constants, y_constants, X_rnn, y_rnn, modules, operator_list, config):

    epoch_best_rewards = []
    epoch_best_expressions = []
    epoch_best_sequences = []
    start_time = time.time()


    kwargs = config.training
    # Initialize operators, RNN, and optimizer
    # Establish GPU device if necessary
    if (kwargs.use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # pdb.set_trace()
    operators = Operators(operator_list, modules, device)

    dsr_rnn = DSRRNN_MULTI(operators, modules, device, config).to(device)
    if (kwargs.optimizer == 'adam'):
        optim = torch.optim.Adam(dsr_rnn.parameters(), lr=kwargs.lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=kwargs.lr)

    # total_params = sum(param.numel() for param in dsr_rnn.parameters())
    # print(f"Total number of parameters: {total_params}")
    
    expr_cache = {}

    # Best expression and its performance
    best_expression, best_sequence, best_modules, best_performance = None, None, None, float('-inf')

    # the first epoch, has no module to use
    sequences, sequence_lengths, entropies, log_probabilities = dsr_rnn.sample_sequence(kwargs.initial_batch_size, use_module=False, device=device)
    sequences_int = sequences
    sequences_lengths = sequence_lengths
    strings = modules.turn_sequence_to_string(sequences, operators)

    for i in range(kwargs.num_batches):
        print("*************************Epoch{}************************************".format(i))
        # Convert sequences into Pytorch expressions that can be evaluated
        expressions = []
        rewards = []
        for j in range(len(sequences_int)):
            new_expr = Expression(operators, sequences_int[j].long().tolist(), sequences_lengths[j].long().tolist()).to(device)
            expressions.append(new_expr)

            if kwargs.use_cache:
                if str(sequences_int[j][:sequence_lengths[j]]) in expr_cache.keys():
                    print("repeated!")
                    rewards.append(expr_cache[str(sequences_int[j][:sequence_lengths[j]])])
                else:
                    optimize_constants([new_expr], X_constants, y_constants, kwargs.inner_lr, 
                                       kwargs.inner_num_epochs, kwargs.inner_optimizer)
                    if kwargs.doing_logits:
                        reward = reward_logit(new_expr, X_rnn, y_rnn)
                    else:
                        reward = benchmark(new_expr, X_rnn, y_rnn)
                    rewards.append(reward)
            else:
                optimize_constants([new_expr], X_constants, y_constants, 
                                   kwargs.inner_lr, kwargs.inner_num_epochs, kwargs.inner_optimizer)
                if kwargs.doing_logits:
                    reward = reward_logit(new_expr, X_rnn, y_rnn)
                else:
                    reward = benchmark(new_expr, X_rnn, y_rnn)
                rewards.append(reward)

        rewards = torch.tensor(rewards)
            
        # Update best expression
        best_epoch_expression = expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())
        best_epoch_sequence = strings[np.argmax(rewards)]
        epoch_best_sequences.append(best_epoch_sequence)
        if (max(rewards) > best_performance):
            best_performance = max(rewards)
            best_expression = best_epoch_expression
            best_sequence = best_epoch_sequence
            best_modules = copy.deepcopy(operators.modules)

        # Early stopping criteria
        if (best_performance >= 0.999):  
            best_str = str(best_expression)
            if (kwargs.live_print):
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
                print(f"""Best Sequence: {best_sequence}""")
            break
          
        end_time = time.time()
        if end_time-start_time>48*60*60:
            print("Exceed the maximum time cost (48 hours).")
            break
        # if i*initial_batch_size >= 2000000:
        #     print("2M expressions have been calculated.") 
        #     break

        # Compute risk threshold
        if (i == 0 and kwargs.scale_initial_risk):
            threshold = np.quantile(rewards, 1 - (1 - kwargs.risk_factor) / (kwargs.initial_batch_size / kwargs.batch_size))
        else:
            threshold = np.quantile(rewards, kwargs.risk_factor)
        # print("threshold:", threshold)
        if kwargs.doing_logits:
            indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] >= threshold])
        else:
            indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold])
        # print("indices_to_keep:", indices_to_keep.shape)

        if (len(indices_to_keep) == 0 and kwargs.summary_print):
            print("Threshold removes all expressions. Terminating.")
            break
        
        # Select corresponding subset of rewards, log_probabilities, and entropies
        rewards = torch.index_select(rewards, 0, indices_to_keep)
        
        log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
        entropies = torch.index_select(entropies, 0, indices_to_keep)

        # Compute risk seeking and entropy gradient
        risk_seeking_grad = torch.sum((rewards - threshold) * log_probabilities, axis=0)
        entropy_grad = torch.sum(entropies, axis=0)

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_grad = torch.clip(risk_seeking_grad / len(rewards), -1e6, 1e6)
        entropy_grad = kwargs.entropy_coefficient * torch.clip(entropy_grad / len(rewards), -1e6, 1e6)

        # Compute loss and backpropagate
        loss = -1 * kwargs.lr * (risk_seeking_grad + entropy_grad)
        loss.requires_grad_(True)
        loss.backward()
        optim.step()

        # Epoch Summary
        if (kwargs.live_print):
            print(f"""Epoch: {i+1} ({round(float(time.time() - start_time), 2)}s elapsed)
            Best Performance (Overall): {best_performance}
            Best Expression (Overall): {best_expression}
            Best Sequence (Overall): {best_sequence}
            Best Module (Overall): {best_modules}
            """)
        
        # update module with a frequency
        if kwargs.use_module:
            if i%kwargs.update_module_freq == 0:
                better_performed_exprs = torch.index_select(sequences_int, 0, indices_to_keep)
                modules.update_module(better_performed_exprs, operators, module_update_mechanism=kwargs.module_update_mechanism)
                operators.update_module(modules)

            sequences, sequence_lengths, entropies, log_probabilities = dsr_rnn.sample_sequence(kwargs.initial_batch_size, 
                                                                                                use_module=kwargs.use_module, 
                                                                                                device=device)
            # re-order the sequences that contains module.
            strings, sequences, sequences_lengths = dsr_rnn.reorder_sequences(sequences, sequence_lengths)
            sequences_lengths = torch.tensor(sequences_lengths)

            sequences_int = torch.zeros(len(sequences), torch.max(sequences_lengths))
            for k in range(len(sequences)):
                for j in range(sequences_lengths[k]):
                    sequences_int[k][j] = operators.operator_list.index(sequences[k][j])
        else:
            sequences_int, sequences_lengths, entropies, log_probabilities = dsr_rnn.sample_sequence(kwargs.initial_batch_size, 
                                                                                                     use_module=kwargs.use_module, 
                                                                                                     device=device)
            strings = modules.turn_sequence_to_string(sequences_int, operators)

    best_modules_string = ''
    for key in best_modules.modules.keys():
        best_modules_string += str(key)+':'+str(best_modules.modules[key]['traversal']) + '. '


    if (kwargs.summary_print):
        print(f"""
        Time Elapsed: {round(float(time.time() - start_time), 2)}s
        Epochs Required: {i+1}
        Best Performance: {round(best_performance.item(),3)}
        Best Expression: {best_expression}
        Best Sequence: {best_sequence}
        Best Modules: {best_modules_string}
        """)
    print("{} Non-repeated expresesion in toal.".format(len(expr_cache.keys())))
    
    del operators, dsr_rnn

    
    return [best_performance, 
            best_expression, 
            best_sequence, 
            best_modules, 
            best_modules_string,
            round(float(time.time() - start_time), 2), 
            i+1]


###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_rnn, y_rnn):
    """Obtain reward for a given expression using the passed X_rnn and y_rnn
    """
    with torch.no_grad():
        y_pred = expression(X_rnn)        
        return reward_nrmse(y_pred, y_rnn)

def reward_nrmse(y_pred, y_rnn):
    """Compute NRMSE between predicted y and actual y
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    val = torch.std(y_rnn) * val # Normalize using stdev of targets
    # val = val/(torch.std(y_rnn)+1e-6) # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()

def reward_logit(expression, X_rnn, y_rnn):
    with torch.no_grad():
        y_pred = expression(X_rnn)
        # print("y_pred.shape:", y_pred.shape, "y_rnn.shape:", y_rnn.shape)
        outputs = [1 if y_pred_i == y_rnn_i else 0 for y_pred_i, y_rnn_i in zip(y_pred, y_rnn)]
        outputs = np.mean(np.array(outputs))
        return outputs

def r2(expression, X_rnn, y_rnn):
    with torch.no_grad():
        y_pred = expression(X_rnn)
        y_pred = y_pred.cpu().numpy()
        y_rnn = y_rnn.cpu().numpy()
        reward = r2_score(y_pred, y_rnn)
        return reward
