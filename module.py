import numpy as np
from binaryTree import *
import random

class Module:
    def __init__(self, module_num):
        # module_num: allowed maximum module number
        self.module_num = module_num

        # initialize modules
        self.modules = {}
        for i in range(self.module_num):
            self.modules['M'+str(i+1)] = None
        
    def add_module(self, module):
        for key in self.modules.keys():
            if self.modules[key] is None:
                self.modules[key] = module
                break

class UpdateModule: # the module class that used for updating module
    def __init__(self, module_num, max_length=8, max_arity=4, mutation=False):
        self.module_num = module_num
        self.modules = {}
        for i in range(self.module_num):
            self.modules['M'+str(i+1)] = None
        self.max_length = max_length
        self.max_arity = max_arity
        self.mutation = mutation
        
    
    def add_module(self, module):
        for key in self.modules.keys():
            if self.modules[key] is None:
                self.modules[key] = module
                break

    def substitute_module(self, new_modules):
        module_list = list(new_modules.keys())
        i = 0
        while i < self.module_num and module_list!=[]:
            self.modules['M'+str(i+1)] = {'traversal': new_modules[module_list[0]], 'arity': new_modules[module_list[0]].count('var')}
            module_list = module_list[1:]
            i += 1

    def substitute_module_with_prob(self, new_modules, top_k=10):
        # give some exploration rate
        # select top_k modules to provide module update
        module_list = list(new_modules.keys())[:top_k]
        random.shuffle(module_list)
        i = 0
        while i < self.module_num and module_list!=[]:
            self.modules['M'+str(i+1)] = {'traversal': new_modules[module_list[0]], 'arity': new_modules[module_list[0]].count('var')}
            module_list = module_list[1:]
            i += 1
    
    def turn_sequence_to_string(self, sequences, operators):
        strings = []
        for i in range(len(sequences)):
            temp = []
            for seq in sequences[i]:
                temp.append(operators.operator_list[int(seq)])
            strings.append(temp)
        return strings

    def sample_module(self, strings):
        # traversal sequences to obtain sub-structures
        sub_structures = []
        for string in strings:
            tree = Tree(string)
            tree.construct()
            for i in range(2, self.max_length): # at leat 2 operators
                seqs, _ = tree.traversal_substructure(i)
                if len(seqs) != 0:
                    # complete the module
                    for seq in seqs:
                        counter = 1
                        for s in seq:
                            counter += arity_dict[s] - 1
                        for j in range(counter):
                            seq.append('var')
                    arity = seq.count('var')
                    if arity > self.max_arity:
                        pass # drop the module that exceeds the maximum arity
                    else:
                        sub_structures.append(seq)
        
        # calculate the frequency of each module
        no_repeated = []
        frequency = {}
        freq_module = {}
        for sub_struct in sub_structures:
            if sub_struct in no_repeated:
                frequency[str(sub_struct)] += 1
            else:
                frequency[str(sub_struct)] = 1
                no_repeated.append(sub_struct)
                freq_module[str(sub_struct)] = sub_struct

        sorted_dict = {}
        sorted_keys = sorted(frequency, key=frequency.get, reverse=True)
        for w in sorted_keys:
            sorted_dict[w] = freq_module[w]
        return sorted_dict
    
    def evaluate_module(self, strings):
        # return the value of modules
        # use the frequency as evaluation value
        freq = {}
        for i in range(self.module_num):
            freq['M'+str(i+1)] = 0

        all_num = 0
        for string in strings:
            for token in string:
                if token in list(freq.keys()):
                    freq[token] += 1
                all_num += 1

        for key in freq.keys():
            freq[key] = freq[key]/all_num
        
        return freq


    def update_module(self, sequences, operators, last_values=None, module_update_mechanism='self'):
        strings = self.turn_sequence_to_string(sequences, operators)
        
        # evaluate the performance of modules
        # values = self.evaluate_module(strings)

        # decide to update which modules according to the values


        # traversal sequences to obtain sub-structures
        if module_update_mechanism == 'self':
            new_modules = self.sample_module(strings)
        elif module_update_mechanism == 'random':
            new_modules = self.random_sample(strings)
        else:
            print("Error! not supported module_update_mechanism.")
            quit()
        # update modules
        if self.mutation == False:
            self.substitute_module(new_modules)
        else:
            self.substitute_module_with_prob(new_modules)

    def random_sample(self, strings):
        # traversal sequences to obtain sub-structures
        sub_structures = []
        for string in strings:
            tree = Tree(string)
            tree.construct()
            for i in range(2, self.max_length): # at leat 2 operators
                seqs, _ = tree.traversal_substructure(i)
                if len(seqs) != 0:
                    # complete the module
                    for seq in seqs:
                        counter = 1
                        for s in seq:
                            counter += arity_dict[s] - 1
                        for j in range(counter):
                            seq.append('var')
                    arity = seq.count('var')
                    if arity > self.max_arity:
                        pass # drop the module that exceeds the maximum arity
                    else:
                        sub_structures.append(seq)
        
        np.random.shuffle(np.array(sub_structures))
        # calculate the frequency of each module
        no_repeated = []
        unique_module = {}
        for sub_struct in sub_structures:
            if sub_struct not in no_repeated:
                unique_module[str(sub_struct)] = sub_struct

        return unique_module
