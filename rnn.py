import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

###############################################################################
# Sequence RNN Class
###############################################################################

class DSRRNN_MULTI(nn.Module):
    def __init__(self, operators, modules, device, config):
        super(DSRRNN_MULTI, self).__init__()
        
        self.modules = modules

        kwargs = config.training
        self.input_size = kwargs.max_arity*len(operators) # One-hot encoded parent and sibling
        self.hidden_size = kwargs.hidden_size
        self.output_size = len(operators) # Output is a softmax distribution over all operators
        self.max_arity = kwargs.max_arity
        self.num_layers = kwargs.num_layers
        self.dropout = kwargs.dropout
        self.operators = operators
        self.device = device

        self.type = kwargs.type
        self.repeat = config.prior.repeat.max_

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True).to(self.device)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size), requires_grad=True).to(self.device)

        self.min_length = kwargs.min_length
        self.max_length = kwargs.max_length

        if (self.type == 'rnn'):
            self.rnn = nn.RNN(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        elif (self.type == 'lstm'):
            self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                proj_size = self.output_size,
                dropout = self.dropout
            ).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size), requires_grad=True).to(self.device)
        elif (self.type == 'gru'):
            self.gru = nn.GRU(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    def sample_sequence(self, n, use_module=False, device=None):
        print("sample sequence, use module or not.", use_module)
        
        sequences = torch.zeros((n, 0))
        entropies = torch.zeros((n, 0)) # Entropy for each sequence
        log_probs = torch.zeros((n, 0)) # Log probability for each token

        sequence_mask = torch.ones((n, 1), dtype=torch.bool)

        counters = torch.ones(n) # Number of tokens that must be sampled to complete expression
        lengths = torch.zeros(n) # Number of tokens currently in expression

        input_tensor = self.init_input.repeat(n, 1) # sample n expressions in an epoch
        # input_tensor = self.get_tensor_input(initial_obs)
        hidden_tensor = self.init_hidden.repeat(n, 1)

        if (self.type == 'lstm'):
            hidden_lstm = self.init_hidden_lstm.repeat(n, 1)
        
        # While there are still tokens left for sequences in the batch
        while(sequence_mask.all(dim=1).any()):
            if (self.type == 'rnn'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
            elif (self.type == 'lstm'):
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)
            elif (self.type == 'gru'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)

            # control use module or not
            if use_module:
                pass
            else:
                for i in range(len(list(self.modules.modules.keys()))):
                    output[:, -(i+1)] = torch.zeros(output.shape[0])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, sequence_mask.sum(dim=1)-1, sequences, device)
            output = output / (torch.sum(output, axis=1)[:, None]+1e-10)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(torch.tensor(output, dtype=torch.float32), validate_args = False)
            token = dist.sample()
            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), axis=1)
            lengths += 1
        
            # Add log probability of current token
            log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)

            # Add entropy of current token
            entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)

            # Update counter
            counters -= 1
            counters += torch.isin(token, self.operators.arity_one).long() * 1
            counters += torch.isin(token, self.operators.arity_two).long() * 2
            counters += torch.isin(token, self.operators.arity_three).long() * 3
            counters += torch.isin(token, self.operators.arity_four).long() * 4
 
            # Update sequence mask
            # This is for the next token that we sample. Basically, we know if the
            # next token will be valid or not based on whether we've just completed the sequence (or have in the past)
            sequence_mask = torch.cat(
                (sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),
                axis=1
            )
            
            # in this place, turn the 
            # Compute next parent and sibling; assemble next input tensor
            parent_siblings = self.get_parent_siblings(sequences, sequence_mask.sum(dim=1), device)
            input_tensor = self.get_next_input(parent_siblings)

        # Filter entropies log probabilities using the sequence_mask
        entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)
        log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
        # print("entropies:", entropies)
        sequence_lengths = torch.sum(sequence_mask.long(), axis=1)
        return sequences, sequence_lengths, entropies, log_probs

    def sample_with_labels(self, sequences_label, use_module=False, device=None):
        n = sequences_label.shape[0]
        sequences = torch.zeros((n, 0))
        entropies = torch.zeros((n, 0)) # Entropy for each sequence
        log_probs = torch.zeros((n, 0)) # Log probability for each token

        sequence_mask = torch.ones((n, 1), dtype=torch.bool)
        counters = torch.ones(n) # Number of tokens that must be sampled to complete expression
        lengths = torch.zeros(n) # Number of tokens currently in expression
        
        input_tensor = self.init_input.repeat(n, 1) # sample n expressions in an epoch
        # input_tensor = self.get_tensor_input(initial_obs)
        hidden_tensor = self.init_hidden.repeat(n, 1)

        if (self.type == 'lstm'):
            hidden_lstm = self.init_hidden_lstm.repeat(n, 1)
        
        t = 0
        while(sequence_mask.all(dim=1).any()):
            t+=1
            if (self.type == 'rnn'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
            elif (self.type == 'lstm'):
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)
            elif (self.type == 'gru'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)

            # control use module or not
            if use_module:
                pass
            else:
                for i in range(len(list(self.modules.modules.keys()))):
                    output[:, -(i+1)] = torch.zeros(output.shape[0])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, sequence_mask.sum(dim=1)-1, sequences, device)
            output = output / (torch.sum(output, axis=1)[:, None]+1e-10)

            dist = torch.distributions.Categorical(torch.tensor(output, dtype=torch.float32), validate_args = False)
            # use the selected token
            token = sequences_label[:, sequences.shape[1]]
            sequences = torch.cat((sequences, token[:, None]), axis=1)
            # print("token:", token, "t:", t)
            log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)

            # Add entropy of current token
            entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)

            # Update counter
            counters -= 1
            counters += torch.isin(token, self.operators.arity_one).long() * 1
            counters += torch.isin(token, self.operators.arity_two).long() * 2
            counters += torch.isin(token, self.operators.arity_three).long() * 3
            counters += torch.isin(token, self.operators.arity_four).long() * 4
            lengths += 1

            sequence_mask = torch.cat(
                (sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),
                axis=1
            )
            
            # in this place, turn the 
            # Compute next parent and sibling; assemble next input tensor
            parent_siblings = self.get_parent_siblings(sequences, sequence_mask.sum(dim=1), device)
            input_tensor = self.get_next_input(parent_siblings)


        entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)
        log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
        return entropies, log_probs
    
    def forward(self, input, hidden, hidden_lstm=None):
        """Input should be [parent, sibling]
        """
        if (self.type == 'rnn'):
            output, hidden = self.rnn(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hidden[0, :]
        elif (self.type == 'lstm'):
            output, (hn, cn) = self.lstm(input[:, None].float(), (hidden_lstm[None, :], hidden[None, :]))
            output = self.activation(output[:, 0, :])
            return output, cn[0, :], hn[0, :]
        elif (self.type == 'gru'):
            output, hn = self.gru(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]

    def apply_constraints(self, output, counters, lengths, sequences, device):
        # Add small epsilon (small positive number) to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        epsilon = torch.ones(output.shape) * 1e-20
        output = output + epsilon.to(self.device)
        
        # ~ Check that minimum length will be met ~
        # Explanation here
        min_boolean_mask = (counters + lengths >= torch.ones(counters.shape) * self.min_length).long()[:, None]
        min_length_mask = torch.max(self.operators.nonzero_arity_mask[None, :], min_boolean_mask)
        output = torch.minimum(output, min_length_mask)

        # ~ Check that maximum length won't be exceed ~
        max_boolean_mask = (counters + lengths <= torch.ones(counters.shape) * (self.max_length - self.operators.max_arity)).long()[:, None] # sub 4: the maximum arity is 4
        max_length_mask = torch.max(self.operators.zero_arity_mask[None, :], max_boolean_mask)
        output = torch.minimum(output, max_length_mask)
        
        # ~ Ensure that all expressions have a variable ~
        nonvar_zeroarity_mask = (~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)).long()
        if (lengths[0].item() == 0.0): # First thing we sample can't be
            output = torch.minimum(output, nonvar_zeroarity_mask)
        else:
            nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # Don't sample a nonvar zeroarity token if the counter is at 1 and
            # we haven't sampled a variable yet
            counter_mask = (counters == 1)
            contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(nonvar_zeroarity_mask.shape)).long()
            output = torch.minimum(output, nonvar_zeroarity_mask)

        # ~ Ensure that all trig functions can not be nested
        ancestors = self.get_ancestors(sequences, lengths) # the returned ancestors is in string
        for i in range(len(ancestors)):
            if 'cos' in ancestors[i] or 'sin' in ancestors[i] or 'tan' in ancestors[i]:
                # output[i] = torch.minimum(output[i], self.operators.non_sin_cos_mask.long())
                if 'cos' in self.operators.operator_list:
                    output[i][self.operators.operator_list.index('cos')] = 0
                if 'sin' in self.operators.operator_list:
                    output[i][self.operators.operator_list.index('sin')] = 0
                if 'tan' in self.operators.operator_list:
                    output[i][self.operators.operator_list.index('tan')] = 0
                # control the module, cos, sin, tan may contain in a module
                for key in self.modules.modules.keys():
                    if 'cos' in self.modules.modules[key]['traversal'] and 'cos' in self.operators.operator_list:
                        output[i][self.operators.operator_list.index(key)] = 0
                    if 'sin' in self.modules.modules[key]['traversal'] and 'sin' in self.operators.operator_list:
                        output[i][self.operators.operator_list.index(key)] = 0
                    if 'tan' in self.modules.modules[key]['traversal'] and 'tan' in self.operators.operator_list:
                        output[i][self.operators.operator_list.index(key)] = 0

        
        # ~ Ensure that log and exp, square and sqrt, arccos and cos, arcsin and sin can not be adjacent
        # ~ when parent is log, the child can not be exp
        for i in range(len(ancestors)):
            if ancestors[i]!=[]:
                token = ancestors[i][-1]
                if 'log' in self.operators.operator_list and 'exp' in self.operators.operator_list:
                    if token == 'log':
                        output[i][self.operators.operator_list.index('exp')] = 0
                    if token == 'exp':
                        output[i][self.operators.operator_list.index('log')] = 0
                if 'sqrt' in self.operators.operator_list and 'square' in self.operators.operator_list:
                    if token == 'sqrt':
                        output[i][self.operators.operator_list.index('square')] = 0
                    if token == 'square':
                        output[i][self.operators.operator_list.index('sqrt')] = 0
                if "+" in self.operators.operator_list:
                    output[i][self.operators.operator_list.index('+')] = 0
                if "-" in self.operators.operator_list:
                    output[i][self.operators.operator_list.index('-')] = 0


        # ~ Ensure that const token is not the only unique child of all non-terminal tokens
        parent_siblings = self.get_parent_siblings(sequences, lengths, device)
        constant_index = self.operators.constant_index
        for i in range(len(parent_siblings)):
            parent = parent_siblings[i][0]
            siblings = parent_siblings[i][1:]

            if self.operators.arity_i(int(parent)) == 1:
                for index in constant_index:
                    output[i][index] = 0
            else:
                num = 0
                for sibling in siblings:
                    if sibling in constant_index:
                        num += 1
                if num == self.operators.arity_i(int(parent)) - 1:
                    # at leat a variable
                    for index in constant_index:
                        output[i][index] = 0
            
        # ~ Ensure that each operator in the operator_list won't occur more than self.prior.repeat.max_ times
        batch_size, L = sequences.size()

        if L > self.repeat: 
            # repeat_masks = torch.ones((batch_size, self.operators.L), dtype=sequences.dtype).long()
            mask_length = torch.arange(L, device=sequences.device).expand(batch_size, L) < lengths.unsqueeze(1)
            padded_sequences = sequences * mask_length
            # print("padded_sequences:", padded_sequences)
            one_hot = torch.nn.functional.one_hot(padded_sequences.long(), 
                                                  num_classes=self.operators.L).to(sequences.dtype)
            one_hot_masked = one_hot * mask_length.unsqueeze(-1)
            counts = torch.sum(one_hot_masked, dim=1)
            mask_counts = counts < self.repeat
            mask_counts[:, self.operators.arity_zero.long()] = True
            output *= mask_counts

        return output

    def get_ancestors(self, sequences, lengths):
        # use to judge whether has cos(sin())
        if sequences.numel():
            _, re_sequences, re_lengths = self.reorder_sequences(sequences, lengths)
            ancestors = []
            for seqs, length in zip(re_sequences, re_lengths):
                stack = []
                v = 0
                for i in range(int(length)):
                    if self.operators.arity_dict[seqs[i]] == 0:
                        v += 1
                        start_index = len(stack)
                        for j in range(len(stack)-1, -1, -1):
                            if self.operators.arity_dict[stack[j]] == v:
                                start_index -= 1
                                v = 1
                            else:
                                break
                        stack = stack[:start_index]
                    else:
                        stack.append(seqs[i])
                    
                ancestors.append(copy.deepcopy(stack))
        else:
            ancestors = []
        return ancestors

    def reorder_sequences(self, sequences_with_module, sequence_length_with_module):
        strings = []
        # change int to string.
        for line, length in zip(sequences_with_module, sequence_length_with_module):
            # print("line:", line)
            temp = []
            for i in range(length):
                temp.append(self.operators.operator_list[int(line[i])])
            strings.append(copy.deepcopy(temp))

        # use arity to re-order sequence, turn module to operators
        def is_module_in(string):
            for token in string:
                if 'M' in token:
                    return True
            return False

        # strings = sequences_with_module
        sequences = []
        sequences_lengths = []
        for string in strings:
            while is_module_in(string):
                seqs = []
                temp = string
                while temp != []:
                    token = temp[0]
                    temp = temp[1:]
                    if 'M' in token:
                        flag = False
                        for i in range(len(self.modules.modules[token]['traversal'])):
                            if self.modules.modules[token]['traversal'][i] == 'var':
                                # sample a var from the sequence
                                arity = 1
                                flag = True
                                while temp!= []:
                                    arity += self.operators.arity_dict[temp[0]] - 1
                                    seqs.append(temp[0])
                                    temp = temp[1:]
                                    if arity == 0:
                                        break
                            else:
                                # if is a operator, add it to seqs
                                if flag:
                                    if arity == 0:
                                        seqs.append(self.modules.modules[token]['traversal'][i])
                                else:
                                    seqs.append(self.modules.modules[token]['traversal'][i])
                    else:
                        seqs.append(token)
                string = seqs
            sequences.append(string)
            sequences_lengths.append(len(string))
        return  strings, sequences, sequences_lengths

    def get_parent_siblings(self, sequences, lengths, device):
        parent_siblings_seq = torch.ones((lengths.shape[0], self.max_arity)) * (-1)
        recent = int(sequences.shape[1])-1

        c = torch.zeros(lengths.shape[0])
        # initial False, means if the finding is finished
        c_mask = torch.bitwise_or((lengths == 0), torch.zeros(lengths.shape[0], 1).all(dim=1))
        # print("c_mask:", c_mask)
        
        for i in range(recent, -1, -1):
            # Determine arity of the i-th tokens, to know the pairs of sibling and parent
            token_i = sequences[:, i]
            # counter = counter + arity - 1
            arity = torch.zeros(sequences.shape[0])
            # for mudule, add arities that larger than 2
            # arity += torch.isin(token_i, self.operators.arity_six).long() * 6
            # arity += torch.isin(token_i, self.operators.arity_five).long() * 5
            arity += torch.isin(token_i, self.operators.arity_four).long() * 4
            arity += torch.isin(token_i, self.operators.arity_three).long() * 3
            arity += torch.isin(token_i, self.operators.arity_two).long() * 2
            arity += torch.isin(token_i, self.operators.arity_one).long() * 1

            # Increment c by arity of the i-th token, minus 1
            c += arity
            c -= 1

            for l in range(len(c)):
                # print("i:", i, "l:", l, "mask:",c_mask[l],  "sequences:", sequences[l, :])
                parent_siblings = torch.ones(self.max_arity) * (-1)
                # if arity(token_i) > 0:
                if c[l] > 0 and ~c_mask[l]:
                    parent_siblings[0] = token_i[l]
                    c_mask[l] = True
                    parent_siblings_seq[l] = parent_siblings.to(device)
                    continue
                # if the counter >= 0:
                if c[l] >= 0 and ~c_mask[l]:
                    stack = []
                    for j in range(recent, i, -1):
                        token_j = sequences[l, j].item()
                        if token_j in self.operators.arity_zero:
                            arity_j = 0
                        elif token_j in self.operators.arity_one:
                            arity_j = 1
                        elif token_j in self.operators.arity_two:
                            arity_j = 2
                        elif token_j in self.operators.arity_three:
                            arity_j = 3
                        else:
                            arity_j = 4
                        for ari in range(arity_j):
                            stack.pop()
                        stack.append(token_j)
                    parent_siblings[0] = sequences[l, i].item()
                    if len(stack)!=0:
                        if len(stack)>3:
                            print("Error!")
                        stack.reverse()
                        for j in range(len(stack)):
                            parent_siblings[1+j] = stack[j]
                    parent_siblings_seq[l] = parent_siblings.to(device)
                    c_mask[l] = True
                else:
                    parent_siblings = torch.ones(self.max_arity) * (-1)

        return parent_siblings_seq

    def get_next_input(self, parent_siblings):
        # Just convert -1 to 1 for now; it'll be zeroed out later
        parent = torch.abs(parent_siblings[:, 0]).long()
        sibling1 = torch.abs(parent_siblings[:, 1]).long()
        sibling2 = torch.abs(parent_siblings[:, 2]).long()
        sibling3 = torch.abs(parent_siblings[:, 3]).long()

        # Generate one-hot encoded tensors
        parent_onehot = F.one_hot(parent, num_classes=len(self.operators))
        sibling1_onehot = F.one_hot(sibling1, num_classes=len(self.operators))
        sibling2_onehot = F.one_hot(sibling2, num_classes=len(self.operators))
        sibling3_onehot = F.one_hot(sibling3, num_classes=len(self.operators))
        

        # Use a mask to zero out values that are -1. Parent should never be -1,
        # but we do it anyway.
        parent_mask = (~(parent_siblings[:, 0] == -1)).long()[:, None]
        parent_onehot = parent_onehot * parent_mask
        sibling1_mask = (~(parent_siblings[:, 1] == -1)).long()[:, None]
        sibling1_onehot = sibling1_onehot * sibling1_mask
        sibling2_mask = (~(parent_siblings[:, 2] == -1)).long()[:, None]
        sibling2_onehot = sibling2_onehot * sibling2_mask
        sibling3_mask = (~(parent_siblings[:, 3] == -1)).long()[:, None]
        sibling3_onehot = sibling3_onehot * sibling3_mask

        input_tensor = torch.cat((parent_onehot, sibling1_onehot, sibling2_onehot, sibling3_onehot), axis=1)
        return input_tensor
