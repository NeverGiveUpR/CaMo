import torch

class Operators:
    """
    The list of valid nonvariable operators may be found in nonvar_operators.
    All variable operators must have prefix 'var_'. Constant value operators
    are fine too (e.g. 3.14), but they must be passed as floats.
    """
    nonvar_operators = [
        '*', '+', '-', '/', '^',
        'cos', 'sin', 'tan', 'arcsin', 'arccos',
        'exp', 'log',
        'sqrt', 'square',
        'neg',  
        'c', # ephemeral constant,
        'AND', 'OR', 'NOT', 'NOR', 'NAND' # logit operator
    ]
    nonvar_arity = {
        '*': 2,
        '+': 2,
        '-': 2,
        '/': 2,
        '^': 2,
        'cos': 1,
        'sin': 1,
        'tan': 1,
        'arcsin': 1,
        'arccos': 1,
        'exp': 1,
        'log': 1,
        'sqrt': 1,
        'square': 1,
        'neg': 1,
        'c': 0,
        'AND': 2,
        'OR': 2,
        'NOT': 1,
        'NOR': 2,
        'NAND': 2
    }
    function_mapping = {
        '*': 'torch.mul',
        '+': 'torch.add',
        '-': 'torch.subtract',
        '/': 'torch.divide',
        '^': 'torch.pow',
        'cos': 'torch.cos',
        'sin': 'torch.sin',
        'tan': 'torch.tan',
        'arcsin': 'torch.arcsin',
        'arccos': 'torch.arccos',
        'exp': 'torch.exp',
        'log': 'torch.log',
        'sqrt': 'torch.sqrt',
        'square': 'torch.square',
        'neg': 'torch.neg',
        'AND': 'AND',
        'OR': 'OR',
        'NOT': 'NOT',
        'NOR': 'NOR',
        'NAND': 'NAND'
    }

    def __init__(self, operator_list, modules, device, max_arity=4, use_module=False):
        """Description here
        """
        self.max_arity = max_arity
        # add module operator to the operator list
        self.modules = modules
        self.operator_list = operator_list
        
        # add modules
        self.module_operators = [x for x in operator_list if "M" in x]
        for i in range(self.modules.module_num):
            operator_list.append('M'+str(i+1))
            self.module_operators.append('M'+str(i+1))

        self.L = len(self.operator_list)

        self.constant_operators = [x for x in operator_list if x.replace('.', '').strip('-').isnumeric()]
        self.nonvar_operators = [x for x in self.operator_list if "var_" not in x and x not in self.constant_operators]
        self.var_operators = [x for x in operator_list if x not in self.nonvar_operators and x not in self.constant_operators]
        
        self.__check_operator_list() # Sanity check

        self.device = device

        self.constant_index = []
        try:
            for x in self.constant_operators:
                self.constant_index.append(self.operator_list.index(x))
            self.constant_index.append(self.operator_list.index('c'))
        except:
            pass
        
        # print("operator_list:", self.operator_list)
        # print("self.modules:", self.modules.modules)
        # Construct data structures for handling arity
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators}, **{x: 0 for x in self.constant_operators}, \
            **{x:self.modules.modules[x]['arity'] for x in self.module_operators}) # arity==0
        # print("self.arity_dict:", self.arity_dict)
        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list])# to obtain zero arity operator
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list])
        self.variable_mask = torch.Tensor([1 if x in self.var_operators else 0 for x in self.operator_list])
        self.nonvariable_mask = torch.Tensor([0 if x in self.var_operators else 1 for x in self.operator_list])

        self.arity_four = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==4])
        self.arity_three = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==3])
        # Contains indices of all operators with arity 2
        self.arity_two = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==2])
        # Contains indices of all operators with arity 1
        self.arity_one = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==1])
        # Contains indices of all operators with arity 0
        self.arity_zero = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==0])
        # Contains indices of all operators that are variables
        self.variable_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if operator_list[i] in self.var_operators])

        # Construct data structures for handling function and variable mappings
        self.func_dict = dict(self.function_mapping)
        # not contain modules
        self.var_dict = {var: i for i, var in enumerate(self.var_operators)}

        trig_names = ["sin", "cos", "tan", "csc", "sec", "cot"]
        trig_names += ["arc" + name for name in trig_names]
        self.trig_tokens = [i for i, t in enumerate(self.operator_list) if t in trig_names]
        inverse_tokens = {
            "inv" : "inv",
            "neg" : "neg",
            "exp" : "log",
            "log" : "exp",
            "sqrt" : "square",
            "square" : "sqrt"
        }
        token_from_name = {t : i for i, t in enumerate(self.operator_list)}
        self.inverse_tokens = {token_from_name[k] : token_from_name[v] for k, v in inverse_tokens.items() if k in token_from_name and v in token_from_name}

    def __check_operator_list(self):
        """Throws exception if operator list is bad
        """
        invalid = [x for x in self.nonvar_operators if x not in Operators.nonvar_operators and 'M' not in x]
        if (len(invalid) > 0):
            raise ValueError(f"""Invalid operators: {str(invalid)}""")
        return True

    def __getitem__(self, i):
        try:
            return self.operator_list[i]
        except:
            return self.operator_list.index(i)

    def arity(self, operator):
        try:
            return self.arity_dict[operator]
        except NameError:
            print("Invalid operator")

    def arity_i(self, index):
        try:
            return self.arity_dict[self.operator_list[index]]
        except NameError:
            print("Invalid index")

    def update_module(self, modules):
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators}, **{x: 0 for x in self.constant_operators}, \
            **{x:modules.modules[x]['arity'] for x in self.module_operators}) # arity==0

        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list]).to(self.device)# to obtain zero arity operator
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list]).to(self.device)
        self.variable_mask = torch.Tensor([1 if x in self.var_operators else 0 for x in self.operator_list])
        self.nonvariable_mask = torch.Tensor([0 if x in self.var_operators else 1 for x in self.operator_list])

        self.arity_four = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==4])
        self.arity_three = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==3])
        self.arity_two = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==2])
        self.arity_one = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==1])
        self.arity_zero = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==0])

    def func(self, operator):
        return self.func_dict[operator]

    def func_i(self, index):
        return self.func_dict[self.operator_list[index]]

    def var(self, operator):
        return self.var_dict[operator]

    def var_i(self, index):
        return self.var_dict[self.operator_list[index]]

    def __len__(self):
        return len(self.operator_list)
    
class TokenNotFoundError(Exception):
    pass
