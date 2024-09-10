from utils import *
import copy

class Node:
    def __init__(self, operator, arity, order):
        # order: the order of pre-order traversal
        self.operator = operator
        self.arity = arity
        self.lchild = None
        self.rchild = None
        self.order = order

    def add_child(self, node):
        if (self.lchild is None):
            self.lchild = node
            return True
        elif (self.rchild is None):
            self.rchild = node
            return True
        else:
            return False

    
class Tree:
    def __init__(self, traversal):
        self.traversal = traversal
        self.root = Node(self.traversal[0], arity_dict[self.traversal[0]], 0)
        self.nodes = [self.root]

    def construct(self):
        def construct_(root):
            nonlocal traversal
            if len(traversal)!=0:
                if root.arity == 0:
                    return
                elif root.arity == 1:
                    if root.lchild is None:
                        node = Node(traversal[0], arity_dict[traversal[0]], len(self.traversal)-len(traversal))
                        self.nodes.append(node)
                        # print("node:", node.operator)
                        root.lchild = node
                        traversal = traversal[1:]
                        construct_(root.lchild)
                else:
                    if root.lchild is None:
                        node = Node(traversal[0], arity_dict[traversal[0]], len(self.traversal)-len(traversal))
                        self.nodes.append(node)
                        # print("node:", node.operator)
                        root.lchild = node
                        traversal = traversal[1:]
                        construct_(root.lchild)
                    if root.rchild is None:
                        node = Node(traversal[0], arity_dict[traversal[0]], len(self.traversal)-len(traversal))
                        self.nodes.append(node)
                        # print("node:", node.operator)
                        root.rchild = node
                        traversal = traversal[1:]
                        construct_(root.rchild)
            return
        traversal = copy.deepcopy(self.traversal)
        traversal = traversal[1:]
        construct_(self.root)

    def printer(self):
        def printer_(root):
            print(root.operator, root.order)
            if root.lchild:
                print(" root:", root.operator, root.order, "  left child:", root.lchild.operator, root.lchild.order)
                printer_(root.lchild)
            if root.rchild:
                print(" root:", root.operator, root.order, "  right child:", root.rchild.operator, root.rchild.order)
                printer_(root.rchild)
        printer_(self.root)

    def is_leaf(self, node):
        if node.lchild is None and node.rchild is None:
            return True
        return False

    def recursive_traversal(self, root, condition, ban_list=[]):
        # condition: the specified max node number
        sequences = []
        order_seqs = []
        seqs = []
        o_seqs = []
        def recursive_traversal_(node):
            nonlocal seqs, o_seqs
            seqs.append(node.operator)
            o_seqs.append(node.order)
            # print(node.operator)
            if len(o_seqs) == condition:
                # seqs.append(node.operator)
                sequences.append(copy.deepcopy(seqs))
                order_seqs.append(copy.deepcopy(o_seqs))
                seqs.pop()
                seqs.append('var')
                o_seqs.pop()
                return
            else:
                if node.lchild and self.is_leaf(node.lchild) is False:
                    if node.lchild.order not in ban_list:
                        recursive_traversal_(node.lchild)
                    else:
                        seqs.append('var')
                if node.rchild and self.is_leaf(node.rchild) is False:
                    if  node.rchild.order not in ban_list:
                        recursive_traversal_(node.rchild)
                    else:
                        seqs.append('var')
            return 

        recursive_traversal_(root)
        return sequences, order_seqs, seqs, o_seqs

    def traversal_a_node_substructre(self, root, condition):
        # root: the specified root node, could be the root of a subtree of the original
        # condition: the specified max node number
        def add_banlist(ban_list, node_order):
            # add the node according to the node order
            if ban_list == []:
                return [node_order]
            else:
                new_ban_list = []
                for nr in ban_list:
                    if node_order>nr:
                        new_ban_list.append(nr)
                    elif node_order < nr:
                        new_ban_list.append(node_order)
                        return new_ban_list
                    else:
                        print("Error! the node_order repeated!")
                new_ban_list.append(node_order)
                return new_ban_list

        ban_list = []
        all_sequences = []
        all_orders = []
        # print("the first ban_list:", ban_list)
        while True:
            sequences, order_seqs, seqs, o_seqs= self.recursive_traversal(root, condition, ban_list)
            ban_nr = o_seqs[-1]
            ban_list = add_banlist(ban_list, ban_nr)
            if sequences != []:
                all_sequences.extend(sequences)
                all_orders.extend(order_seqs)
            if ban_list == [root.order]:
                # ban the root node, the tree has traversaled.
                break
        return all_sequences, all_orders

    def traversal_substructure(self, condition):
        all_node_seqs = []
        all_node_ords = []
        # print("self.nodes:", self.nodes)
        for node in self.nodes:
            if self.is_leaf(node) is False: # non-leaf node
                seqs, orders = self.traversal_a_node_substructre(node, condition)
                all_node_seqs.extend(seqs)
                all_node_ords.extend(orders)
                # print("node {}({}) has {} substructure of condition {}.".format(node.operator, node.order, len(seqs), condition))
        return all_node_seqs, all_node_ords
