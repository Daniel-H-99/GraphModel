import torch
import numpy as np 
import scipy.sparse as sp 


def load_graph():
    graph_path = "../data/train.csv"
    node_label_path = "../data/node_ingredient.csv"
    NUM_INGREDIENT = None
    NUM_FOOD = None
    NUM_CUISINE = None
    INGREDIENT_START_IDX = 0
    FOOD_START_IDX = None
    CUISINE_START_IDX = None

    
    print(f"loading graph from {graph_path}")

    # load node labels (names of ingredients)
    ingredient_idx = np.genfromtxt("{}".format(graph_path), delimiter='\n', dtype=np.dtype(str)).tolist()
    NUM_INGREDIENT = len(ingredient_idx)
    FOOD_START_IDX = NUM_INGREDIENT

    # load edges
    # 
    # edges : list of ordered pairs 
    edges = []
    with open(graph_path, 'r') as f:
        lines = f.readlines()

    NUM_FOOD = len(lines)
    CUISINE_START_IDX = NUM_INGREDIENT + NUM_FOOD
    food_idx = list(map(lambda x: str(x), range(NUM_FOOD)))
    cuisine_idx = []

    for i, line in enumerate(lines):
        tokens = line.strip().split(',')
        ingredients, cuisine = tokens[:-1], tokens[-1]

        # get  food node number
        food_node = food_idx.index(str(i)) 

        # get cuisine node number
        if cuisine not in cuisine_idx:
            cuisine_idx.append(cuisine)
        cuisine_node = CUISINE_START_IDX + cuisine_idx.index(cuisine)
        
        edges.extend([
            [food_node, cuisine_node],
            [cuisine_node, food_node]
        ])

        for a in ingredients:
            edges.extend([
                [int(a), food_node],
                [food_node, int(a)]
            ])
            for b in ingredients:
                if a == b: 
                    continue
                edges.append([int(a), int(b)])

    NUM_CUISINE = len(cuisine_idx)

    node_labels = np.concatenate([ingredient_idx, food_idx, cuisine_idx], axis=0).astype(np.str)
    edges = np.array(edges, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(node_labels.shape[0], node_labels.shape[0]), dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)

    # features are initialized to zero
    # features = normalize_features(features)
    adj = normalize_adj(adj+sp.eye(adj.shape[0]))

    idx_ingredient = [0, NUM_INGREDIENT - 1] 
    idx_food = [FOOD_START_IDX, CUISINE_START_IDX - 1]
    idx_cuisine = [CUISINE_START_IDX, CUISINE_START_IDX + NUM_CUISINE - 1]

    adj = torch.FloatTensor(np.array(adj.todense()))

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return {'adj': adj, 'idx_ingredient': idx_ingredient, 'idx_food': idx_food, 'idx_cuisine': idx_cuisine, 'node_labels': node_labels}

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def normalize_adj(mx): # A_hat = DAD
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx_to =  mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx_to

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_to =  r_mat_inv.dot(mx) 
    return mx_to 

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot 

import random
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import copy

# data manager for recording, saving, and plotting
class AverageMeter(object):
    def __init__(self, args, name='noname', save_all=False, surfix='.', x_label=None):
        self.args = args
        self.name = name
        self.save_all = save_all
        self.surfix = surfix
        self.path = os.path.join(args.path, args.result_dir, args.name, surfix)
        self.x_label = x_label
        self.reset()
    def reset(self):
        self.max = - 100000000
        self.min = 100000000
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.save_all:
            self.data = []
        self.listeners = []
    def load_array(self, data):
        self.max = max(data)
        self.min = min(data)
        self.val = data[-1]
        self.sum = sum(data)
        self.count = len(data)
        if self.save_all:
            self.data.extend(data)
    def update(self, val, weight=1):
        prev = copy.copy(self)
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        if self.save_all:
            self.data.append(val)
        is_max, is_min = False, False
        if val > self.max:
            self.max = val
            is_max = True
        if val < self.min:
            self.min = val
            is_min = True
        new = copy.copy(self)
        for listener in self.listeners:
            listener.notify(prev, new)
            del prev, new
        return (is_max, is_min)
    def save(self):
        with open(os.path.join(self.path, "{}.txt".format(self.name)), "w") as file:
            file.write("max: {0:.4f}\nmin: {1:.4f}".format(self.max, self.min))
        if self.save_all:
            np.savetxt(os.path.join(self.path, "{}.csv".format(self.name)), self.data, delimiter=',')
    def plot(self, scatter=True):
        assert self.save_all
        plot_1D(self.args, self.data, scatter=scatter, surfix=self.surfix, name=self.name, x_label=self.x_label, y_label=self.name)
    def plot_over(self, rhs, scatter=True, x_label=True, y_label=True, title=None, save=True):
        assert self.save_all and rhs.save_all
        plot_2D(self.args, self.data, rhs.data, scatter=scatter, surfix=self.surfix, name=self.name, x_label=self.x_label, y_label=self.name)
    def attach_combo_listener(self, f, threshold=1):
        listener = ComboListener(f, threshold)
        self.listeners.append(listener)
        return listener

class Listener:
    def __init__(self):
        self.value = None
    def listen(self):
        return self.value

class ComboListener(Listener):
    def __init__(self, f, threshold):
        super(ComboListener, self).__init__()
        self.f = f
        self.threshold = threshold
        self.cnt = 0
        self.value = False
    def notify(self, prev, new):
        if self.f(prev, new):
            self.cnt += 1
        else:
            self.cnt = 0
        if self.cnt >= self.threshold:
            self.value = True
    
    
# convert idice of words to real words
def seq2sen(batch, vocab):
    sen_list = []

    for seq in batch:
        seq_strip = seq[:seq.index(1)+1]
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list

# shuffle source and target lists in paired manner
def shuffle_list(src, tgt):
    index = list(range(len(src)))
    random.shuffle(index)

    shuffle_src = []
    shuffle_tgt = []

    for i in index:
        shuffle_src.append(src[i])
        shuffle_tgt.append(tgt[i])

    return shuffle_src, shuffle_tgt

# simple metric whether each predicted words match to original ones
def val_check(pred, ans):
    # pred, ans: (batch x length)
    batch, length = pred.shape
    num_correct = (pred == ans).sum()
    total = batch * length
    
    return num_correct, total

# save data, such as model, optimizer
def save(args, surfix, data):
    torch.save(data, os.path.join(args.path, args.ckpt_dir, args.name, "{}.pt".format(surfix)))

# load data, such as model, optimizer
def load(args, surfix, map_location='cpu'):
    return torch.load(os.path.join(args.path, args.ckpt_dir, "{}.pt".format(surfix)), map_location=map_location)

# draw 1D plot
def plot_1D(args, x, scatter=True, surfix='.', name='noname', x_label=None, y_label=None):
    if scatter:
        plot = plt.scatter(range(1, 1+ len(x)), x)
    else:
        plot = plt.plot(range(1, 1 + len(x)), x)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(os.path.join(args.path, args.result_dir, args.name, surfix, "{}.jpg".format(name)))
    plt.close(plt.gcf())
    
# draw 2D plot
def plot_2D(args, x, y, scatter=True, surfix='.', name='noname', x_label=None, y_label=None):
    assert len(x) == len(y)
    if scatter:
        plot = plt.scatter(x, y)
    else:
        plot = plt.plot(x, y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(os.path.join(args.path, args.result_dir, args.name, surfix, "{}.jpg".format(name)))
    plt.close(plt.gcf())
    