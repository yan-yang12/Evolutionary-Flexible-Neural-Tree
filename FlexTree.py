import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import random
import copy
from datetime import datetime
from tqdm import tqdm

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from sklearn.model_selection import train_test_split

import defs


class FlexNeuron:
    def __init__(self, n_features, n_child, a, b, neuron_ratio, depth, empty = True):
        self.n_features = n_features
        self.depth = depth
        self.n_flex = 1
        self.n_leaf = 0
        self.a = nn.Parameter(a, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        self.weights = nn.Parameter(torch.rand((n_child, 1)), requires_grad=True)  # weight is n_child by 1
        torch.nn.init.xavier_normal_(self.weights)

        self.n_child = n_child
        self.neuron_ratio = neuron_ratio
        self.input_list = self.create_children(n_child, neuron_ratio)

    def forward(self, X_matrix):
        out = torch.tensor(np.zeros((X_matrix.shape[0], self.n_child))).float()
        for idx in range(len(self.input_list)):
            out[:, idx] = self.input_list[idx].forward(X_matrix).reshape((-1))  # concatenate input to n by n_child
        out = torch.matmul(out, self.weights) - self.a
        out = torch.div(out, self.b)
        out = torch.exp(-out ** 2)
        return out

    def create_children(self, n_child, neuron_ratio):
        children_list = []
        for i in range(n_child):
            if (random.random() < neuron_ratio) and (self.depth > 1):
                new_neuron = FlexNeuron(self.n_features, random.randint(2, max(n_child + 2, 3)), self.a, self.b,
                                        neuron_ratio, self.depth - 1)
                children_list.append(new_neuron)
                self.n_flex += new_neuron.n_flex
                self.n_leaf += new_neuron.n_leaf
            else:
                children_list.append(LeafNode(self.n_features))
                self.n_leaf += 1
        return children_list

    def recal_n_leaf(self):
        self.n_leaf = 0
        for child in self.input_list:
            if isinstance(child, FlexNeuron):
                self.n_leaf += child.recal_n_leaf()
            else:
                self.n_leaf += 1
        return self.n_leaf

    def recal_n_flex(self):
        self.n_flex = 1
        for child in self.input_list:
            if isinstance(child, FlexNeuron):
                self.n_flex += child.recal_n_flex()
        return self.n_flex

    def get_param(self):
        out = [self.a, self.b, self.weights]
        for child in self.input_list:
            if isinstance(child, FlexNeuron):
                out.extend(child.get_param())
        return out

    def get_nth_FN(self, nth):
        idx = 0
        for child in self.input_list:
            if isinstance(child, FlexNeuron):
                if nth == 0:
                    return self.input_list[idx]
                nth -= 1
                temp = child.get_nth_FN(nth)
                if isinstance(temp, FlexNeuron):
                    nth -= 1
                    return temp
            idx += 1
        return None

    def overwrite_nth_FN(self, nth, new_FN):
        idx = 0
        for child in self.input_list:
            if nth == 0 and isinstance(child, FlexNeuron):
                self.input_list[idx] = new_FN
                nth -= 1
            if isinstance(child, FlexNeuron):
                nth -= 1
                child.overwrite_nth_FN(nth, new_FN)
            if nth < 0:
                return
            idx += 1

    def change_one_terminal(self, to_change):
        idx = 0
        for child in self.input_list:
            if to_change == 0 and isinstance(child, LeafNode):
                self.input_list[idx] = LeafNode(self.n_features)
                to_change = -1
            if isinstance(child, LeafNode):
                to_change -= 1
            if isinstance(child, FlexNeuron):
                child.change_one_terminal(to_change)
            if to_change < 0:
                return
            idx += 1

    def change_all_terminal(self):
        idx = 0
        for child in self.input_list:
            if isinstance(child, LeafNode):
                self.input_list[idx] = LeafNode(self.n_features)
            if isinstance(child, FlexNeuron):
                child.change_all_terminal()
            idx += 1

    def grow(self, to_grow):
        idx = 0
        for child in self.input_list:
            if to_grow == 0 and isinstance(child, LeafNode):
                self.input_list[idx] = FlexNeuron(self.n_features, random.randint(2, max(self.n_child, 3)),
                                                  self.a, self.b, self.neuron_ratio, self.depth - 1)
                to_grow = -1
            if isinstance(child, LeafNode):
                to_grow -= 1
            if isinstance(child, FlexNeuron):
                child.grow(to_grow)
            if to_grow < 0:
                return
            idx += 1

    def prune(self, to_pruned):
        idx = 0
        for child in self.input_list:
            if to_pruned == 0 and isinstance(child, FlexNeuron):
                self.input_list[idx] = LeafNode(self.n_features)
                to_pruned = -1
            if isinstance(child, FlexNeuron):
                to_pruned -= 1
                child.prune(to_pruned)
            if to_pruned < 0:
                return
            idx += 1

    def __str__(self):
        child_str = ''
        for child in self.input_list:
            child_str += str(child) + ','
        return f'FN+{self.n_child};[{child_str} depth:{self.depth}] \n'


class LeafNode:
    def __init__(self, n_features):
        self.idx = random.randint(0, n_features - 1)

    def forward(self, X_matrix):
        return X_matrix[:, self.idx].clone().detach()

    def __str__(self):
        return str(self.idx)


class FlexTree(nn.Module):

    def __init__(self, x, max_depth=3, neuron_ratio=0.3, empty=False):
        super(FlexTree, self).__init__()
        if empty:
            self.max_depth = 0
            self.out_neuron = None
            self.optimizer = None
            self.criterion = None
            return

        self.max_depth = max_depth
        x_len = x.shape[1]
        n_0 = random.randint(4, max(x_len // 3, 6))
        a = torch.tensor(0.2)
        b = torch.tensor(0.5)

        self.out_neuron = FlexNeuron(x_len, n_0, a, b, neuron_ratio, max_depth)
        self.optimizer = torch.optim.Adam(self.get_all_param(), lr=1e-2)
        self.criterion = nn.BCELoss()

    # over ride deep copy for faster computation
    def __deepcopy__(self, memodict={}):
        copy_object = FlexTree(x=None, empty=True)
        copy_object.max_depth = self.max_depth
        copy_object.out_neuron = copy.deepcopy(self.out_neuron)
        copy_object.optimizer = copy.deepcopy(self.optimizer)
        copy_object.criterion = copy.deepcopy(self.criterion)
        return copy_object

    def forward(self, X_matrix):
        return self.out_neuron.forward(X_matrix)

    def recal_stat(self):
        self.out_neuron.recal_n_flex()
        self.out_neuron.recal_n_leaf()

    def mutate(self, proba):
        if random.random() < proba:
            func_list = [self.change_one_terminal, self.change_all_terminal, self.grow, self.prune]
            mutation = random.choice(func_list)
            mutation()
            self.recal_stat()
        return self

    def change_one_terminal(self):
        to_change = random.randint(0, self.out_neuron.n_leaf - 2)
        self.out_neuron.change_one_terminal(to_change)

    def change_all_terminal(self):
        self.out_neuron.change_all_terminal()

    def grow(self):
        # number of leafs to travel to grow the target
        to_grow = random.randint(0, self.out_neuron.n_leaf - 2)
        self.out_neuron.grow(to_grow)

    def prune(self):
        # if the tree has only one neuron, do nothing
        if self.out_neuron.n_flex == 1:
            return

        # number of flex neurons to travel to delete the target
        to_pruned = random.randint(0, self.out_neuron.n_flex - 2)
        self.out_neuron.prune(to_pruned)

    def crossover(self, other_tree):
        # if either is a single tree, do nothing
        if (self.out_neuron.n_flex == 1) or (other_tree.out_neuron.n_flex == 1):
            return

        to_cross_1 = random.randint(0, self.out_neuron.n_flex - 2)
        temp_FN_1 = copy.deepcopy(self.out_neuron.get_nth_FN(to_cross_1))
        to_cross_2 = random.randint(0, other_tree.out_neuron.n_flex - 2)
        temp_FN_2 = copy.deepcopy(other_tree.out_neuron.get_nth_FN(to_cross_2))
        if temp_FN_1 is None or temp_FN_2 is None:
            return
        self.out_neuron.overwrite_nth_FN(to_cross_1, temp_FN_2)
        other_tree.out_neuron.overwrite_nth_FN(to_cross_2, temp_FN_1)
        self.recal_stat()
        other_tree.recal_stat()

    # get all trainable parameters for optimizer
    def get_all_param(self):
        return self.out_neuron.get_param()

    def train(self, X_train, y_train, nIter=1200, verbose=False):
        for epoch in range(nIter):
            self.optimizer.zero_grad()
            y_predicted = self.forward(X_train)
            loss = self.criterion(y_predicted, y_train)
            loss.backward()
            self.optimizer.step()
            if verbose and (epoch + 1) % 100 == 0:
                print(f'epoch: {epoch + 1}, loss: {loss.item():.4f}, accuracy: {self.evaluate(X_train, y_train)}')

    def evaluate(self, X, y):
        with torch.no_grad():
            y_predicted = self.forward(X).round()
            accuracy = torch.eq(y_predicted, y).sum() / float(y.shape[0])
        return accuracy

    def confusion_matrix(self, X, y):
        with torch.no_grad():
            y_predicted = self.forward(X).round()

        return confusion_matrix(y, y_predicted)

    def __str__(self):
        return str(self.out_neuron)


def main(X_train, X_test, y_train, y_test):
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).reshape((-1, 1))
    y_test = torch.from_numpy(y_test.astype(np.float32)).reshape((-1, 1))
    # -------------------------------EVOLUTION-------------------------------
    pop_size = 20
    n_generation = 1
    CROSS_OVER_PROBA = 0.6
    REPRO_RATE = 0.4
    MUTATAION_PROBA = 0.4
    SELECTION_THRES = 40

    pop_hist = []
    median_hist = []
    max_hist = []
    upper_hist = []
    lower_hist = []
    # initialize population
    print('Initializing population...')
    population = []
    with Pool() as p:
        fn = partial(defs.ini_pop, X_train=X_train, y_train=y_train)
        for tr in tqdm(p.imap_unordered(fn, range(pop_size)), total=pop_size):
            population.append(tr)

    print('Finished initialization, starting evolution...')
    for gen in range(n_generation):
        print(f'-------------------GEN {gen+1}-------------------')
        with Pool() as p:
            fn = partial(defs.eval_tr, X=X_train, y=y_train)
            score_list = p.map(fn, population)

        # Calculate deletion thresthold and log the state of population
        threshold = np.percentile(score_list, SELECTION_THRES)
        median_hist.append(np.percentile(score_list, 50))
        max_hist.append(np.max(score_list))
        upper_hist.append(np.percentile(score_list, 75))
        lower_hist.append(np.percentile(score_list, 25))
        print(f'Mean metric is {np.mean(score_list):.4f}')

        # delete inferior trees
        for idx in np.flip(np.where(score_list < threshold)[0]):
            del population[idx]
            score_list = np.delete(score_list, idx)

        # save a copy of best tree for next generation
        best_tree = copy.deepcopy(population[np.argmax(score_list)])
        print(f'Best metric is {np.max(score_list):.4f}')
        print('Crossover...')
        for idx in range(len(population)):
            if random.random() < CROSS_OVER_PROBA:
                other_tree_idx = idx
                while other_tree_idx == idx:
                    other_tree_idx = random.randint(0, len(population)-1)
                population[idx].crossover(population[other_tree_idx])

        # Mutate and reproduce
        with Pool() as p:
            print('Mutating...')
            fn = partial(defs.mutate_tr, X_train=X_train, y_train=y_train, MUTATAION_PROBA=MUTATAION_PROBA)
            p.map(fn, population)
            if len(population) > 2*pop_size:
                print('Population too large, skipping reproduction...')
                new_offsprings = []
            else:
                print('Reproducing...')
                fn = partial(defs.reproduce_tr, REPRO_RATE=REPRO_RATE, X_train=X_train, y_train=y_train)
                new_offsprings = p.map(fn, population)
        population.extend(list(filter(None.__ne__, new_offsprings)))
        population.append(best_tree)

        # print the population number
        pop_hist.append(len(population))
        print(f'Population: {len(population)}')

        # if population is too small, do reproduction once more
        if len(population) < 0.4 * pop_size:
            print('Population too small, reproducing more...')
            with Pool() as p:
                new_offsprings = p.map(fn, population)
            population.extend(list(filter(None.__ne__, new_offsprings)))

    print('-------------------RESULT-------------------')
    print('Best Tree Structure is')
    best_tree = population[np.argmax(score_list)]
    print(best_tree)
    print(f'Best Accuracy is {best_tree.evaluate(X_test, y_test):.4f}')
    print(f'Confusion Matrix:\n {best_tree.confusion_matrix(X_test, y_test)}')

    # make a log locally
    filename = "log\\" + datetime.now().strftime("%d-%H-%M") + ".txt"
    with open(filename, 'w+') as f:
        write_str = str(best_tree) + '\n'
        write_str += f'Best Accuracy is {best_tree.evaluate(X_test, y_test):.4f} \n'
        write_str += f'Confusion Matrix:\n {best_tree.confusion_matrix(X_test, y_test)}'
        f.write(write_str)

    # plot evolution graph
    plt.plot(range(n_generation), max_hist, label='Maximum Performance')
    plt.plot(range(n_generation), median_hist, label='Median Performance')
    plt.fill_between(range(n_generation), lower_hist, upper_hist, alpha=0.2, label='Interquartile Range')
    plt.title('Evolution of the Population')
    plt.ylabel('Performance (Complexity Adjusted)')
    plt.xlabel('Generation')
    plt.legend(loc=4)
    plt.show()

if __name__ == '__main__':

    # Synthetic dataset
    x = np.random.random((1000, 8))
    y = 1 / (1 + np.exp(-(x[:, 1] ** 2 + x[:, 2] * x[:, 4] + np.sin(x[:, 3]) - 5 * x[:, 7] + np.sqrt(x[:, 5] * 3))))
    y = np.round(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Wine dataset
    # df = pd.read_csv('winequality-red.csv', sep=";")
    # df['label'] = df['quality'] >= 6
    # df = df * 1
    # y = df['label'].to_numpy()
    # del df['label']
    # del df['quality']
    # x = df.to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    main(X_train, X_test, y_train, y_test)