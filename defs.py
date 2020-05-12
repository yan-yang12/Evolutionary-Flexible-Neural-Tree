from FlexTree import FlexTree
import random
import copy
import matplotlib.pyplot as plt


def ini_pop(i, X_train, y_train):
    new_tree = FlexTree(X_train)
    new_tree.train(X_train, y_train)
    return new_tree


# Regularization paramters should be changed here
# score function
def eval_tr(tr, X, y):
    return tr.evaluate(X, y) - max(tr.out_neuron.n_leaf-6, 0) ** 2 * 0.003 - max(tr.out_neuron.n_flex-3, 0) ** 2 * 0.001


def mutate_tr(tree, X_train, y_train, MUTATAION_PROBA):
    tree.mutate(MUTATAION_PROBA)
    tree.train(X_train, y_train)


def reproduce_tr(tree, REPRO_RATE, X_train, y_train):
    if random.random() < REPRO_RATE:
        new_offspring = copy.deepcopy(tree).mutate(1)
        new_offspring.train(X_train, y_train)
        return new_offspring
    return None
