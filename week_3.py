#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 3.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_3.py

A 4-panel figure, `week_3.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys
import os
import os.path
import argparse
import pprint

import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils


# ADD YOUR CODE BELOW

# -- Question 1 --

def nearest_neighbours_predict(train_X, train_y, test_X, neighbours=1):
    """
    Predict labels for test data based on neighbourhood in
    training set.

    # Arguments:
        train_X: an array of sample data for training, where rows
            are samples and columns are features.
        train_y: vector of class labels corresponding to the training
            samples, must be same length as number of rows in X
        test_X: an array of sample data to generate predictions for,
            in same layout as train_X.
        neighbours: how many neighbours to canvass at each test point

    # Returns
        test_y: predicted labels for the samples in test_X
    """
    assert (train_X.shape[0] == train_y.shape[0])
    assert (train_X.shape[1] == test_X.shape[1])

    test_y = []
    for x in test_X:
        dist = []
        for x_train in train_X:
            dist.append(np.linalg.norm(x_train - x))
        neighbors_ind = np.argsort(dist)[:neighbours]
        labels = [train_y[i] for i in neighbors_ind]
        test_y.append(utils.vote(labels))

    return np.array(test_y)


# -- Question 2 --

def misclassification(y, cls, weights=None):
    """
    Calculate (optionally-weighted) misclassification error for
    a given set of labels if assigned the given class.

    # Arguments
        y: a set of class labels
        cls: a candidate classification for the set
        weights: optional weights vector specifying relative
            importance of the samples labelled by y

    # Returns
        err: the misclassification error of the candidate labels
    """

    indicator = [1 if y[i] != cls else 0 for i in range(len(y))]

    if weights is None:
        weights = np.repeat(1/len(y), len(indicator))

    err = np.dot(weights, indicator)

    return err


def decision_node_split(X, y, cls=None, weights=None, min_size=3):
    """
    Find (by brute force) a split point that best improves the weighted
    misclassification error rate compared to the original one (or not, if
    there is no improvement possible).

    Features are assumed to be numeric and the test condition is
    greater-or-equal.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this

    # Returns:
        feature: index of the feature to test (or None, if no split)
        thresh: value of the feature to test (or None, if no split)
        c0: class assigned to the set with feature < thresh
            (or None, if no split)
        c1: class assigned to the set with feature >= thresh
            (or None, if no split)
    """
    assert (X.shape[0] == len(y))

    """  feature = None
    thresh = None 
    
    if cls is None:
        cls = utils.vote(y)
    
    if len(np.delete(y, np.where(y == cls))) == 0:
        return None, None, None, None
    else:
        c1 = utils.vote(np.delete(y, np.where(y == cls)))
    loss = misclassification(y, cls, weights)
    gini_loss = utils.gini_impurity(y)
    split = False
    
    for f in range(len(X.T)):
        
        tests = np.unique(X[:,f])
        
        for t in tests:
            #print(y[X[:, f] >= t], t)
            y_pred = y[X[:, f] >= t]
            
            if len(y_pred) != 0:
                cls = utils.vote(y_pred)
                error = misclassification(y_pred, cls, weights)
            gini_error = utils.gini_impurity(y_pred)
            
            if error < loss and gini_error < gini_loss:
                split = True
                loss = error
                gini_loss = gini_error
                feature = f
                thresh = t
    
    if split == False or len(X[:,f] >= thresh) < min_size:
        return None, None, None, None
    else: 
        return feature, thresh, cls, c1   
    """
    if len(y) < min_size * 2:
        return None, None, None, None

    if cls is None:
        cls = utils.vote(y)

    if weights is None:
        weights = np.ones(len(y))/len(y)

    g = misclassification(y, cls=cls, weights=weights)
    if g == 0:
        return None, None, None, None

    best_feat, best_thresh = None, None
    best_c0, best_c1 = None, None
    best_improvement = 0

    for feat in range(X.shape[-1]):
        for thresh in X[:, feat]:
            set1 = X[:, feat] >= thresh
            set0 = ~set1

            if (np.sum(set0) < min_size) or (np.sum(set1) < min_size):
                continue

            y0 = y[set0]
            y1 = y[set1]

            w0 = weights[set0]
            w1 = weights[set1]

            cc0 = np.unique(y0)
            cc1 = np.unique(y1)

            gg0 = [misclassification(y0, cls=cc, weights=w0) for cc in cc0]
            gg1 = [misclassification(y1, cls=cc, weights=w1) for cc in cc1]

            c0 = cc0[np.argmin(gg0)]
            c1 = cc1[np.argmin(gg1)]

            g0 = np.min(gg0)
            g1 = np.min(gg1)

            improvement = g - (g0 + g1)

            if improvement > best_improvement:
                best_feat = feat
                best_thresh = thresh
                best_improvement = improvement
                best_c0 = c0
                best_c1 = c1

    if best_feat is None:
        return None, None, None, None

    return best_feat, best_thresh, best_c0, best_c1


def decision_tree_train(X, y, cls=None, weights=None,
                        min_size=3, depth=0, max_depth=10):
    """
    Recursively choose split points for a training dataset
    until no further improvement occurs.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
        depth: current recursion depth
        max_depth: maximum allowed recursion depth

    # Returns:
        tree: a dict containing (some of) the following keys:
            'kind' : either 'leaf' or 'decision'
            'class' : the class assigned to this node (leaf)
            'feature' : index of feature on which to split (decision)
            'thresh' : threshold at which to split the feature (decision)
            'below' : a nested tree for when feature < thresh (decision)
            'above' : a nested tree for when feature >= thresh (decision)
    """
    # TODO: implement this

    """     data = X
        labels = y
        pred = cls
        tree = dict()
        
        while depth <= max_depth:
            subtree = {}
            feature, thresh, c0, c1 = decision_node_split(data, labels, pred, weights, min_size)
            
            if feature is not None:
                depth += 1
                X_above = data[data[:,feature] >= thresh]
                X_below = data[data[:,feature] < thresh]
                y_above = labels[data[:,feature] >= thresh]
                y_below = labels[data[:,feature] < thresh]
                
                
                subtree["kind"] = 'decision'
                subtree["feature"] = feature
                subtree['thresh'] = thresh
                subtree['below'] = decision_tree_train(X_below, y_below, c0, weights, min_size, depth, max_depth)
                subtree['above'] = decision_tree_train(X_above, y_above, c1, weights, min_size, depth, max_depth)
            else:
                depth+=1
                subtree['kind'] = 'leaf'
                subtree['class'] = pred
                
        return subtree """
    if cls is None:
        cls = utils.vote(y)

    if depth == max_depth:
        return {'kind': 'leaf', 'class': cls}

    feat, thresh, cls0, cls1 = decision_node_split(
        X, y, cls=cls, weights=weights, min_size=min_size)

    if feat is None:
        return {'kind': 'leaf', 'class': cls}

    set1 = X[:, feat] >= thresh
    set0 = ~set1

    return {'kind': 'decision',
            'feature': feat,
            'thresh': thresh,
            'above': decision_tree_train(X[set1, :], y[set1], cls1, None if weights is None else weights[set1], min_size, depth+1, max_depth),
            'below': decision_tree_train(X[set0, :], y[set0], cls0, None if weights is None else weights[set0], min_size, depth+1, max_depth)}

def predict(tree, x):
        while True:
            if tree['kind'] == 'leaf':
                return tree['class']
            
            tree = tree['above'] if x[tree['feature']] >= tree['thresh'] else tree['below']
            
def decision_tree_predict(tree, X):
    """
    Predict labels for test data using a fitted decision tree.

    # Arguments
        tree: a decision tree dictionary returned by decision_tree_train
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
           
    return np.array([predict(tree, X[i,:]) for i in range(len(X))])


# -- Question 3 --

def random_forest_train(X, y, k, rng, min_size=3, max_depth=10):
    """
    Train a (simplified) random forest of decision trees.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the number of trees in the forest
        rng: an instance of numpy.random.Generator
            from which to draw random numbers        
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth

    # Returns:
        forest: a list of tree dicts as returned by decision_tree_train
    """

    # TODO: implement this
    forest = []
    for i in range(k):
        indices = rng.choice(np.arange(len(X)), size = len(X))
        
        samplesX = X[indices,:]
        samplesY = y[indices]
        
        forest.append(decision_tree_train(samplesX, samplesY, None, None, min_size, 0, max_depth))
    
    return forest


def random_forest_predict(forest, X):
    """
    Predict labels for test data using a fitted random
    forest of decision trees.

    # Arguments
        forest: a list of decision tree dicts
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    
    y = np.array([decision_tree_predict(tree, X) for tree in forest])
    
    return np.array([utils.vote(y[:,i]) for i in range(y.shape[1])])


# -- Question 4 --

def adaboost_train(X, y, k, min_size=1, max_depth=1, epsilon=1e-8):
    """
    Iteratively train a set of decision tree classifiers
    using AdaBoost.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the maximum number of weak classifiers to train
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth -- by default we just
            use decision stumps
        epsilon: threshold below which the error is considered 0

    # Returns:
        trees: a list of tree dicts as returned by decision_tree_train
        alphas: a vector of weights indicating how much credence to
            given each of the decision tree predictions
    """
    # TODO: implement this
    
    weights = np.ones(len(y))/len(y)
    trees = []
    alphas = []
    
    def weighted_error(y_pred, y, weights):
        if weights is None:
            weights = np.ones(len(y))/len(y)
        
        indicator = [1 if y[i] != y_pred[i] else 0 for i in range(len(y))]    
        err = np.dot(weights, indicator)

        return err
    
    for t in range(k):
        trees.append(decision_tree_train(X, y, None, weights, min_size, 0, max_depth))
        print(trees[-1])
        #y_pred = np.array([c0 if X[i, feature] >= thresh else c1 for i in range(len(X)) for i in range(len(X))])
        y_pred = decision_tree_predict(trees[-1], X)
                
        error = weighted_error(y_pred, y, weights)
        if error < epsilon:
            alphas.append(1)
            break
        
        alphas.append(np.log((1-error)/error))
        
        weights = weights * np.exp(alphas[-1] * (y_pred != y))
        
        weights = weights/np.sum(weights)
        
    return trees, alphas


def adaboost_predict(trees, alphas, X):
    """
    Predict labels for test data using a fitted AdaBoost
    ensemble of decision trees.

    # Arguments
        trees: a list of decision tree dicts
        alphas: a vector of weights for the trees
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    # TODO: implement this

    y_pred = []
    for t in range(len(trees)):
        pred = decision_tree_predict(trees[t], X)
        pred[pred == 0] = -1
        
        y_pred.append(pred)
        
    y_pred = np.sum(y_pred, axis = 0)
    
    return np.array([1 if alphas[t] * y_pred[i] > 0 else 0 for i in range(len(y_pred))])


# TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(
        description='week 3 coursework script for COMP0088')
    ap.add_argument(
        '-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples',
                    help='number of samples to use', type=int, default=50)
    ap.add_argument('-k', '--neighbours',
                    help='number of neighbours for k-NN fit', type=int, default=3)
    ap.add_argument('-m', '--min_size',
                    help='smallest acceptable tree node', type=int, default=3)
    ap.add_argument(
        '-w', '--weak', help='how many weak classifiers to train for AdaBoost', type=int, default=10)
    ap.add_argument(
        '-f', '--forest', help='how many trees to train for random forest', type=int, default=10)
    ap.add_argument('-r', '--resolution',
                    help='grid sampling resolution for classification plots', type=int, default=20)
    ap.add_argument(
        '-d', '--data', help='CSV file containing training data', default='week_3_data.csv')
    ap.add_argument('file', help='name of output file to produce',
                    nargs='?', default='week_3.pdf')
    return ap.parse_args()


if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)

    print(f'loading data from {args.data}')
    df = pd.read_csv(args.data)
    X = df[['X1', 'X2']].values[:args.num_samples, :]
    y = df['Multi'].values[:args.num_samples]

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(nrows=2, ncols=2)

    print(f'Q1: checking {args.neighbours}-nearest neighbours fit')
    # this is a fudge -- there's no training phase, so check implementation with a dummy prediction
    dummy = nearest_neighbours_predict(
        X[:2, :], y[:2], X[:2, :], neighbours=args.neighbours)
    if dummy is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(
            axs[0, 0], f'{args.neighbours}-Nearest Neighbours')
    else:
        print(f'Q1: plotting {args.neighbours}-nearest neighbours fit')
        def nn_cls(z): return nearest_neighbours_predict(
            X, y, z, neighbours=args.neighbours)
        utils.plot_classification_map(
            axs[0, 0], nn_cls, X, y, resolution=args.resolution, title=f'{args.neighbours}-Nearest Neighbours')

    print('Q2: testing misclassification error')
    all_right = misclassification(np.ones(3), 1)
    all_wrong = misclassification(np.ones(3), 0)
    fifty_fifty = misclassification(
        np.concatenate((np.ones(3), np.zeros(3))), 1)

    right_msg = 'correct' if np.isclose(all_right, 0) else 'wrong, should be 0'
    wrong_msg = 'correct' if np.isclose(all_wrong, 1) else 'wrong, should be 1'
    fifty_msg = 'correct' if np.isclose(
        fifty_fifty, 0.5) else 'wrong should b 0.5'

    print(f' all right: {all_right} - {right_msg}')
    print(f' all wrong: {all_wrong} - {wrong_msg}')
    print(f' fifty-fifty: {fifty_fifty} - {fifty_msg}')

    print('Q2: fitting decision tree')
    tree = decision_tree_train(X, y, min_size=args.min_size)

    if tree is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(axs[0, 1], 'Decision Tree')
    else:
        print('Q2: plotting decision tree fit')
        def tree_cls(z): return decision_tree_predict(tree, z)
        utils.plot_classification_map(
            axs[0, 1], tree_cls, X, y, resolution=args.resolution, title='Decision Tree')

    print(f'Q3: fitting random forest with {args.forest} trees')
    forest = random_forest_train(
        X, y, args.forest, rng=rng, min_size=args.min_size)

    if forest is None:
        print('random forest not implemented')
        utils.plot_unimplemented(axs[1, 0], 'Random Forest')
    else:
        print('Q3: plotting random forest fit')
        def forest_cls(z): return random_forest_predict(forest, z)
        utils.plot_classification_map(
            axs[1, 0], forest_cls, X, y, resolution=args.resolution, title=f'Random Forest ({args.forest} Trees)')

    print('Q4: fitting adaboost ensemble')
    # swap to binary labels since we're only doing 2-class AdaBoost
    y = df['Binary'].values[:args.num_samples]
    trees, alphas = adaboost_train(X, y, args.weak)

    if trees is None:
        print('adaboost not implemented')
        utils.plot_unimplemented(axs[1, 1], 'AdaBoost')
    else:
        print('Q4: plotting AdaBoost fit')
        def ada_cls(z): return adaboost_predict(trees, alphas, z)
        utils.plot_classification_map(
            axs[1, 1], ada_cls, X, y, resolution=args.resolution, title=f'AdaBoost ({args.weak} Stumps)')

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
