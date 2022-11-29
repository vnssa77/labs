#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 1.

This first introductory set of exercises is largely intended
as a warm up and practice session. It is an opportunity to check
that you have a functioning Python 3 system with the requisite libraries, to get
a feel for some basic data manipulation and plotting, and to ensure that
everything makes sense and runs smoothly.

Add your code as specified below. You shouldn't need to load further external
code that isn't already explicitly imported.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_1.py

A 4-panel figure, `week_1.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys
import os
import os.path
import argparse

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import utils

np.random.seed(42)

# ADD YOUR CODE BELOW
# np.random.seed(42)

# -- Question 1 --

def generate_noisy_linear(num_samples, weights, sigma, limits, rng):
    """
    Draw samples from a linear model with additive Gaussian noise.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples output values
    """

    num_features = len(weights) - 1
    errors = np.random.normal(0, sigma, size=(num_samples))

    x = rng.uniform(low=limits[0], high=limits[1],
                    size=(num_samples, num_features))
    X = np.c_[np.ones(num_samples), x]

    y = np.matmul(X, weights) + errors
    return x, y


def plot_noisy_linear_1d(axes, num_samples, weights, sigma, limits, rng):
    """
    Generate and plot points from a noisy single-feature linear model,
    along with a line showing the true (noiseless) relationship.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        None
    """
    assert (len(weights) == 2)
    x, y = generate_noisy_linear(num_samples, weights, sigma, limits, rng)

    axes.plot(x, y, marker='o', linestyle=' ', color='red')
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    axes.plot(x, y_pred, marker='', linestyle='-', label='Fitted Line')
    axes.set_xlabel('$x$')
    axes.set_ylabel('$y$')
    axes.set_title('1D Linear Regression')


def plot_noisy_linear_2d(axes, resolution, weights, sigma, limits, rng):
    """
    Produce a plot illustrating a noisy two-feature linear model.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        resolution: how densely should the model be sampled?
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        None
    """
    
    X, y = generate_noisy_linear(resolution, weights, sigma, limits, rng)
    
    x0 = np.linspace(limits[0], limits[1], resolution)
    x1 = np.linspace(limits[0], limits[1], resolution)
    x0, x1 = np.meshgrid(x0, x1)
    

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X, y)
    
    w1, w2 = model.coef_
    c = model.intercept_

    output = w1*x0 + w2*x1 + c

    axes.imshow(output,  extent=[np.min(x0), np.max(x0), np.min(x1), np.max(x1)])
    axes.set_title('2D Linear Regression')
    axes.set_xlabel('$x_0$')
    axes.set_ylabel('$x_1$')
# -- Question 2 --

def generate_linearly_separable(num_samples, weights, limits, rng):
    """
    Draw samples from a binary model with a given linear
    decision boundary.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples binary labels
    """

    num_features = len(weights) - 1
    x = rng.uniform(low=limits[0], high=limits[1],
                    size=(num_samples, num_features))

    X = np.c_[np.ones(num_samples), x]

    y = np.where(np.matmul(X, weights) >= 0, 1, 0)

    return x, y


def plot_linearly_separable_2d(axes, num_samples, weights, limits, rng):
    """
    Plot a linearly separable binary data set in a 2d feature space.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        None
    """
    assert (len(weights) == 3)
    X, y = generate_linearly_separable(num_samples, weights, limits, rng)

    X_1 = X[np.where(y > 0)]
    X_0 = X[np.where(y == 0)]

    axes.plot(X_1[:, 0], X_1[:, 1], marker='x', linestyle='', c='red', label = 'Positive')
    axes.plot(X_0[:, 0], X_0[:, 1], marker='+', linestyle='', c='blue', label = 'Negative')

    import sklearn.linear_model
    model = sklearn.linear_model.LogisticRegression()
    model.fit(X, y)

    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = min(X[:,0]), max(X[:,0])
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    
    point1, point2 = np.mean(xd), np.mean(yd)

    axes.plot(xd, yd, 'k', lw=1, ls='--')
    axes.arrow(point1, point2, w1[0], w2[0], head_width = 0.1)
    axes.set_title('Linearly Seperable Binary Data')
    axes.set_xlabel('$x_0$')
    axes.set_ylabel('$x_1$')
    axes.legend()

# -- Question 3 --

def random_search(function, count, num_samples, limits, rng):
    """
    Randomly sample from a function of `count` features and return
    the best feature vector found.

    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_samples: the number of samples to generate & search
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        x: a vector of length count, containing the found features
    """
    X = rng.uniform(low=limits[0], high=limits[1], size=(num_samples, count))
    y = function(X)
    
    i = np.argmin(y)
    #print(X[i])
    
    return X[i]


def grid_search(function, count, num_divisions, limits):
    """
    Perform a grid search for a function of `count` features and
    return the best feature vector found.

    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_divisions: the number of samples along each feature
            dimension (including endpoints)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i

    # Returns
        x: a vector of length count, containing the found features
    """
    
    X, y = utils.grid_sample(function, count, num_divisions, limits)
    
    y = y.reshape(-1,1)
    X = X.reshape(-1,2)
    index = np.argmin(y)
    x = X[index]
    

    return x


def plot_searches_2d(axes, function, limits, resolution,
                     num_divisions, num_samples, rng, true_min=None):
    """
    Plot a 2D function along with minimum values found by
    grid and random searching.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        function: a function taking a single input array of
            shape (..., 2), where the last dimension
            indexes the features
        limits: a tuple (low, high) specifying the value
            range of both input features x1 and x2
        resolution: number of samples along each side
            (including endpoints) for an image representation
            of the function
        num_divisions: the number of samples along each side
            (including endpoints) for a grid search for
            the function minimum
        num_samples: number of samples to draw for a random
            search for the function minimum
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
        true_min: an optional (x1, x2) tuple specifying
            the location of the actual function minimum

    # Returns
        None
    """

    rndMin = random_search(function, 2, num_samples, limits, rng)
    gridMin = grid_search(function, 2, num_divisions, limits)
    
    X, y = utils.grid_sample(function, 2, resolution, limits, rng)
    x0 = X[:,0]
    x1 = X[:,1]
    print(np.min(x0), np.max(x0), np.min(x1), np.max(x1))
    axes.imshow(y,  extent=[np.min(x0), np.max(x0), np.min(x1), np.max(x1)])
    axes.plot(rndMin[0], rndMin[1], marker = 'x', label = 'Random search minimum')
    axes.plot(gridMin[0], gridMin[1], marker = '+', label = 'Grid search minimum')
    if true_min is not None:
        axes.plot(true_min[0], true_min[1], marker = '.', label = 'True minimum')
    axes.legend()
    axes.set_title('Sampling Search')

# TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(
        description='week 1 labwork script for COMP0088')
    ap.add_argument(
        '-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('file', help='name of output file to produce',
                    nargs='?', default='week_1.pdf')
    return ap.parse_args()


def test_func(X):
    """
    Simple example function of 2 variables for
    testing grid & random optimisation.
    """
    return (X[..., 0]-1)**2 + X[..., 1]**2 + 2 * np.abs((X[..., 0]-1) * X[..., 1])


WEIGHTS = np.array([0.5, -0.4, 0.6])
LIMITS = (-5, 5)

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)

    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(nrows=2, ncols=2)

    print('Q1: noisy continuous data')
    print('plotting 1D data')
    plot_noisy_linear_1d(axs[0, 0], 50, WEIGHTS[1:], 0.5, LIMITS, rng)
    print('plotting 2D data')
    plot_noisy_linear_2d(axs[0, 1], 100, WEIGHTS, 0.2, LIMITS, rng)

    print('\nQ2: binary separable data')
    print('plotting 2D labelled data')
    plot_linearly_separable_2d(
        axs[1, 0], num_samples=100, weights=WEIGHTS, limits=LIMITS, rng=rng)

    print('\nQ3: searching for a minimiser')
    print('plotting searches')
    plot_searches_2d(axs[1, 1], test_func, limits=LIMITS, resolution=100,
                     num_divisions=10, num_samples=100, rng=rng, true_min=(1, 0))

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
