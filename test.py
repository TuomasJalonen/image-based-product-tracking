#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:44:36 2020

This file is used for testing the system.
If you find this code useful, please cite:
    @citation info

@author: Tuomas Jalonen
"""

import numpy as np
from tensorflow.keras.models import load_model
from scipy.optimize import linear_sum_assignment
import matplotlib.pylab as plt

# This is the path to folder containing .npy-files created with
# Preprocess_data.py
DIR = '/path/'

# This is the path to the trained Keras model we want to use for testing
DIR_MODEL = '/path/'

# Load test images. Note that we assume they are in paired order:
# test_input_A[x] is pair for test_input_B[x]. Thus, later the correct pairs
# will be diagonal in probability tables.
test_input_A = np.load(f'{DIR}/test_input_A.npy')
test_input_B = np.load(f'{DIR}/test_input_B.npy')

# Let's calculate first probabilities for all the possible test image
# combinations. That is, for 500 pairs, there will be 500x500 probabilities.

# First, let's load the probabilities from file if possible:
try:
    test_predictions = np.load(f'{DIR}/test_predictions.npy')

# If it's not possible, do:
except:

    # create list for predictions
    test_predictions = []

    # Load the Keras model to be used in testing
    model = load_model(DIR_MODEL)

    # Loop test size
    for i in range(len(test_input_A)):

        # Print progress
        print(i)

        # Get image_A and broadcast it to test size, e.g. (500, 224, 224, 3).
        # The idea here is to test this one image with all images in
        # test_input_B, and Keras requires the same input size.
        img_A = test_input_A[i]
        img_A = np.broadcast_to(img_A, (len(test_input_A), 224, 224, 3))

        # Predict the probabilities
        output = model.predict([img_A, test_input_B])

        # Fix the output and append to test_predictions
        output = np.delete(output, 0, axis=1)
        output = np.squeeze(output, axis=1)
        test_predictions.append(output)

    # Transform test_predictions as numpy array and save for later use
    test_predictions = np.array(test_predictions)
    np.save(f'{DIR}/test_predictions.npy', test_predictions)

# This is a list of cluster sizes we want to test
cluster_sizes = [1, 2, 4, 5, 10, 20, 50]

# Create dictionaris for avg accuracies for both decision rules
avg_accuracies_greedy = {}
avg_accuracies_hungarian = {}

def calc_greedy(prediction_table):
    """
    This function calculates number of correct and incorrect decisions using
    Greedy algorithm.

    Parameters
    ----------
    prediction_table : two-dimensional matrix of probabilities

    Returns
    -------
    correct_on_cluster : number of correct decisions
    incorrect_on_cluster : number of incorrect decisions

    """

    # Initialize correct and incorrect values to zero and list for predicted
    # items
    correct_on_cluster = 0.0
    incorrect_on_cluster = 0.0
    predicted = []

    # Loop table
    for k in range(len(prediction_table[0])):
        # Select row
        row = prediction_table[k]

        # if a column has been dropped, replace the value with NaN
        for l in predicted:
            row[l] = np.nan
        # Select the highest probability as pair from values that are not NaNs
        predicted_index = np.nanargmax(row)

        # If the predicted_index is diagonal, it is correct.
        if predicted_index == k:
            correct_on_cluster += 1.0
        else:
            incorrect_on_cluster += 1.0

        # Add the predicted index to the list "predicted" so it will be
        # excluded from future decisions
        predicted.append(predicted_index)

    return correct_on_cluster, incorrect_on_cluster

def calc_hungarian(prediction_table):
    """
    This function calculates number of correct and incorrect decisions using
    Hungarian Algorithm

    Parameters
    ----------
    prediction_table : two-dimensional matrix of probabilities

    Returns
    -------
    correct_on_cluster : number of correct decisions
    incorrect_on_cluster : number of incorrect decisions

    """

    # Initialize correct and incorrect values to zero
    correct_on_cluster = 0.0
    incorrect_on_cluster = 0.0

    # Calculate predicted indices using Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(prediction_table, maximize=True)

    # Loop predicted row indices
    for m in row_ind:
        # If the predicted row index is the same as cpredicted column index,
        # they are diagonal, thus making the prediction correct.
        if m == col_ind[m]:
            correct_on_cluster += 1.0
        else:
            incorrect_on_cluster += 1.0

    return correct_on_cluster, incorrect_on_cluster

# print(predictions)
print('Testing started')

# Loop cluster_sizes:
for cluster_size in cluster_sizes:

    print('Cluster size', cluster_size, 'started')

    # Initialize lists for test accuracies
    test_accuracies_greedy = []
    test_accuracies_hungarian = []

    # Loop 1000 tests
    for i in range(1000):

        # Initialize correct and incorrect values of the test to 0
        correct_on_test_greedy = 0
        incorrect_on_test_greedy = 0
        correct_on_test_hungarian = 0
        incorrect_on_test_hungarian = 0

        # Create random clusters of 5 test images by creating clusters
        clusters = np.random.permutation(len(test_input_A)).reshape(
            (int(len(test_input_A) / cluster_size), cluster_size)).astype(int)

        # Loop test clusters
        for cluster in clusters:

            # Get probability tables for the cluster
            cluster_predictions_greedy = test_predictions[:, cluster][cluster, :]
            cluster_predictions_hungarian = np.copy(cluster_predictions_greedy)

            # Calculate number of correct and incorrect predictions
            correct_greedy, incorrect_greedy = calc_greedy(cluster_predictions_greedy)
            correct_hungarian, incorrect_hungarian = calc_hungarian(cluster_predictions_hungarian)

            # Add to the test values
            correct_on_test_greedy += correct_greedy
            incorrect_on_test_greedy += incorrect_greedy

            correct_on_test_hungarian += correct_hungarian
            incorrect_on_test_hungarian += incorrect_hungarian

        # Calculate test accuracies
        test_accuracy_greedy = float(correct_on_test_greedy / (correct_on_test_greedy + incorrect_on_test_greedy))
        test_accuracy_hungarian = float(correct_on_test_hungarian / (correct_on_test_hungarian + incorrect_on_test_hungarian))

        # Add test accuracies to the lists
        test_accuracies_greedy.append(test_accuracy_greedy)
        test_accuracies_hungarian.append(test_accuracy_hungarian)

    # Calculate average accuracy, standard deviation and append to cluster
    # accuracies
    average_accuracy_greedy = np.mean(test_accuracies_greedy)
    std_accuracy_greedy = np.std(test_accuracies_greedy)
    avg_accuracies_greedy[cluster_size] = average_accuracy_greedy
    print('avg_acc_greedy', average_accuracy_greedy, 'cluster_size', cluster_size)
    print('std_acc_greedy', std_accuracy_greedy, 'cluster_size', cluster_size)

    # Calculate average accuracy, standard deviation and append to cluster
    # accuracies
    average_accuracy_hungarian = np.mean(test_accuracies_hungarian)
    std_accuracy_hungarian = np.std(test_accuracies_hungarian)
    avg_accuracies_hungarian[cluster_size] = average_accuracy_hungarian
    print('avg_acc_hungarian', average_accuracy_hungarian, 'cluster_size', cluster_size)
    print('std_acc_hungarian', std_accuracy_hungarian, 'cluster_size', cluster_size)

# Plot cluster accuracies
lists_greedy = sorted(avg_accuracies_greedy.items())
cluster_sizes, avg_accuracies_greedy = zip(*lists_greedy)
plt.plot(cluster_sizes, avg_accuracies_greedy, label='Greedy')

# Plot cluster accuracies
lists_hungarian = sorted(avg_accuracies_hungarian.items())
cluster_sizes, avg_accuracies_hungarian = zip(*lists_hungarian)
plt.plot(cluster_sizes, avg_accuracies_hungarian, label='Hungarian')

# Save cluster accuracies of both rules to the same plot
plt.ylabel('Average Accuracy')
plt.xlabel('Test cluster size')
plt.legend()
plt.xticks(range(0, 51, 5))
plt.savefig('Test_accuracies.png')
