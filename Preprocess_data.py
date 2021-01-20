#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:02:20 2020

@author: Tuomas Jalonen

This file is used for building the preprocessing the data.
If you find this code useful, please cite:
    @citation info

"""
import os
import pandas as pd
import cv2
import numpy as np

def process_img(path, size):
    """

    Parameters
    ----------
    path : Complete path to the image
    size : target image size, e.g. (224, 224)

    Returns
    -------
    img : Preprocessed image

    """

    img = cv2.imread(path, 1)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_32F)
    img = cv2.resize(img, size)

    return img

# Define targeted image size
IMG_SIZE = (224, 224)

# Define how many image pairs are saved for validation and testing. Rest is
# used for training.
TEST_SIZE = 500
VAL_SIZE = 200

# Define path to CSV-file containing which images are pairs
PATH_TO_CSV = '/path/'

# Define path to folder containing wet images
PATH_TO_WET = '/path/'

# Define path to folder containing dry images
PATH_TO_DRY = '/path/'

# Read csv-file to pandas dataframe
df = pd.read_csv(PATH_TO_CSV, sep=';')

# Print some info
print(df.head())
print(df.info())

# Split the dataframe into test, validation and train data
test = df.sample(n=TEST_SIZE, random_state=42)
data_without_test = df.drop(test.index)
val = data_without_test.sample(n=VAL_SIZE, random_state=42)
train = data_without_test.drop(val.index)

# Create empty lists for images and labels
train_wet_imgs = []
train_dry_imgs = []

test_wet_imgs = []
test_dry_imgs = []

val_wet_imgs = []
val_dry_imgs = []

train_labels = []
test_labels = []
val_labels = []

# Loop dataframe rows
for index, row in df.iterrows():

    # Print progress
    if index % 10 == 0:
        print(index, '/', len(df))

    # Create correct path for the wet image, corresponding dry image
    wet_path = os.path.join(PATH_TO_WET, row['Filename_wet'])
    dry_path_pair = os.path.join(PATH_TO_DRY, row['Filename_dry'])

    # Read and process images
    wet_img = process_img(wet_path, IMG_SIZE)
    dry_img_pair = process_img(dry_path_pair, IMG_SIZE)

    # Next there are three if statements determining whether the image
    # belongs to train, validation or test set
    if row['Filename_wet'] in train['Filename_wet'].values:
        # Append four wet images to the list (one for pair and three for
        # not-pair images)
        train_wet_imgs.append(wet_img)
        train_wet_imgs.append(wet_img)
        train_wet_imgs.append(wet_img)
        train_wet_imgs.append(wet_img)

        # Append the pair dry image to list
        train_dry_imgs.append(dry_img_pair)

        # Get three random image paths from the correct dataset
        dry_path_random1 = os.path.join(
            PATH_TO_DRY,
            train.sample().iloc[0]['Filename_dry'])
        dry_path_random2 = os.path.join(
            PATH_TO_DRY,
            train.sample().iloc[0]['Filename_dry'])
        dry_path_random3 = os.path.join(
            PATH_TO_DRY,
            train.sample().iloc[0]['Filename_dry'])

        # Make sure it's not pair
        while dry_path_pair == dry_path_random1:
            dry_path_random1 = os.path.join(
                PATH_TO_DRY,
                train.sample().iloc[0]['Filename_dry'])
        while dry_path_pair == dry_path_random2:
            dry_path_random3 = os.path.join(
                PATH_TO_DRY,
                train.sample().iloc[0]['Filename_dry'])
        while dry_path_pair == dry_path_random3:
            dry_path_random3 = os.path.join(
                PATH_TO_DRY,
                train.sample().iloc[0]['Filename_dry'])

        # Process the image and append to list
        dry_img_random1 = process_img(dry_path_random1, IMG_SIZE)
        dry_img_random2 = process_img(dry_path_random2, IMG_SIZE)
        dry_img_random3 = process_img(dry_path_random3, IMG_SIZE)
        train_dry_imgs.append(dry_img_random1)
        train_dry_imgs.append(dry_img_random2)
        train_dry_imgs.append(dry_img_random3)

        # Append correct labels to lists
        train_labels.append([0, 1])
        train_labels.append([1, 0])
        train_labels.append([1, 0])
        train_labels.append([1, 0])

        continue

    if row['Filename_wet'] in val['Filename_wet'].values:
        # Append four wet images to the list (one for pair and three for
        # not-pair images)
        val_wet_imgs.append(wet_img)
        val_wet_imgs.append(wet_img)
        val_wet_imgs.append(wet_img)
        val_wet_imgs.append(wet_img)

        # Append the pair dry image to list
        val_dry_imgs.append(dry_img_pair)

        # Get random image path from the correct dataset
        dry_path_random1 = os.path.join(
            PATH_TO_DRY,
            val.sample().iloc[0]['Filename_dry'])
        dry_path_random2 = os.path.join(
            PATH_TO_DRY,
            val.sample().iloc[0]['Filename_dry'])
        dry_path_random3 = os.path.join(
            PATH_TO_DRY,
            val.sample().iloc[0]['Filename_dry'])

        # Make sure it's not pair
        while dry_path_pair == dry_path_random1:
            dry_path_random1 = os.path.join(
                PATH_TO_DRY,
                val.sample().iloc[0]['Filename_dry'])
        while dry_path_pair == dry_path_random2:
            dry_path_random2 = os.path.join(
                PATH_TO_DRY,
                val.sample().iloc[0]['Filename_dry'])
        while dry_path_pair == dry_path_random3:
            dry_path_random3 = os.path.join(
                PATH_TO_DRY,
                val.sample().iloc[0]['Filename_dry'])

        # Process the image and append to list
        dry_img_random1 = process_img(dry_path_random1, IMG_SIZE)
        dry_img_random2 = process_img(dry_path_random2, IMG_SIZE)
        dry_img_random3 = process_img(dry_path_random3, IMG_SIZE)
        val_dry_imgs.append(dry_img_random1)
        val_dry_imgs.append(dry_img_random2)
        val_dry_imgs.append(dry_img_random3)

        # Append correct labels to lists
        val_labels.append([0, 1])
        val_labels.append([1, 0])
        val_labels.append([1, 0])
        val_labels.append([1, 0])

        continue

    if row['Filename_wet'] in test['Filename_wet'].values:
        # Append wet image to the list
        test_wet_imgs.append(wet_img)

        # Append the pair dry image to list
        test_dry_imgs.append(dry_img_pair)

        # Append correct labels to lists
        test_labels.append([0, 1])

        continue

# let's double the sample size by cross-sampling:
train_input_A = train_wet_imgs + train_dry_imgs
train_input_B = train_dry_imgs + train_wet_imgs
val_input_A = val_wet_imgs + val_dry_imgs
val_input_B = val_dry_imgs + val_wet_imgs
test_input_A = test_wet_imgs
test_input_B = test_dry_imgs

train_labels += train_labels
# test_labels stay the same
val_labels += val_labels

# Save the inputs to Numpy arrays
np.save('train_input_A.npy',
        np.array(train_input_A))
np.save('train_input_B.npy',
        np.array(train_input_B))
np.save('test_input_A.npy',
        np.array(test_input_A))
np.save('test_input_B.npy',
        np.array(test_input_B))
np.save('val_input_A.npy',
        np.array(val_input_A))
np.save('val_input_B.npy',
        np.array(val_input_B))
np.save('train_labels.npy',
        np.array(train_labels))
np.save('test_labels.npy',
        np.array(test_labels))
np.save('val_labels.npy',
        np.array(val_labels))
