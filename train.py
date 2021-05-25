#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:47:00 2020

@author: Tuomas Jalonen

This file is used for building the Siamese network model.

"""

# Import packages
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Lambda, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

# This is the path to folder containing .npy-files created with
# Preprocess_data.py
DIR = '/path/'

def dot_product(vecs, normalize=False):
    """
    This function returns dot product of the input vectors.

    Parameters
    ----------
    vecs : Flattened output vectors of VGG16 feature extractor
    normalize : The default is False.

    Returns
    -------
    Dot product of the vectors

    """
    vec_x, vec_y = vecs

    # if normalize:
    #     vec_x = K.l2_normalize(vec_x, axis=0)
    #     vec_y = K.l2_normalize(vec_x, axis=0)

    return K.prod(K.stack([vec_x, vec_y], axis=1), axis=1)

# This defines the similarity model.
# The inputs are extracted features from VGG16 and outputs probabilities for
# Pair and Not-pair
def similarity_model(vector_a, vector_b):
    """
    This function defines the similarity model by taking outputs from feature
    extraction models as inputs, merges them with lambda layer, adds
    fully-connected layers and finally outputs probabilities for input images
    being pair and not being pair.

    Parameters
    ----------
    vector_A : Output of VGG16 with image A
    vector_B : Output of VGG16 with image B

    Returns
    -------
    pred : probabilities for Pair and Not-pair

    """

    merged = Lambda(dot_product,
                    output_shape=vector_a[0])([vector_a, vector_b])

    fc1 = Dense(512, kernel_initializer='he_normal')(merged)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Activation("relu")(fc1)

    fc2 = Dense(128, kernel_initializer='he_normal')(fc1)
    fc2 = Dropout(0.1)(fc2)
    fc2 = Activation("relu")(fc2)

    fc3 = Dense(10, kernel_initializer='he_normal')(fc2)
    fc3 = Activation("relu")(fc3)

    pred = Dense(2, kernel_initializer='normal')(fc2)
    pred = Activation("softmax")(pred)

    return pred

# Define feature extraction models for both inputs. Here we use VGG16 with
# pretrained weights
model_1 = VGG16(weights='imagenet', include_top=True)
model_2 = VGG16(weights='imagenet', include_top=True)

# Let's freeze the VGG16 layers and give them unique names
for layer in model_1.layers:
    layer.trainable = False
    layer._name = layer._name + "_1"
for layer in model_2.layers:
    layer.trainable = False
    layer._name = layer._name + "_2"

# Get outputs from Flatten layers
v1 = model_1.get_layer("flatten_1").output
v2 = model_2.get_layer("flatten_2").output

# Now we can compile the whole model
preds = similarity_model(v1, v2)
model = Model(inputs=[model_1.input, model_2.input], outputs=preds)

LR = 0.00001
opt = Adam(lr=LR)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Load data
print('Loading data')
train_input_A = np.load(f'{DIR}/train_input_A.npy')
train_input_B = np.load(f'{DIR}/train_input_B.npy')
val_input_A = np.load(f'{DIR}/val_input_A.npy')
val_input_B = np.load(f'{DIR}/val_input_B.npy')
train_labels = np.load(f'{DIR}/train_labels.npy')
val_labels = np.load(f'{DIR}/val_labels.npy')

# train the model
print("[INFO] training model...")

checkpoint = ModelCheckpoint(
    'model{epoch:04d}_{val_loss:.4f}.h5',
    monitor='val_loss',
    save_best_only=True)

BATCH_SIZE = 64
print('Batch size:', BATCH_SIZE)
print('Learning rate:', LR)
EPOCHS = 200

# Define base generator
generator = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode="nearest")

# This function takes base generator and defines flow method to feed the
# training process with image batches. Here the data is also augmented with
# flips.
def create_generator(generator, batch_size):
    """

    Parameters
    ----------
    generator : base generator
    batch_size : Intended batch size

    Yields
    ------
    Image batches: batch_size * (image_A, image_B, label)

    """
    gen_x1 = generator.flow(x=train_input_A,
                            y=train_labels,
                            batch_size=batch_size,
                            shuffle=False,
                            seed=7)

    gen_x2 = generator.flow(x=train_input_B,
                            y=train_labels,
                            batch_size=batch_size,
                            shuffle=False,
                            seed=7)
    while True:
        x_1 = gen_x1.next()
        x_2 = gen_x2.next()
        yield [x_1[0], x_2[0]], x_2[1]

# Define the multiple data generator
datagen = create_generator(generator, BATCH_SIZE)

# Train the model
history = model.fit(datagen,
                    validation_data=([val_input_A, val_input_B], val_labels),
                    epochs=EPOCHS, callbacks=[checkpoint], shuffle=True,
                    steps_per_epoch=len(train_input_A)/BATCH_SIZE,
                    use_multiprocessing=True)

# Plot accuracies
plt.plot(history.history['accuracy'], label='Accuracy (training)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.savefig('history_accuracies.png')
plt.clf()

# Plot losses
plt.plot(history.history['loss'], label='Loss (training)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.savefig('history_losses.png')

# Save history file
np.save('my_history.npy', history.history)
