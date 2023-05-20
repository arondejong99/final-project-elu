# Import the necessary modules
# If you get a ModuleNotFoundError, use %pip install {module} to install the module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import pydicom as dicom
from pathlib import Path
import time
from zipfile import ZipFile 
import cv2
from tqdm import tqdm
import glob
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization, Flatten, Dropout, Dense, Convolution1D, MaxPool1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, recall_score
import visualkeras
import xmltodict
import visualkeras

# Create function that shows the distribution between a set of columns
def distributionPlot(x, colors, labels, title):
    x = x.value_counts(normalize = True)

    plt.pie(x = x, colors = colors, labels = x.index, autopct = "%0.3f%%", startangle = 90)
    plt.title(title)
    plt.show()

# Create function that shows dicom image based on input path
def displayDicomImage(basepath, path, title, ax):
    # enter DICOM image name for pattern
    # result is a list of 1 element
    filename = dicom.data.data_manager.get_files(basepath, path)[0]

    img = dicom.dcmread(filename)

    ax.imshow(img.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
    ax.set_title(title)

def displayPngImage(path, title, ax):
    img = mpimg.imread(path)
    ax.imshow(X = img , cmap=plt.cm.bone)
    ax.set_title(title)

# Create function to create binary model
def create_binary_model(modelName, inputShape, filters):
    # Create the layers

    # Input layer
    input_layer = Input(shape=inputShape)

    # Convolutional and normalization layers
    conv1_layer = Convolution1D(filters[0], (5), padding='same', kernel_regularizer=l2(0.001), activation=relu)(input_layer)
    maxpool1_layer = MaxPool1D(pool_size=(2))(conv1_layer)

    # Flatten and dropout layer
    flat1_layer = Flatten()(maxpool1_layer)
    drop1_layer = Dropout(0.5)(flat1_layer)

    # Final Dense layer
    dense_layer = Dense(512, activation = relu)(drop1_layer)

    # Output layers
    pred_layer_bin = Dense(1, activation=sigmoid, name = "illness_output")(dense_layer)
    

    model = Model(inputs=input_layer, outputs=pred_layer_bin, name = modelName)
    return model

def create_categorical_model(modelName, inputShape, filters): 
    # Create the layers

    # Input layer
    input_layer = Input(shape=inputShape)

    # Convolutional and normalization layers
    conv1_layer = Convolution1D(filters[0], (5), padding='same', kernel_regularizer=l2(0.001), activation=relu)(input_layer)
    maxpool1_layer = MaxPool1D(pool_size=(2))(conv1_layer)
    norm1_layer = BatchNormalization(renorm = True)(maxpool1_layer)

    # Flatten and dropout layer
    flat1_layer = Flatten()(norm1_layer)
    drop1_layer = Dropout(0.25)(flat1_layer)

    # Final Dense layer
    dense_layer = Dense(512, activation = relu)(drop1_layer)

    # Output layer
    pred_layer_cat = Dense(4, activation='softmax', name = "cancer_type_output")(dense_layer)

    model = Model(inputs=input_layer, outputs=pred_layer_cat, name = modelName)
    return model

def create_regression_model(modelName, inputShape, filters): 
    # Create the layers

    # Input layer
    input_layer = Input(shape=inputShape)

    # Convolutional and normalization layers
    conv1_layer = Convolution1D(filters[0], (5), padding='same', kernel_regularizer=l2(0.001), activation='relu')(input_layer)
    conv2_layer = Convolution1D(filters[1], (3), padding='same', kernel_regularizer=l2(0.001), activation=relu)(conv1_layer)
    maxpool1_layer = MaxPool1D(pool_size=(2))(conv2_layer)
    
    conv3_layer = Convolution1D(filters[2], (5), padding='same', kernel_regularizer=l2(0.001), activation=relu)(maxpool1_layer)
    conv4_layer = Convolution1D(filters[3], (3), padding='same', kernel_regularizer=l2(0.001), activation=relu)(conv3_layer)
    #maxpool2_layer = MaxPool1D(pool_size=(2))(conv2_layer)
    #norm1_layer = BatchNormalization(renorm = True)(maxpool1_layer)

    flat1_layer = Flatten()(conv4_layer)
    drop1_layer = Dropout(0.5)(flat1_layer)

    # Final Dense layer
    dense_layer = Dense(256, activation = relu)(drop1_layer)
    dense_layer2 = Dense(128, activation = relu)(dense_layer)
    dense_layer3 = Dense(64, activation = relu)(dense_layer2)

    # Output layer
    pred_layer_reg = Dense(1, activation="linear", name = "tumor_size_output")(dense_layer3)

    model = Model(inputs=input_layer, outputs=pred_layer_reg, name = modelName)
    return model

# Trains a model
def train_model(model, metrics, loss_function, optimizer, monitor,
                train_ds, validation_ds, epochs, batch_size,
                steps_per_epcoh, patience):
    ###
#  This function compiles and trains the model
# ###
    model.compile(optimizer=optimizer, loss = loss_function, metrics = metrics)

    # Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience = patience, monitor = monitor)

    # Train the model
    hist = model.fit(train_ds.batch(batch_size=batch_size), validation_data = validation_ds.batch(batch_size=batch_size), 
                     epochs = epochs, steps_per_epoch=steps_per_epcoh,
                     callbacks = [early_stopping_monitor])

    return(hist)

# Create a function that displays the Accuracy of the training and validation data
def plot_accuracy(acc, val_acc, title, ylabel):
    x = np.arange(1, len(acc) + 1)
    x_ticks = np.arange(1, len(acc) + 1, 2)

    # summarize history for accuracy
    plt.plot(x, acc)
    plt.plot(x, val_acc)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.xticks(x_ticks)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Create a function that displays the correlation matrix
def plot_confusion_matrix(preds, true_values, labels, title):
    
    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_values, preds)

    # Display the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot = True, cbar = False, 
                fmt = 'd', cmap = 'Blues', ax = ax, xticklabels = labels,
                yticklabels = labels)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Actual values")
    ax.set_title(title)
    plt.show()

print("Functions import succesfull")