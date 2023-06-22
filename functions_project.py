# Import the necessary modules
# If you get a ModuleNotFoundError, use %pip install {module} to install the module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import pydicom as dicom
import time
import xmltodict
import glob
import cv2
from tqdm import tqdm
from zipfile import ZipFile 
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization, Flatten, Dropout, Dense, Convolution1D, MaxPool1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve, roc_auc_score, RocCurveDisplay, explained_variance_score

# Get the normal lung dataset from the zip file
def getNormalLungsData(normal_lungs_path):
    # open the zip file in read mode
    with ZipFile(normal_lungs_path, 'r') as normal_lungs_zip: 
        # Extract all the files
        normal_lungs_zip.extractall()

    # Extract the PNG files from the source folder final-project-elu\normal_valid
    NormalLungFiles = glob.glob('*/*.png', recursive=True)

    normalLungsImages = []

    # Read each PNG file in a for loop and resize them form 
    for pathNormalLung in tqdm(NormalLungFiles):
        lungImage = cv2.imread(pathNormalLung, cv2.IMREAD_GRAYSCALE)
        lungImageResized = cv2.resize(lungImage, dsize = (512,512))
        normalLungsImages.append(lungImageResized)

    return(normalLungsImages)

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

    flat1_layer = Flatten()(conv4_layer)
    drop1_layer = Dropout(0.5)(flat1_layer)

    # Final Dense layer
    dense_layer = Dense(256, activation = relu)(drop1_layer)
    dense_layer2 = Dense(128, activation = relu)(dense_layer)
    dense_layer3 = Dense(64, activation = relu)(dense_layer2)

    # Output layer
    pred_layer_reg = Dense(1, activation="linear", name = "tumor_size_output")(dense_layer3)

    model = Model(inputs=input_layer, outputs=pred_layer_reg, name = modelName)
    return(model)

# Trains a model
def train_model(model, metrics, loss_function, optimizer, monitor,
                train_ds, validation_ds, epochs, batch_size,
                steps_per_epcoh, patience):
    ###
#  This function compiles and trains the model
# ###
    # Set start time
    start_time = time.time()
    model.compile(optimizer=optimizer, loss = loss_function, metrics = metrics)

    # Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience = patience, monitor = monitor)

    # Train the model
    hist = model.fit(train_ds.batch(batch_size=batch_size), validation_data = validation_ds.batch(batch_size=batch_size), 
                     epochs = epochs, steps_per_epoch=steps_per_epcoh,
                     callbacks = [early_stopping_monitor])
    
    # Print time execution
    print(f"Execution time: {(time.time() - start_time):.3f} seconds")
    return(hist)

# Create a function that displays the Accuracy of the training and validation data
def plot_accuracy(acc, val_acc, title, ylabel):
    x = np.arange(1, len(acc) + 1)
    x_ticks = np.arange(1, len(acc) + 1, 2)

    fig, ax = plt.subplots(figsize = (8, 8))

    # summarize history for accuracy
    ax.plot(x, acc)
    ax.plot(x, val_acc)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('epoch')
    ax.set_xticks(x_ticks)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Create a function that displays the correlation matrix
def plot_confusion_matrix(preds, true_values, labels, title):
    
    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_values, preds)

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize = (8,8))
    sns.heatmap(conf_matrix, annot = True, cbar = False, 
                fmt = 'd', cmap = 'Blues', ax = ax, xticklabels = labels,
                yticklabels = labels)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Actual values")
    ax.set_title(title)
    plt.show()

# A function that displays the ROC curve with the AUC score in it
def displayRocCurve(true, pred, estimator_name):
    fig, ax = plt.subplots(figsize = (8, 8))
    # Calculate the auc based on the fpr and tpr
    fpr, tpr, _ = roc_curve(true, pred)
    #print(fpr, tpr)
    auc_bin = roc_auc_score(true, pred)

    roc_display = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = auc_bin, 
                                  estimator_name = estimator_name)

    roc_display.plot(ax = ax)

def regression_results(y_true, y_pred):

    # Regression metrics to use
    explained_variance=explained_variance_score(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred) 
    mse=mean_squared_error(y_true, y_pred) 
   # mean_sq_log_error=mean_squared_log_error(y_true, y_pred)
    r2=r2_score(y_true, y_pred)

    # Print out the regression metrics
    print(f"Explained_variance score for Regression Neural Network: {explained_variance:.2f}")    
    #print(f"Mean Squared Log error for Regression Neural Network: {mean_sq_log_error:.4f}")
    print(f"R2 score for Regression Neural Network: {r2:.4f}")
    print(f"Mean absolute error score for Regression Neural Network: {mae:.4f}")
    print(f"Mean Squared error for Regression Neural Network: {mse:.4f}")
    print(f"Root Mean Squared error for Regression Neural Network: {np.sqrt(mse):.4f}")

def createAnnotationDf(filePath, cols):
    # Open the zip file
    with ZipFile(filePath, 'r') as zip_annotations:
        # Get all the names of the directories and filter them if they have an xml file
        listOfiles = zip_annotations.namelist()
        listPaths = [path for path in listOfiles if str(path).endswith(".xml")]

        # Set up list and append them for each xml file
        xmin, ymin, xmax, ymax, patientId, sopInstanceId = [],[],[],[], [], []
        for path in range(len(listPaths)):
            with zip_annotations.open(listPaths[path]) as fd:
                # Some xml files are invalid. Therefore we use a try and except method            
                try:
                    # Retrieve the data
                    path_file = Path(str(fd))
                    read = fd.read()
                    doc = xmltodict.parse(read)
                    patientId.append("Lung_Dx-" + path_file.parts[-2])
                    sopInstanceId.append(str(path_file.stem))
                    
                    # Appen the retrieved data to the lists set earlier
                    xmin.append(doc["annotation"]["object"]["bndbox"]["xmin"])
                    ymin.append(doc["annotation"]["object"]["bndbox"]["ymin"])
                    xmax.append(doc["annotation"]["object"]["bndbox"]["xmax"])
                    ymax.append(doc["annotation"]["object"]["bndbox"]["xmin"])
                    
                except:
                    continue

        # Create the dataframe for the annotation data
        annot_df = pd.DataFrame(zip(patientId, sopInstanceId, xmin, ymin, xmax, ymax), columns=cols)     

        return(annot_df)

print("Functions import succesfull")