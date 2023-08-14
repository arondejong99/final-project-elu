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

# Functions that import and reads the normal lung dataset from the zip file
def get_normal_lungs_data(normal_lungs_path):
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

def create_annotation_df(filePath, cols):
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

# Create function that shows the distribution between a set of columns
def distributionPlot(x, colors, labels, title):
    x = x.value_counts(normalize = True)

    plt.pie(x = x, colors = colors, labels = x.index, autopct = "%0.3f%%", startangle = 90)
    plt.title(title)
    plt.show()

# Create function that shows dicom image based on input path
def display_dicom_image(basepath, path, title, ax):
    # enter DICOM image name for pattern
    # result is a list of 1 element
    filename = dicom.data.data_manager.get_files(basepath, path)[0]

    img = dicom.dcmread(filename)

    # set the color map to bone
    ax.imshow(img.pixel_array, cmap=plt.cm.bone)  
    ax.set_title(title)

# Create a function that displays an PNG image
def display_png_image(path, title, ax):
    # Read the image
    img = mpimg.imread(path)

    # Dispaly the image
    ax.imshow(X = img , cmap=plt.cm.bone)
    ax.set_title(title)

print("Functions import succesfull")