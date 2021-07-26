###################################
## import the necessary packages ##
###################################
import argparse

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from Sudokunet import SudokuNet


###########################################################
## Construct the argument parser and parse the arguments ##
###########################################################
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

#########################################################################################
## Initialize the initial learning rate, number of epochs to train for, and batch size ##
## and data parameters																   ##
#########################################################################################
INIT_LR = 1e-3
EPOCHS = 10
BS = 128
CLASSES = 10 
INPUT_SHAPE = (28, 28, 1)


#################################################################
# Grab the MNIST dataset and split between train and test sets ##
#################################################################
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
assert (INPUT_SHAPE == (28, 28, 1))

# Add a channel (grayscale) dimension to the digits
x_train = x_train.reshape((x_train.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
x_test  = x_test.reshape ((x_test.shape[0] , INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))

# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train_bin = keras.utils.to_categorical(y_train, CLASSES)
y_test_bin = keras.utils.to_categorical(y_test, CLASSES)
classes_ = np.unique(np.union1d(y_train, y_test))
print(classes_)

# from sklearn.preprocessing import LabelBinarizer
# le = LabelBinarizer()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)
# print(le.classes_)

########################################
## Initialize the optimizer and model ##
########################################
print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width = INPUT_SHAPE[0], height = INPUT_SHAPE[1], nb_channel = INPUT_SHAPE[2], nb_classes=CLASSES)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
H = model.fit(
	x_train, y_train_bin,
	validation_data = (x_test, y_test_bin),
	batch_size = BS,
	epochs = EPOCHS,
	verbose = 1)

# Evaluate the network
print("[INFO] evaluating network...")

score = model.evaluate(x_test, y_test_bin, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


predictions = model.predict(x_test)
print(classification_report(y_test_bin.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in classes_]))

################################
# Serialize the model to disk ##
################################
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")