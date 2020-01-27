import os
import pickle
import time
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import hickle


# create tensorboard for statistics
from preprocessing import CATEGORIES

NAME = f"cat_dog_detection_cnn_{int(time.time())}"
tensorboard = TensorBoard(log_dir=os.path.join("logs", NAME))

# load preprocessed data
X = hickle.load("pickles/X.hkl")
Y = hickle.load("pickles/Y.hkl")

# Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation="softmax"))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
earlystop = EarlyStopping(patience=10)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="rmsprop", metrics=["accuracy"])
model.summary()


# run model with gpu
model.fit(X, Y, batch_size=64, epochs=14, shuffle=True, callbacks=[tensorboard, earlystop, learning_rate_reduction], validation_split=0.1)
model.save(os.path.join("model", "main.h5"))

# tensorboard --logdir=logs/