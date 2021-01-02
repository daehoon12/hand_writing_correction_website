# EMNIST Data set : hand written alphabet

# filename : emnist_cnn.py
# history
# =============================
# 20200826 Ver 1.0 초안 및 주석 작성 (안지민)
# 20200828 Ver 1.1 Epohc 수정 (안지민)
# 20200102 Ver 1.2 Model 아키텍처, Hyperparameters 수정 (강대훈)
# =============================
# Ver 1.2

# Visualization Dependencies
from IPython.display import Image, SVG
import seaborn as sns

# Filepaths, Numpy, Tensorflow
import os
import numpy as np
import tensorflow as tf

# Keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

sns.set()


# pip install emnist
# Import Dataset(s)
from emnist import list_datasets
list_datasets()

from emnist import extract_test_samples
from emnist import extract_training_samples
X_train, Y_train = extract_training_samples('letters')
X_test, Y_test = extract_test_samples('letters')

# Flatten Data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)

# Rescale to 0 -> 1 by dividing by max pixel value (255)
# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255

# Create an ImageDataGenerator and do Image Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

# One-Hot Encoding

num_classes = 27
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Callback
class Callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('accuracy')>0.98):
        print("\n 98% acc reached")
        self.model.stop_training = True

callbacks = Callbacks()
# Empty Sequential model
from tensorflow.keras.models import Sequential
model = Sequential()

#Layers

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding ='same', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding ='same',))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(27, activation='softmax'))
# # 1 - number of elements (pixels) in each image
# # Dense layer - when every node from previous layer is connected to each node in current layer
# model.add(Dense(500, activation='relu'))
#
# # Second Hidden Layer
# model.add(Dense(500, activation='relu'))
#
# # Output Layer - number of nodes corresponds to number of y labels
# model.add(Dense(num_classes, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

# Train the Model
train_generator = train_datagen.flow(X_train, Y_train)
validation_generator = validation_datagen.flow(X_test, Y_test)

history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks = [callbacks])
# Train Model
model.summary()

# Save Model
model.save("emnist_trained.h5")

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
