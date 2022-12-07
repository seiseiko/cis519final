# -*- coding: utf-8 -*-
"""519 Final - Mobilenet
# **Machine Learning for COVID-19 Diagnosis**
---
Team: Lanqing Bao, Yuqi Zhang, Zeyuan Xu

# 1. Library, model and data setup
"""

# basic
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
# for google colab
#drive.mount('/gdrive')
# %cd /gdrive/MyDrive/519Final/

# for ml
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

"""###**Pre-trained model loading**
Utilized in prev paper
1.   ResNet50
2.   VGG19
3.   MobileNet
4.   Xception
5.   Inception
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

"""### **Data loading**

The dataset has the following directory structure:

<pre>
<b>data</b>
|__ <b>train</b>
    |______ <b>COVID</b>: [covid.0.jpg, covid.1.jpg, covid.2.jpg ....]
    |______ <b>NORMAL</b>: [normal.0.jpg, normal.1.jpg, normal.2.jpg ...]
    |______ <b>VIRAL</b>: [Viral Pneumonia-1.png, Viral Pneumonia-2.png, Viral Pneumonia-3.png ...]
|__ <b>validation</b>
    |______ <b>COVID</b>: [covid.0.jpg, covid.1.jpg, covid.2.jpg ....]
    |______ <b>NORMAL</b>: [normal.0.jpg, normal.1.jpg, normal.2.jpg ...]
    |______ <b>VIRAL</b>: [Viral Pneumonia-1.png, Viral Pneumonia-2.png, Viral 
|__ <b>test</b>
    |______ <b>COVID</b>: [covid.0.jpg, covid.1.jpg, covid.2.jpg ....]
    |______ <b>NORMAL</b>: [normal.0.jpg, normal.1.jpg, normal.2.jpg ...]
    |______ <b>VIRAL</b>: [Viral Pneumonia-1.png, Viral Pneumonia-2.png, Viral 
</pre>
"""
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def specificity(y_true, y_pred):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)

    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def f_measure(y_true, y_pred):

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    spec = specificity(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((spec * recall) / (spec + recall + K.epsilon()))


def f1_score(y_true, y_pred):

    def precision(y_true, y_pred):
        """Precision metric.
         Only computes a batch-wise average of precision.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * (p * r) / (p + r + K.epsilon())
    return f1_score

batch_size = 128
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
data_path = './data/'
train_data_path = os.path.join(data_path, 'train/')
val_data_path = os.path.join(data_path, 'validation/')
test_data_path = os.path.join(data_path, 'test/')

"""ImageDataGenerator usage reference
https://colab.research.google.com/github/tfindiamooc/tfindiamooc.github.io/blob/master/colabs/image_classification_and_visualization.ipynb#scrollTo=Giv0wMQzVrVw
"""

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode='rgb')

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    val_data_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode='rgb', )

"""# 2. Transfer learning

## MobileNetV2
"""

Model_MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',
                                                                  input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
# non-trainable
for layer in Model_MobileNetV2.layers:
    layer.trainable = False

Model_MobileNetV2.summary()

model = tf.keras.Sequential([
    Model_MobileNetV2,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.summary()

time_callback = TimeHistory()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("./checkpoint/Modified_Mobilenet.h5", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    time_callback
]
# Compiling the model, set up the metrics
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       'accuracy',
                       tf.keras.metrics.TrueNegatives(),
                       tf.keras.metrics.TruePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.FalsePositives(),
                       specificity,f_measure,f1_score])

# test if gpu is available
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# training
history = model.fit(train_generator,
                    validation_data=val_generator, epochs=150,
                    callbacks=[callbacks])

# save model and training history, epoch time
names = ['Xception','VGG16','Mobilenet']
current_model_name = names[2]
model.save("./saved-models/Modified_"+current_model_name+".h5")
with open('./saved-history/'+current_model_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
print('time:',time_callback.times)
