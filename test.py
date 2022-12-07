

import os
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


batch_size = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
data_path = './data/'
train_data_path = os.path.join(data_path, 'train/')
val_data_path = os.path.join(data_path, 'validation/')
test_data_path = os.path.join(data_path, 'test/')


test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode='rgb')

filenames = test_generator.filenames
nb_samples = len(filenames)
models = ['Modified_EfficientNet.h5','Modified_Inception.h5','Modified_InceptionResnet.h5','Modified_Mobilenet.h5',
          'Modified_VGG19.h5','Modified_Xception.h5']
for name in models:
    model = keras.models.load_model("./saved-models/"+name, custom_objects={"specificity": specificity,"f_measure":f_measure,"f1_score":f1_score})
    print(name)
    results = model.evaluate(test_generator,steps = nb_samples)
    print(model.metrics_names)
    print(results)