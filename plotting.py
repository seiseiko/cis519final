
"""#Result"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

current_model_name = 'Xception'
history = np.load('./saved-history/Modified_Xception.npy',allow_pickle=True)
fig_path = os.path.join('./fig',current_model_name+'/')
with open('/saved-history/'+current_model_name, "rb") as file_pi:
    history = pickle.load(file_pi)

if not os.path.exists(fig_path):
    os.makedirs(fig_path)
print(history)
# Accuracy & Loss
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig(fig_path+'accuracy.png')

# loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss - '+current_model_name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig(fig_path+'loss.png')

# Precision, Recall, Macro F1 score
val_recall = history['val_recall']
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history['val_precision']
avg_precision = np.mean(val_precision)
avg_precision

Train_accuracy = history['accuracy']

epochs = range(1, len(Train_accuracy) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0, 1)
plt.show()
plt.savefig(fig_path+'PercisionRecall.png')