
model = tf.keras.Sequential([
    Model_Xcep,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.summary()

time_callback = TimeHistory()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("./checkpoint/Modified_XCept.h5", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    time_callback
]

Epoch 1/150
2022-12-06 19:19:56.395818: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2022-12-06 19:19:58.166073: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
95/95 [==============================] - 37s 332ms/step - loss: 0.4450 - precision: 0.6404 - recall: 0.9059 - accuracy: 0.8471 - true_negatives: 18077.0000 - true_positives: 10980.0000 - false_negatives: 1141.0000 - false_positives: 6165.0000 - specificity: 0.6435 - f_measure: 0.7521 - f1_score: 0.7521 - val_loss: 0.3620 - val_precision: 0.7792 - val_recall: 0.9413 - val_accuracy: 0.9076 - val_true_negatives: 2626.0000 - val_true_positives: 1426.0000 - val_false_negatives: 89.0000 - val_false_positives: 404.0000 - val_specificity: 0.6461 - val_f_measure: 0.7659 - val_f1_score: 0.8526 - lr: 0.0010
Epoch 2/150
95/95 [==============================] - 30s 319ms/step - loss: 0.2447 - precision: 0.7171 - recall: 0.9668 - accuracy: 0.9121 - true_negatives: 19620.0000 - true_positives: 11718.0000 - false_negatives: 403.0000 - false_positives: 4622.0000 - specificity: 0.7001 - f_measure: 0.8119 - f1_score: 0.8236 - val_loss: 0.2541 - val_precision: 0.6815 - val_recall: 0.9888 - val_accuracy: 0.9182 - val_true_negatives: 2330.0000 - val_true_positives: 1498.0000 - val_false_negatives: 17.0000 - val_false_positives: 700.0000 - val_specificity: 0.6453 - val_f_measure: 0.7809 - val_f1_score: 0.8074 - lr: 0.0010
Epoch 3/150
95/95 [==============================] - 30s 315ms/step - loss: 0.2020 - precision: 0.7340 - recall: 0.9801 - accuracy: 0.9263 - true_negatives: 19937.0000 - true_positives: 11880.0000 - false_negatives: 241.0000 - false_positives: 4305.0000 - specificity: 0.7290 - f_measure: 0.8360 - f1_score: 0.8397 - val_loss: 0.1995 - val_precision: 0.6507 - val_recall: 0.9934 - val_accuracy: 0.9287 - val_true_negatives: 2222.0000 - val_true_positives: 1505.0000 - val_false_negatives: 10.0000 - val_false_positives: 808.0000 - val_specificity: 0.6541 - val_f_measure: 0.7887 - val_f1_score: 0.7865 - lr: 0.0010
Epoch 4/150
95/95 [==============================] - 32s 331ms/step - loss: 0.1727 - precision: 0.7370 - recall: 0.9856 - accuracy: 0.9357 - true_negatives: 19979.0000 - true_positives: 11947.0000 - false_negatives: 174.0000 - false_positives: 4263.0000 - specificity: 0.7464 - f_measure: 0.8495 - f1_score: 0.8439 - val_loss: 0.1919 - val_precision: 0.7990 - val_recall: 0.9710 - val_accuracy: 0.9287 - val_true_negatives: 2660.0000 - val_true_positives: 1471.0000 - val_false_negatives: 44.0000 - val_false_positives: 370.0000 - val_specificity: 0.7863 - val_f_measure: 0.8689 - val_f1_score: 0.8767 - lr: 0.0010
Epoch 5/150
95/95 [==============================] - 31s 324ms/step - loss: 0.1638 - precision: 0.7394 - recall: 0.9873 - accuracy: 0.9380 - true_negatives: 20025.0000 - true_positives: 11967.0000 - false_negatives: 154.0000 - false_positives: 4217.0000 - specificity: 0.7548 - f_measure: 0.8555 - f1_score: 0.8458 - val_loss: 0.1811 - val_precision: 0.8069 - val_recall: 0.9683 - val_accuracy: 0.9320 - val_true_negatives: 2679.0000 - val_true_positives: 1467.0000 - val_false_negatives: 48.0000 - val_false_positives: 351.0000 - val_specificity: 0.8242 - val_f_measure: 0.8904 - val_f1_score: 0.8802 - lr: 0.0010
Epoch 6/150
95/95 [==============================] - 32s 333ms/step - loss: 0.1504 - precision: 0.7344 - recall: 0.9879 - accuracy: 0.9460 - true_negatives: 19911.0000 - true_positives: 11974.0000 - false_negatives: 147.0000 - false_positives: 4331.0000 - specificity: 0.7558 - f_measure: 0.8563 - f1_score: 0.8429 - val_loss: 0.1721 - val_precision: 0.7292 - val_recall: 0.9901 - val_accuracy: 0.9373 - val_true_negatives: 2473.0000 - val_true_positives: 1500.0000 - val_false_negatives: 15.0000 - val_false_positives: 557.0000 - val_specificity: 0.7488 - val_f_measure: 0.8526 - val_f1_score: 0.8400 - lr: 0.0010
Epoch 7/150
95/95 [==============================] - 30s 319ms/step - loss: 0.1367 - precision: 0.7271 - recall: 0.9907 - accuracy: 0.9475 - true_negatives: 19735.0000 - true_positives: 12008.0000 - false_negatives: 113.0000 - false_positives: 4507.0000 - specificity: 0.7581 - f_measure: 0.8588 - f1_score: 0.8392 - val_loss: 0.1878 - val_precision: 0.6758 - val_recall: 0.9908 - val_accuracy: 0.9294 - val_true_negatives: 2310.0000 - val_true_positives: 1501.0000 - val_false_negatives: 14.0000 - val_false_positives: 720.0000 - val_specificity: 0.7100 - val_f_measure: 0.8272 - val_f1_score: 0.8039 - lr: 0.0010
Epoch 8/150
95/95 [==============================] - ETA: 0s - loss: 0.1296 - precision: 0.7351 - recall: 0.9922 - accuracy: 0.9516 - true_negatives: 19909.0000 - true_positives: 12027.0000 - false_negatives: 94.0000 - false_positives: 4333.0000 - specificity: 0.7656 - f_measure: 0.8643 - f1_score: 0.8449
Epoch 8: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
95/95 [==============================] - 31s 325ms/step - loss: 0.1296 - precision: 0.7351 - recall: 0.9922 - accuracy: 0.9516 - true_negatives: 19909.0000 - true_positives: 12027.0000 - false_negatives: 94.0000 - false_positives: 4333.0000 - specificity: 0.7656 - f_measure: 0.8643 - f1_score: 0.8449 - val_loss: 0.2002 - val_precision: 0.7910 - val_recall: 0.9617 - val_accuracy: 0.9254 - val_true_negatives: 2645.0000 - val_true_positives: 1457.0000 - val_false_negatives: 58.0000 - val_false_positives: 385.0000 - val_specificity: 0.8120 - val_f_measure: 0.8805 - val_f1_score: 0.8681 - lr: 0.0010
Epoch 9/150
95/95 [==============================] - 30s 312ms/step - loss: 0.1064 - precision: 0.7378 - recall: 0.9947 - accuracy: 0.9616 - true_negatives: 19957.0000 - true_positives: 12057.0000 - false_negatives: 64.0000 - false_positives: 4285.0000 - specificity: 0.7722 - f_measure: 0.8694 - f1_score: 0.8474 - val_loss: 0.1875 - val_precision: 0.6691 - val_recall: 0.9861 - val_accuracy: 0.9333 - val_true_negatives: 2291.0000 - val_true_positives: 1494.0000 - val_false_negatives: 21.0000 - val_false_positives: 739.0000 - val_specificity: 0.7394 - val_f_measure: 0.8451 - val_f1_score: 0.7975 - lr: 5.0000e-04
Epoch 10/150
95/95 [==============================] - ETA: 0s - loss: 0.0960 - precision: 0.7545 - recall: 0.9950 - accuracy: 0.9677 - true_negatives: 20318.0000 - true_positives: 12061.0000 - false_negatives: 60.0000 - false_positives: 3924.0000 - specificity: 0.7848 - f_measure: 0.8774 - f1_score: 0.8588
Epoch 10: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
95/95 [==============================] - 30s 311ms/step - loss: 0.0960 - precision: 0.7545 - recall: 0.9950 - accuracy: 0.9677 - true_negatives: 20318.0000 - true_positives: 12061.0000 - false_negatives: 60.0000 - false_positives: 3924.0000 - specificity: 0.7848 - f_measure: 0.8774 - f1_score: 0.8588 - val_loss: 0.1739 - val_precision: 0.7371 - val_recall: 0.9881 - val_accuracy: 0.9406 - val_true_negatives: 2496.0000 - val_true_positives: 1497.0000 - val_false_negatives: 18.0000 - val_false_positives: 534.0000 - val_specificity: 0.7659 - val_f_measure: 0.8628 - val_f1_score: 0.8447 - lr: 5.0000e-04
Epoch 11/150
95/95 [==============================] - 30s 317ms/step - loss: 0.0850 - precision: 0.7662 - recall: 0.9955 - accuracy: 0.9697 - true_negatives: 20561.0000 - true_positives: 12066.0000 - false_negatives: 55.0000 - false_positives: 3681.0000 - specificity: 0.7940 - f_measure: 0.8834 - f1_score: 0.8661 - val_loss: 0.1489 - val_precision: 0.7949 - val_recall: 0.9848 - val_accuracy: 0.9472 - val_true_negatives: 2645.0000 - val_true_positives: 1492.0000 - val_false_negatives: 23.0000 - val_false_positives: 385.0000 - val_specificity: 0.8189 - val_f_measure: 0.8942 - val_f1_score: 0.8799 - lr: 2.5000e-04
Epoch 12/150
95/95 [==============================] - 30s 313ms/step - loss: 0.0757 - precision: 0.7731 - recall: 0.9957 - accuracy: 0.9724 - true_negatives: 20699.0000 - true_positives: 12069.0000 - false_negatives: 52.0000 - false_positives: 3543.0000 - specificity: 0.7994 - f_measure: 0.8868 - f1_score: 0.8706 - val_loss: 0.1778 - val_precision: 0.6890 - val_recall: 0.9914 - val_accuracy: 0.9333 - val_true_negatives: 2352.0000 - val_true_positives: 1502.0000 - val_false_negatives: 13.0000 - val_false_positives: 678.0000 - val_specificity: 0.7472 - val_f_measure: 0.8521 - val_f1_score: 0.8140 - lr: 2.5000e-04
Epoch 13/150
95/95 [==============================] - ETA: 0s - loss: 0.0691 - precision: 0.7765 - recall: 0.9968 - accuracy: 0.9784 - true_negatives: 20764.0000 - true_positives: 12082.0000 - false_negatives: 39.0000 - false_positives: 3478.0000 - specificity: 0.8037 - f_measure: 0.8898 - f1_score: 0.8733
Epoch 13: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
95/95 [==============================] - 30s 317ms/step - loss: 0.0691 - precision: 0.7765 - recall: 0.9968 - accuracy: 0.9784 - true_negatives: 20764.0000 - true_positives: 12082.0000 - false_negatives: 39.0000 - false_positives: 3478.0000 - specificity: 0.8037 - f_measure: 0.8898 - f1_score: 0.8733 - val_loss: 0.1573 - val_precision: 0.7841 - val_recall: 0.9855 - val_accuracy: 0.9452 - val_true_negatives: 2619.0000 - val_true_positives: 1493.0000 - val_false_negatives: 22.0000 - val_false_positives: 411.0000 - val_specificity: 0.8123 - val_f_measure: 0.8905 - val_f1_score: 0.8734 - lr: 2.5000e-04
Epoch 14/150
95/95 [==============================] - 31s 327ms/step - loss: 0.0632 - precision: 0.7827 - recall: 0.9974 - accuracy: 0.9790 - true_negatives: 20885.0000 - true_positives: 12090.0000 - false_negatives: 31.0000 - false_positives: 3357.0000 - specificity: 0.8080 - f_measure: 0.8927 - f1_score: 0.8774 - val_loss: 0.1526 - val_precision: 0.7692 - val_recall: 0.9875 - val_accuracy: 0.9406 - val_true_negatives: 2581.0000 - val_true_positives: 1496.0000 - val_false_negatives: 19.0000 - val_false_positives: 449.0000 - val_specificity: 0.8050 - val_f_measure: 0.8869 - val_f1_score: 0.8650 - lr: 1.2500e-04
Epoch 15/150
95/95 [==============================] - 32s 339ms/step - loss: 0.0601 - precision: 0.7834 - recall: 0.9974 - accuracy: 0.9807 - true_negatives: 20899.0000 - true_positives: 12089.0000 - false_negatives: 32.0000 - false_positives: 3343.0000 - specificity: 0.8088 - f_measure: 0.8932 - f1_score: 0.8778 - val_loss: 0.1487 - val_precision: 0.7769 - val_recall: 0.9881 - val_accuracy: 0.9452 - val_true_negatives: 2600.0000 - val_true_positives: 1497.0000 - val_false_negatives: 18.0000 - val_false_positives: 430.0000 - val_specificity: 0.8086 - val_f_measure: 0.8893 - val_f1_score: 0.8699 - lr: 1.2500e-04
Epoch 15: early stopping
[36.797059059143066, 30.366305828094482, 30.0098237991333, 31.591169357299805, 30.877872943878174, 31.700512886047363, 30.369661569595337, 30.96563220024109, 29.70181369781494, 29.68198251724243, 30.229843378067017, 29.88531494140625, 30.245854139328003, 31.137711763381958, 32.33769774436951]
