import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization # type: ignore
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import joblib
from sklearn.utils.class_weight import compute_class_weight



def cnn():
    model = Sequential()
    model.add(InputLayer(input_shape=(128, 128, 1)))

    # 1st conv block
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    # 2nd conv block
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 3rd conv block
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Fully connected block
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model



def weighted_f1(y_true, y_pred):
    # Ensure consistent data types
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.round(y_pred)  # Round y_pred to binary predictions (0 or 1)

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred)  # True positives
    fp = tf.reduce_sum((1 - y_true) * y_pred)  # False positives
    fn = tf.reduce_sum(y_true * (1 - y_pred))  # False negatives

    # Avoid division by zero in precision and recall
    precision = tf.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = tf.where(tp + fn > 0, tp / (tp + fn), 0.0)

    # Compute F1 score
    f1 = tf.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0.0)

    return f1


if __name__ == '__main__':
    file_path = r"C:\Users\leeyo\OneDrive\Desktop\STA_221\Data\split_data_wo_gan.pkl"
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    X_train = X_train.reshape((-1, 128, 128, 1))
    X_test = X_test.reshape((-1, 128, 128, 1))
    # Debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    y_train = np.array(y_train)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = cnn()
    model.save(r"C:\Users\leeyo\OneDrive\Desktop\STA_221\Model\cnn.h5")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2,
          class_weight=class_weights, callbacks=[early_stopping])
    # Evaluate the model
    test_loss, test_f1 = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Test Loss: {test_loss}, Accuracy: {test_f1}")

    pred = model.predict(X_test)
    pred_binary = (pred > 0.5).astype(int).flatten()
    f1 = metrics.f1_score(pred_binary, y_test, average="weighted")
    print(f"F1 Score: {f1}")