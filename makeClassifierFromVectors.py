import numpy as np
from acrlist import acr
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model


def classifier(X_train, Y_train, X_test, Y_test, classes, acronym):
    xx = tf.keras.utils.normalize(X_train)
    xtest = tf.keras.utils.normalize(X_test)
    tf_model = Sequential()
    tf_model.add(Dense(64, input_shape=(100,),activation=tf.nn.relu))
    tf_model.add(Dropout(0.3))
    tf_model.add(Dense(128, activation=tf.nn.relu))
    tf_model.add(Dropout(0.3))
    tf_model.add(Dense(128, activation=tf.nn.relu))
    tf_model.add(Dropout(0.3))
    tf_model.add(Dense(64, activation=tf.nn.relu))
    tf_model.add(Dense(classes, activation=tf.nn.softmax))
    tf_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    tf_model.fit(np.array(xx), np.array(Y_train), epochs=50)
    val_loss, val_acc = tf_model.evaluate(np.array(xtest), np.array(Y_test))
    print(val_loss, val_acc)
    tf_model.save("classifiers/"+str(acronym)+".h5")

for acronym, expansions in acr.items():
    Test_train_data = np.load("inferred_vector/"+str(acronym)+".npy", allow_pickle=True)
    print(np.shape(Test_train_data))
    classifier(Test_train_data[2], Test_train_data[3], Test_train_data[0], Test_train_data[1], len(expansions), acronym)
    print("Done with classifier model for " + str(acronym))
    print()

