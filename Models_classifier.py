import pickle 
import gensim
import os
import sys
import json
from acrlist import acr
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model


infile = open("doc2vec.pickle", "rb")
model = pickle.load(infile) 
infile.close()


def classifier(X_train, Y_train, X_test, Y_test, classes):
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
    return tf_model

for acronym, expansions in acr.items():
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for expansion in expansions:
        with open("train_data/"+str(expansion)+".txt") as file:
            tweets = file.readlines()
            for tweet in tweets:
                tweet = json.loads(tweet)
                tweet = tweet.lower()
                tweet = tweet.replace(expansion, acronym)
                X_train.append(model.infer_vector(tweet.split()))
                Y_train.append(expansions.index(expansion))
        with open("test_data/"+str(expansion)+".txt") as file:
            tweets = file.readlines()
            for tweet in tweets:
                tweet = json.loads(tweet)
                tweet = tweet.lower()
                tweet = tweet.replace(expansion, acronym)
                X_test.append(model.infer_vector(tweet.split()))
                Y_test.append(expansions.index(expansion))
        print(X_train[0])
    classifier(X_train, Y_train, X_test, Y_test, len(expansions)).save("classifiers/"+str(acronym)+".h5")
    print("Done with classifier model for " + str(acronym))
    print()

