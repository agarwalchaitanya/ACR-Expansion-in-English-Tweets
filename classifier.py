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


infile = open("doc2vec.py", "rb")
model = pickle.load(infile) 
infile.close()


def classifier(X_train, Y_train, X_test, Y_test):
  #takes Doc2Vec as input layer instead of Word Embeddings, and trains classifiers for each acronym
  tf_model = Sequential()
  tf_model.add(InputLayer(input_shape=[100]))
  tf_model.add(LSTM(200, dropout_U = 0.2, dropout_W = 0.2))
  tf_model.add(Dense(len(set(Y_train)), activation="softmax"))
  tf_model.compile(loss="catrgorical_crossentropy", optimizer="adam", metrics="['accuracy]")
  tf_model.fit(X_train, Y_train, batch_size=32, nb_epoch=100, verbose=1)
  score, acc = tf_model.evaluate(X_test, Y_test, verbose=1, batch_size=32)
  print("Score: %.2f" % (score))
  print("Validation Accuracy: %.2f" % (acc))
  return tf_model

for acronym, expansions in acr.items():
  train_data = []
  test_data = []
  for expansion in expansions:
    with open("train_data/"+str(expansion)) as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        train_data.append([model.infer_vector(tweet.split()), expansion])
    with open("test_data/"+str(expansion)+".txt") as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        test_data.append([model.infer_vector(tweet.split()), expansion])
  classifier(train_data[0], train_data[1], test_data[0], test_data[1]).save(str(acronym)+".h5")