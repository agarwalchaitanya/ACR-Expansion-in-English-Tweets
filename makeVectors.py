import pickle 
import gensim
import os
import sys
import json
from acrlist import acr
import numpy as np
infile = open("doc2vec.pickle", "rb")
model = pickle.load(infile) 
infile.close()

for acronym, expansions in acr.items():
  Train_test_data = [[],[],[],[]]
  for expansion in expansions:
    with open("train_data/"+str(expansion)+'.txt') as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        Train_test_data[0].append(model.infer_vector(tweet.split()))
        Train_test_data[1].append(expansions.index(expansion))
    with open("test_data/"+str(expansion)+".txt") as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        Train_test_data[2].append(model.infer_vector(tweet.split()))
        Train_test_data[3].append(expansions.index(expansion))
  np.save("inferred_vector/"+str(acronym),Train_test_data)
  print("Done with test and traing date of: "+str(acronym))
  