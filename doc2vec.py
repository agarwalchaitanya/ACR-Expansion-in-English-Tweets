import numpy as numpy
import sys
from tqdm import tqdm
import os
import pandas as pd 
import string
import gensim
import json
import pickle
from acrlist import acr

train_data_tweets = []
train_data_acronyms = []
print("processing data!")
for acronym, expansions in acr.items():
  for expansion in expansions:
    with open("train_data/"+str(expansion)) as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        train_data_tweets.append(tweet.split())
        train_data_acronyms.append(expansion)

def create_tagged_document(split_tweets):
  for i, tweet in enumerate(split_tweets):
    yield gensim.models.doc2vec.TaggedDocument(tweet, [train_data_acronyms[i]])

train_data = list(create_tagged_document(train_data_tweets))

model = gensim.models.doc2vec.Doc2Vec(vector_soze=50, min_count=2, epochs=100)
model.build_vocab(train_data)
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

out_blob = open("doc2vec.pickle", "wb")
pickle.dump(model, out_blob)
out_blob.close()
