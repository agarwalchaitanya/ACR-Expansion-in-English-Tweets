import numpy as numpy
import sys
from tqdm import tqdm
import os
import pandas as pd 
import string
import gensim
import json
import pickle

#(x)define a dictionary containing the expansion and acronym; and an array of acronyms 
#(x)iterate over the files in the directory
#()for each file, train a common doc2vec, and a unique classifier for for each acronym

acr = {
  'brb': ['be right back', 'bathroom break'],
  'cc': ['carbon copy', 'i understand'],
  'dl': ['download', 'down low', 'doing laundry'],
  'eta': ['estimated time of arrival', 'edited to add'],
  'gf': ['girlfriend', 'gluten-free'],
  'gg': ['good game', 'good grief'],
  'gl': ['good luck', 'get lost'],
  'hoas': ['hold on a second', 'heck of a shot'],
  'hw': ['homework', 'hardware'],
  'ic': ['i see', 'in character'],
  'im': ['instant messenger', 'instant message'],
  'k': ['ok', 'kiss'],
  'lol': ['laughing out loud', 'league of legends', 'lots of love', 'little old lady'],
  'n/a': ['not availabale', 'not applicable'],
  'nc': ['no comment', 'nice call', 'not cool'],
  'nm': ['not much', 'nevermind'],
  'np': ['no problem', 'neope ts'],
  'ot': ['off topic', 'other topic', 'overtime'],
  'pm': ['pm', 'private message'],
  'pos': ['parent over shoulder', 'piece of shit', 'power of suggestion'],
  're': ['regarding', 'resident evil'],
  'rotfl': ['rolling on the floor laughing', 'rolling over freakin\' laughing'],
  'smh':['shaking my head', 'smash my head', 'scratching my head'],
  'sos': ['someone over shoulder', 'save our souls', 'same old stuff', 'someone special'],
  'tc': ['take care', 'that\'s cool'],
  'ur': ['your', 'you are'],
  'wb': ['write back', 'welcome back', 'way bored'],
  'y': ['why', 'yawning']
}
train_data_tweets = []
train_data_acronyms = []
print("processing data!")
for acronym, expansions in acr.items():
  for expansion in expansions:
    with open("data/"+str(expansion)) as file:
      tweets = file.readlines()
      for tweet in tweets:
        tweet = json.loads(tweet)
        tweet = tweet.lower()
        tweet = tweet.replace(expansion, acronym)
        train_data_tweets.append(tweet.split())
        train_data_acronyms.append(acronym)

def create_tagged_document(split_tweets):
  for i, tweet in enumerate(split_tweets):
    yield gensim.models.doc2vec.TaggedDocument(tweet, [train_data_acronyms[i]])

train_data = list(create_tagged_document(train_data_tweets))
print("data processed!")
print("training data!")
model = gensim.models.doc2vec.Doc2Vec(vector_soze=50, min_count=2, epochs=100)
model.build_vocab(train_data)
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
print("data trained")

out_blob = open("doc2vec.pickle", "wb")
pickle.dump(model, out_blob)
out_blob.close()