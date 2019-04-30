from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from acrlist import acr
import pickle
import os
import tensorflow as tf 
import keras
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np
from ACR.settings import BASE_DIR as base

# Create your views here.
d_model = pickle.load(open(base+'/doc2vec.pickle', 'rb'))


def index(request):    
    if request.method == 'POST':
      search_string = request.POST['search_string']
      search_string = search_string.lower()
      for acronym, expansions in acr.items():
        if acronym in search_string.split():
          infer_vector = d_model.infer_vector(search_string.split())
          keras.backend.clear_session()
          classifier = load_model(base+"/classifiers/"+str(acronym)+".h5")
          test = np.array(tf.keras.utils.normalize(infer_vector))
          dist = classifier.predict(test)
          prediction = classifier.predict_classes(test)
          prediction_ = expansions[np.asscalar(prediction)]
          template = loader.get_template('search/index.html')
          context = {
            'search_string' : search_string,
            'prediction' : prediction_,
            'acronym' : acronym,
          }
          return HttpResponse(template.render(context, request))
    return render(request, 'search/index.html')