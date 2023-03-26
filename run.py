import nltk
import pandas as pd
import re
import tensorflow as tf
import time
import multiprocessing
import io
import gensim
import numpy as np
import matplotlib.pyplot as plt
#import keras_metrics as km
import pickle
import keras
from datetime import date

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Dropout, GRU
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
from datetime import timedelta
from gensim.models import word2vec
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from varname import nameof

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import wget

from numpy import asarray
from numpy import savetxt

"""# Scraping"""
import twint
import nest_asyncio

nest_asyncio.apply()

def column_names():
  return twint.output.panda.Tweets_df.columns

def twint_to_pd(columns):
  return twint.output.panda.Tweets_df[columns]




def main():
    #getdatetoday
    today = date.today()
    today=today.strftime('%Y-%m-%d')
    parameter_geo=pd.read_csv("parameter_geo.csv")
    for i in range(len(parameter_geo)):
      wilayah= parameter_geo.Propinsi[i]
      latitude=parameter_geo.lat_centroid[i]
      longitude=parameter_geo.lon_centroid[i]
      radius=parameter_geo.radius[i]
      param=f"{latitude},{longitude},{radius}"+"km"
      c= twint.Config()
      c.Since = "2023-01-01"
      c.Until = today
      c.Search = "Ganjar Pranowo"
      # c.Limit=1000
      c.Geo=param
      c.Pandas = True
      twint.run.Search(c)
      data = twint_to_pd(['date','username','tweet'])
      data["location"] = wilayah
      judul=f"{wilayah}_JAN_to_Feb_2021.csv"
      data.to_csv(judul, index=False)
    
    

if __name__ == "__main__":
    main()