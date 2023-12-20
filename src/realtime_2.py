import nltk
import pandas as pd
import re
import time
import multiprocessing
import io
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
from datetime import timedelta
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from varname import nameof
import os
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#Membaca CSV Hasil Crawling Terbaru

today = "2023-06-16"#date.today()
yesterday = "2023-06-17"#today - timedelta(days = 1)
kata_kunci="Kota Tegal"
       
twitter_search = f"{kata_kunci} lang:id until:{yesterday} since:{today}"

# Tentukan nama file dengan format "<kueri pencarian>_<tanggal saat ini>.json"
filename = "Kota_Tegal.json"
USING_TOP_SEARCH = True

snscrape_params = '--jsonl'
twitter_search_params = ''

if USING_TOP_SEARCH:
    twitter_search_params += "--top"

os.system(f"cmd /c snscrape {snscrape_params} twitter-search {twitter_search_params} \"{twitter_search}\" > {filename}")
tweets_df = pd.read_json(filename, lines=True)
if len(tweets_df) != 0:    
    tweets_df_rename=tweets_df.rename(columns={"date": "date", "username": "username", "rawContent": "tweet"})
    tweets_df_fix=tweets_df_rename[["date","username", "tweet"]]
    tweets_df_fix.to_csv(f"Kota_Tegal.csv", index=False)
    

data_crawling_kota_tegal = pd.read_csv("Kota_Tegal.csv", lines=True)    
df = data_crawling_kota_tegal[["date", "username", "tweet", "location"]]

## Preprocessing
#menyimpan tweet. (tipe data series pandas)
data_content = df['tweet']
# casefolding
data_casefolding = data_content.str.lower()
#filtering

#url
filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str(tweet)) for tweet in data_casefolding]
#cont
filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
#punctuatuion
filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]
#  hapus #tagger
filtering_tagger = [re.sub(r'#([^\s]+)', '', tweet) for tweet in filtering_punctuation]
#numeric
filtering_numeric = [re.sub(r'\d+', ' ', tweet) for tweet in filtering_tagger]

# # filtering RT , @ dan #
# fungsi_clen_rt = lambda x: re.compile('\#').sub('', re.compile('rt @').sub('@', x, count=1).strip())
# clean = [fungsi_clen_rt for tweet in filtering_numeric]

data_filtering = pd.Series(filtering_numeric)
# #tokenize
tknzr = TweetTokenizer()
data_tokenize = [tknzr.tokenize(tweet) for tweet in data_filtering]
data_tokenize
#slang word
path_dataslang = open("kamus kata baku-clear (1).csv")
dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

def replaceSlang(word):
  if word in list(dataslang[0]):
    indexslang = list(dataslang[0]).index(word)
    return dataslang[1][indexslang]
  else:
    return word

data_formal = []
for data in data_tokenize:
  data_clean = [replaceSlang(word) for word in data]
  data_formal.append(data_clean)
len_data_formal = len(data_formal)
# print(data_formal)
# len_data_formal
nltk.download('stopwords')
default_stop_words = nltk.corpus.stopwords.words('indonesian')
stopwords = set(default_stop_words)

def removeStopWords(line, stopwords):
  words = []
  for word in line:  
    word=str(word)
    word = word.strip()
    if word not in stopwords and word != "" and word != "&":
      words.append(word)

  return words
reviews = [removeStopWords(line,stopwords) for line in data_formal]

# Specify the file path of the pickle file
file_path = 'model2\data_train.pickle'

# Read the pickle file
with open(file_path, 'rb') as file:
    data_train = pickle.load(file)
    
# pembuatan vector kata
vectorizer = TfidfVectorizer()
train_vector = vectorizer.fit_transform(data_train)
reviews2 = [" ".join(r) for r in reviews]

## Implementasi Aspek Hiburan
model_aspek = pickle.load(open('model2/tfidf_Model_Aspek_Wisata_Hiburan_nvb.pkl','rb'))
model_sentiment = pickle.load(open('model2/vektor_tfidf_Model_Sentimen_Wisata_Hiburan_nvb.pkl','rb'))

result = []

for test in reviews2:
    test_data = [str(test)]
    test_vector = vectorizer.transform(test_data)
    pred = model_aspek.predict(test_vector)
    if pred != 1:
        result.append(-1)
    else:
        pred = model_sentiment.predict(test_vector)
        result.append(pred[0])

        
from sklearn.utils.multiclass import unique_labels
unique_labels(result)

df['wisata_hiburan'] = result

## Implementasi Aspek Pendidikan
model_aspek = pickle.load(open('model2/tfidf_Model_Aspek_Wisata_Pendidikan_nvb.pkl','rb'))
model_sentiment = pickle.load(open('model2/vektor_tfidf_Model_Sentimen_Pendidikan_nvb.pkl','rb'))

result = []

for test in reviews2:
    test_data = [str(test)]
    test_vector = vectorizer.transform(test_data)
    pred = model_aspek.predict(test_vector)
    if pred != 1:
        result.append(-1)
    else:
        pred = model_sentiment.predict(test_vector)
        result.append(pred[0])

from sklearn.utils.multiclass import unique_labels
unique_labels(result)

df['pendidikan'] = result

## Implementasi Aspek Fasilitas dan Layanan Publik
model_aspek = pickle.load(open('model2/tfidf_Model_Aspek_Wisata_Fasilitas_Layanan_Publik_nvb.pkl','rb'))
model_sentiment = pickle.load(open('model2/vektor_tfidf_Model_Sentimen_fasilitas_layanan_publik_nvb.pkl','rb'))

result = []

for test in reviews2:
    test_data = [str(test)]
    test_vector = vectorizer.transform(test_data)
    pred = model_aspek.predict(test_vector)
    if pred != 1:
        result.append(-1)
    else:
        pred = model_sentiment.predict(test_vector)
        result.append(pred[0])

from sklearn.utils.multiclass import unique_labels
unique_labels(result)

df['fasilitas_layanan_publik'] = result

## Implementasi Aspek Kuliner
model_aspek = pickle.load(open('model2/tfidf_Model_Aspek_Wisata_Kuliner_nvb.pkl','rb'))
model_sentiment = pickle.load(open('model2/vektor_tfidf_Model_Sentimen_Kuliner_nvb.pkl','rb'))

result = []

for test in reviews2:
    test_data = [str(test)]
    test_vector = vectorizer.transform(test_data)
    pred = model_aspek.predict(test_vector)
    if pred != 1:
        result.append(-1)
    else:
        pred = model_sentiment.predict(test_vector)
        result.append(pred[0])

from sklearn.utils.multiclass import unique_labels
unique_labels(result)

df['kuliner'] = result

## Store ke DB
df=df.dropna()
df=df.drop_duplicates()
df.head()
import pymysql
def insert_to_list_rekomen_db(rp2):
    # Connect to the database
    connection = pymysql.connect(host='127.0.0.1',
                                 port=3306,
                                 user='root',
                                 # password='f#Ur8J3N',
                                 database='dashboard_kotategal_db')
    # create cursor
    cursor=connection.cursor()
    
    cols = "`,`".join([str(i) for i in rp2.columns.tolist()])
    for i,row in rp2.iterrows():
        sql = "INSERT INTO `hasil_sentiment` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))

        # the connection is not autocommitted by default, so we must commit to save our changes
        connection.commit()
        
insert_to_list_rekomen_db(rp2=df)