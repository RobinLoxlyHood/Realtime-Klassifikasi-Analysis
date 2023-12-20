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
import keras_metrics as km
import pickle
import keras
from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import StratifiedKFold
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from keras_preprocessing.sequence import pad_sequences
from varname import nameof
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import wget


import glob
import os

from datetime import date

import pymysql

def replaceSlang(word):
    if word in list(dataslang[0]):
        indexslang = list(dataslang[0]).index(word)
        return dataslang[1][indexslang]
    else:
        return word

def removeStopWords(line, stopwords):
    words = []
    for word in line:  
        word=str(word)
        word = word.strip()
        if word not in stopwords and word != "" and word != "&":
            words.append(word)
    return words

def stemmer(line):
    temp = list()
    for word in line:
        if(word not in white_list):
            word = ind_stemmer.stem(word)
        if(len(word)>3):
            temp.append(word)
        return temp

def FindMaxLength(lst): 
    maxList = max((x) for x in lst) 
    maxLength = max(len(x) for x in lst ) 
    return maxList, maxLength

def insert_to_list_rekomen_db(rp2):
    # Connect to the database
    connection = pymysql.connect(host='127.0.0.1',
                                port=3306,
                                user='root',
                                # password='f#Ur8J3N',
                                database='klasifikasi_sentimen')
    # create cursor
    cursor=connection.cursor()
        
    cols = "`,`".join([str(i) for i in rp2.columns.tolist()])
    for i,row in rp2.iterrows():
        sql = "INSERT INTO `hasil_scraping` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))

        # the connection is not autocommitted by default, so we must commit to save our changes
        connection.commit()


path = r"D:\Materi Kuliah\SKRIPSI"
os.chdir(path)

parameter_geo=pd.read_csv("Data/Data Spatial/parameter_geo.csv")
# Menghapus spasi dan menggabungkan kata-kata pada kolom "Propinsi"
parameter_geo['Propinsi'] = parameter_geo['Propinsi'].replace(' ', '', regex=True)
for i in range(len(parameter_geo)):
    today = date.today()
    yesterday = today - timedelta(days = 1)
    tokoh="Anies Baswedan"
    lokasi=str(parameter_geo.Propinsi[i])
    radius=str(parameter_geo.radius[i])+"km"
       
    twitter_search = f"{tokoh} near:{lokasi} within:{radius} lang:id until:{today} since:{yesterday}"

    # Tentukan nama file dengan format "<kueri pencarian>_<tanggal saat ini>.json"
    filename = "Anies.json"
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
        tweets_df_fix["location"] = lokasi
        tweets_df_fix.to_csv(f"Data/Hasil Crawling/{lokasi}_ANIES.csv", index=False)
        
df_anies= pd.concat(map(pd.read_csv, glob.glob(os.path.join('Data/Hasil Crawling/', "*_ANIES.csv"))))
df_anies= df_anies.reset_index(drop=True)
df_anies = df_anies[["date", "username", "tweet", "location"]]
df_anies.to_csv("Data/Hasil Crawling/data_anies.csv", index=False)
anies=pd.read_csv('Data/Hasil Crawling/data_anies.csv')
anies = anies.drop_duplicates()

# inisiasi variabel awal
start = 0
end = 1000

# loop untuk membagi data menjadi bagian-bagian dengan jumlah baris 1000
while end <= len(anies):
    df_anies = anies.iloc[start:end, 0:4]
    #menyimpan tweet. (tipe data series pandas)
    data_content = df_anies['tweet']
    # casefolding
    data_casefolding = data_content.str.lower()
    #data_casefolding.head()
    #filtering

    #url
    filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweet) for tweet in data_casefolding]
    #cont
    filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
    #punctuatuion
    filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]  #hapus simbol'[!#?,.:";@()-_/\']'
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

    #slang word
    path_dataslang = open("Data/Data Tambahan/kamus kata baku-clear (1).csv")
    dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

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

    
    data_notstopword = [removeStopWords(line,stopwords) for line in data_formal]
    white_list = ["bali"] #ini perlu/tidak perlu diubah karena dianggap sastrawi sebagai imbuhan i

    factory = StemmerFactory()
    ind_stemmer = factory.create_stemmer()


    reviews = [stemmer (line) for line in data_notstopword]
    
    #Pembuatan Kamus kata
    t  = Tokenizer()
    fit_text = reviews
    t.fit_on_texts(fit_text)

    #Pembuatan Id masing-masing kata
    sequences = t.texts_to_sequences(reviews)

    #hapus duplikat kata yang muncul
    list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

    #mencari max length sequence

        
    # Driver Code 
    max_seq, max_length_seq = FindMaxLength(list_set_sequence)
    jumlah_index = len(t.word_index) +1

    print('jumlah index : ',jumlah_index,'\n')
    # print('word_index : ',t.word_index,'\n')
    # print('index kalimat asli     : ', sequences,'\n')
    # print('kalimat tanpa duplikat : ',list_set_sequence,'\n')
    # print('panjang max kalimat : ', max_length_seq,'kata','\n')
    # print('kalimat terpanjang setelah dihapus duplikat : ', max_seq,'\n')

    count_word = [len(i) for i in list_set_sequence]
    # print('list panjang kalimat : ', count_word)
    max_len_word = max(count_word)
    # print(max_len_word)
    
    padding= pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], 
                       maxlen= 50, padding='pre')
    
    nama_model='gabungan'
    model = load_model('model/'+str(nama_model)+'/Fold'+str(2)+'.h5')
    result=[]
    for test in padding:
        pred = model.predict(np.expand_dims(test,axis=0)).argmax(axis=1)
        result.append(pred[0])
    df_anies['Tokoh'] = 'Anies Baswedan'
    df_anies['Sentiment'] = result
    df_anies.loc[df_anies['Sentiment'] == 2, 'Sentiment'] = -1
                
    insert_to_list_rekomen_db(df_anies)
    start += 1000
    end += 1000
    
parameter_geo=pd.read_csv("Data/Data Spatial/parameter_geo.csv")
# Menghapus spasi dan menggabungkan kata-kata pada kolom "Propinsi"
parameter_geo['Propinsi'] = parameter_geo['Propinsi'].replace(' ', '', regex=True)
for i in range(len(parameter_geo)):
    today = date.today()
    yesterday = today - timedelta(days = 1)
    tokoh="Ganjar Pranowo"
    lokasi=str(parameter_geo.Propinsi[i])
    radius=str(parameter_geo.radius[i])+"km"
       
    twitter_search = f"{tokoh} near:{lokasi} within:{radius} lang:id until:{today} since:{yesterday}"

    # Tentukan nama file dengan format "<kueri pencarian>_<tanggal saat ini>.json"
    filename = "Ganjar.json"
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
        tweets_df_fix["location"] = lokasi
        tweets_df_fix.to_csv(f"Data/Hasil Crawling/{lokasi}_GANJAR.csv", index=False)
        
df_ganjar= pd.concat(map(pd.read_csv, glob.glob(os.path.join('Data/Hasil Crawling/', "*_GANJAR.csv"))))
df_ganjar= df_ganjar.reset_index(drop=True)
df_ganjar = df_ganjar[["date", "username", "tweet", "location"]]
df_ganjar.to_csv("Data/Hasil Crawling/data_ganjar.csv", index=False)

ganjar=pd.read_csv('Data/Hasil Crawling/data_ganjar.csv')
ganjar = ganjar.drop_duplicates()

# inisiasi variabel awal
start = 0
end = 1000

# loop untuk membagi data menjadi bagian-bagian dengan jumlah baris 1000
while end <= len(ganjar):
    df_ganjar = ganjar.iloc[start:end, 0:4]
    #menyimpan tweet. (tipe data series pandas)
    data_content = df_ganjar['tweet']
    # casefolding
    data_casefolding = data_content.str.lower()
    #data_casefolding.head()
    #filtering

    #url
    filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweet) for tweet in data_casefolding]
    #cont
    filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
    #punctuatuion
    filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]  #hapus simbol'[!#?,.:";@()-_/\']'
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

    #slang word
    path_dataslang = open("Data/Data Tambahan/kamus kata baku-clear (1).csv")
    dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

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

    data_notstopword = [removeStopWords(line,stopwords) for line in data_formal]
    
    white_list = ["bali"] #ini perlu/tidak perlu diubah karena dianggap sastrawi sebagai imbuhan i

    factory = StemmerFactory()
    ind_stemmer = factory.create_stemmer()

    reviews = [stemmer (line) for line in data_notstopword]
    
    #Pembuatan Kamus kata
    t  = Tokenizer()
    fit_text = reviews
    t.fit_on_texts(fit_text)

    #Pembuatan Id masing-masing kata
    sequences = t.texts_to_sequences(reviews)

    #hapus duplikat kata yang muncul
    list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

    #mencari max length sequence
    # Driver Code 
    max_seq, max_length_seq = FindMaxLength(list_set_sequence)
    jumlah_index = len(t.word_index) +1

    print('jumlah index : ',jumlah_index,'\n')
    # print('word_index : ',t.word_index,'\n')
    # print('index kalimat asli     : ', sequences,'\n')
    # print('kalimat tanpa duplikat : ',list_set_sequence,'\n')
    # print('panjang max kalimat : ', max_length_seq,'kata','\n')
    # print('kalimat terpanjang setelah dihapus duplikat : ', max_seq,'\n')

    count_word = [len(i) for i in list_set_sequence]
    # print('list panjang kalimat : ', count_word)
    max_len_word = max(count_word)
    # print(max_len_word)
    
    padding= pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], 
                       maxlen= 50, padding='pre')
    
    nama_model='gabungan'
    model = load_model('model/'+str(nama_model)+'/Fold'+str(2)+'.h5')
    result=[]
    for test in padding:
        pred = model.predict(np.expand_dims(test,axis=0)).argmax(axis=1)
        result.append(pred[0])
    
    df_ganjar['Tokoh'] = 'Ganjar Pranowo'
    df_ganjar['Sentiment'] = result
    df_ganjar.loc[df_ganjar['Sentiment'] == 2, 'Sentiment'] = -1
    df_ganjar.head()
    
    df_ganjar['Sentiment'].unique()
            
    insert_to_list_rekomen_db(df_ganjar)
    start += 1000
    end += 1000

parameter_geo=pd.read_csv("Data/Data Spatial/parameter_geo.csv")
# Menghapus spasi dan menggabungkan kata-kata pada kolom "Propinsi"
parameter_geo['Propinsi'] = parameter_geo['Propinsi'].replace(' ', '', regex=True)    
for i in range(len(parameter_geo)):
    today = date.today()
    yesterday = today - timedelta(days = 1)
    tokoh="Prabowo Subianto"
    lokasi=str(parameter_geo.Propinsi[i])
    radius=str(parameter_geo.radius[i])+"km"
       
    twitter_search = f"{tokoh} near:{lokasi} within:{radius} lang:id until:{today} since:{yesterday}"

    # Tentukan nama file dengan format "<kueri pencarian>_<tanggal saat ini>.json"
    filename = "Prabowo.json"
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
        tweets_df_fix["location"] = lokasi
        tweets_df_fix.to_csv(f"Data/Hasil Crawling/{lokasi}_PRABOWO.csv", index=False)
        
df_prabowo= pd.concat(map(pd.read_csv, glob.glob(os.path.join('Data/Hasil Crawling/', "*_PRABOWO.csv"))))
df_prabowo= df_prabowo.reset_index(drop=True)
df_prabowo = df_prabowo[["date", "username", "tweet", "location"]]
df_prabowo.to_csv("Data/Hasil Crawling/data_prabowo.csv", index=False)

prabowo=pd.read_csv('Data/Hasil Crawling/data_prabowo.csv')
prabowo = prabowo.drop_duplicates()
prabowo.tail()

# inisiasi variabel awal
start = 0
end = 1000

# loop untuk membagi data menjadi bagian-bagian dengan jumlah baris 1000
while end <= len(prabowo):
    df_prabowo = prabowo.iloc[start:end, 0:4]
    #menyimpan tweet. (tipe data series pandas)
    data_content = df_prabowo['tweet']
    # casefolding
    data_casefolding = data_content.str.lower()
    #data_casefolding.head()
    #filtering

    #url
    filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweet) for tweet in data_casefolding]
    #cont
    filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
    #punctuatuion
    filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]  #hapus simbol'[!#?,.:";@()-_/\']'
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

    #slang word
    path_dataslang = open("Data/Data Tambahan/kamus kata baku-clear (1).csv")
    dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

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

    data_notstopword = [removeStopWords(line,stopwords) for line in data_formal]
    white_list = ["bali"] #ini perlu/tidak perlu diubah karena dianggap sastrawi sebagai imbuhan i

    factory = StemmerFactory()
    ind_stemmer = factory.create_stemmer()

    reviews = [stemmer (line) for line in data_notstopword]
    
    #Pembuatan Kamus kata
    t  = Tokenizer()
    fit_text = reviews
    t.fit_on_texts(fit_text)

    #Pembuatan Id masing-masing kata
    sequences = t.texts_to_sequences(reviews)

    #hapus duplikat kata yang muncul
    list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

    #mencari max length sequence
        
    # Driver Code 
    max_seq, max_length_seq = FindMaxLength(list_set_sequence)
    jumlah_index = len(t.word_index) +1

    print('jumlah index : ',jumlah_index,'\n')
    # print('word_index : ',t.word_index,'\n')
    # print('index kalimat asli     : ', sequences,'\n')
    # print('kalimat tanpa duplikat : ',list_set_sequence,'\n')
    # print('panjang max kalimat : ', max_length_seq,'kata','\n')
    # print('kalimat terpanjang setelah dihapus duplikat : ', max_seq,'\n')

    count_word = [len(i) for i in list_set_sequence]
    # print('list panjang kalimat : ', count_word)
    max_len_word = max(count_word)
    # print(max_len_word)
    
    padding= pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], 
                       maxlen= 50, padding='pre')
    
    nama_model='gabungan'
    model = load_model('model/'+str(nama_model)+'/Fold'+str(2)+'.h5')
    result=[]
    for test in padding:
        pred = model.predict(np.expand_dims(test,axis=0)).argmax(axis=1)
        result.append(pred[0])
        
    df_prabowo['Tokoh'] = 'Prabowo Subianto'
    df_prabowo['Sentiment'] = result
    df_prabowo.loc[df_prabowo['Sentiment'] == 2, 'Sentiment'] = -1
    
            
    insert_to_list_rekomen_db(df_prabowo)
    start += 1000
    end += 1000
    
print("Proses RealTime Selesai")



