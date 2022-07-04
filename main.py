#Author: Smith Cheablam
#Date: July 03, 2022.

# !pip install pythainlp
# !pip install -U marisa-trie

from glob import glob
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

import pandas as pd
import uvicorn
import numpy as np
from numpy import load
import re
import pickle
# import joblib

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV

import pythainlp
from pythainlp import sent_tokenize, word_tokenize, Tokenizer
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
from marisa_trie import Trie

app = FastAPI()

def clean_unseen(text):
  return re.sub(r'[^ก-๙]', '', text).replace('\n', ' ')

def cut_tokenize(text):
  r_tk = word_tokenize(text, engine="newmm", keep_whitespace=False)
  return r_tk

# def norm_spell_check(r_tk):
#   r_sp = [normalize(correct(w)) for w in r_tk] # แต่จะใช้ไม่ได้่กับคำว่า มากกว่าาา
#   return r_sp

def rm_stopword(r_sp):
  stopwords = list(set(thai_stopwords()) - set(['ไม่']))
  # stopwords.append('')
  r_rm = [i for i in r_sp if i not in stopwords]
  return r_rm

def identity_fun(text):
    return text

def preprocess(text):
  text_tk = []
  r_tk = cut_tokenize(text)
  # r_sp = norm_spell_check(r_tk)
  # r_rm = rm_stopword(r_sp)
  r_rm = rm_stopword(r_tk)
  text_tk.append(r_rm)
  return text_tk

X_train = load('X_train.npy', allow_pickle=True)
global count_vectorizer
count_vectorizer = CountVectorizer(
                          analyzer = 'word', #this is default
                          tokenizer=identity_fun, #does no extra tokenizing
                          preprocessor=identity_fun, #no extra preprocessor
                          token_pattern=None
                          )
count_vectorizer = count_vectorizer.fit(X_train)

def convert_to_cnt_vec(text):
  # load_cnt_vec = joblib.load("count_vectorizer.pkl")
  text_count_vec = count_vectorizer.transform(text)
  # print(text_count_vec)
  return text_count_vec


global lr_service
global lr_atmosphere
global lr_cleanliness
global lr_price
global lr_food

lr_service = pickle.load(open('LR\LR_service.pkl', 'rb'))
# lr_atmosphere = pickle.load(open('LR\LR_atmosphere.pkl', 'rb'))
# lr_cleanliness = pickle.load(open('LR\LR_cleanliness.pkl', 'rb'))
# lr_price = pickle.load(open('LR\LR_price.pkl', 'rb'))
lr_food = pickle.load(open('LR\LR_food.pkl', 'rb'))

"""Real Service"""
@app.get("/classify_review")
def classify_review(text:str = 'ข้อความรีวิวร้านอาหาร'):
    # class_names = ['service', 'atmosphere', 'cleanliness', 'price', 'food']
    clean_text = clean_unseen(text)
    text_tk = preprocess(clean_text)
    text_count_vec = convert_to_cnt_vec(text_tk)
    # return text_count_vec

    text_pred_s = lr_service.predict(text_count_vec)[0]
    # text_pred_a = lr_atmosphere.predict(text_count_vec)[0]
    # text_pred_c = lr_cleanliness.predict(text_count_vec)[0]
    # text_pred_p = lr_price.predict(text_count_vec)[0]
    text_pred_f = lr_food.predict(text_count_vec)[0]
    
    return {'service': str(text_pred_s), 
            # 'atmosphere': str(text_pred_a), 
            # 'cleanliness': str(text_pred_c), 
            # 'price': str(text_pred_p), 
            'food(taste)': str(text_pred_f)}
    return text

if __name__ == '__main__':
   uvicorn.run(app, host="0.0.0.0", port=8069, debug=True) 