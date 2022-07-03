#Author: Smith Cheablam
#Date: July 03, 2022.

# !pip install pythainlp
# !pip install -U marisa-trie

from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np
import re
import pickle
import joblib
from fastapi.responses import PlainTextResponse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

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

def convert_to_cnt_vec(text):
  load_cnt_vec = joblib.load("\restaurant_review_clf\count_vectorizer.pkl")
  text_count_vec = load_cnt_vec.transform(text)
  return text_count_vec

@app.get("/Classify_Review_Category")
async def predict_unseen(text:str = 'ข้อความรีวิวร้านอาหาร'):
    class_names = ['service', 'atmosphere', 'cleanliness', 'price', 'food']
    clean_text = clean_unseen(text)
    text_tk = preprocess(clean_text)
    text_count_vec = convert_to_cnt_vec(text_tk)
    # return text_count_vec

    # Loop for predict each class
    df_unseen_text = pd.DataFrame(columns=['review', 'service', 'atmosphere', 'cleanliness', 'price', 'food'])
    check_text = dict.fromkeys(class_names, 0)
    for c in class_names:
        best_model_c = pickle.load(open(f"\restaurant_review_clf\LR_{c}.pkl", "rb"))
        text_pred = best_model_c.predict(text_count_vec)[0]
        # check_text is already dict / json format
        check_text[c] = text_pred

    return check_text

if __name__ == '__main__':
   uvicorn.run(app, host="0.0.0.0", port=80, debug=True) 