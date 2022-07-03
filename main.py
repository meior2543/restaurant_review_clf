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

def result(res):
    return {"result":res}

@app.get("/")
async def main():
    return 'Hello World'

@app.get("/validation-email")
async def validation_email(text):  
    regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(regex,text):
        return True
    else:
        return False

if __name__ == '__main__':
   uvicorn.run(app, host="0.0.0.0", port=80, debug=True) 