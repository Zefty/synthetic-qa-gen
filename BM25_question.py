# Install BM25
import math
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("stopwords")

# Setup functions for cleaning text and stop word dropping  
english_stopwords = list(set(stopwords.words('english')))

def strip_characters(text):
    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)
    t = re.sub('/', ' ', t)
    t = t.replace("'",'')
    return t

def clean(text):
    t = text.lower()
    t = strip_characters(t)
    return t

def tokenize(text):
    words = nltk.word_tokenize(text)
    return list(set([word for word in words 
                            if len(word) > 1
                            and not word in english_stopwords
                            and not word.isnumeric()
                            and word.isalpha()
                    ]
                   )
                )

def preprocess(text):
    t = clean(text)
    tokens = tokenize(t)
    return tokens

knowledgebase = pd.read_csv('splitted_covid_dump-covidQA.csv', sep = '\t', names = ['title', 'text'])

# Process the qa.csv file and create a BM25 corpus 
BM25Corpus = knowledgebase['text'].fillna("").apply(preprocess).to_frame()

# Create the BM25 object
BM25 = BM25Okapi(BM25Corpus['text'].tolist())



############################# Train.source
qcovidsource = pd.read_csv('data/input_data/Q-covid-val/train.source', names = ['question'])

topK = 1 
textfile = open("data/output_data/Q-covid-BM25-val/train.source", "w")
for idx, r in tqdm(qcovidsource.iterrows(), total = qcovidsource.shape[0]): # Iterate each synthetic QA pair in QA.csv and convert into DPR data format above 
    question = preprocess(r['question'])
    docScores = BM25.get_scores(question)
    idxTopKDocs = np.argsort(docScores)[::-1][:topK]
    textfile.write(r['question'] + ' <BM25> ' + knowledgebase['text'][idxTopKDocs].to_list()[0] + "\n")
textfile.close()



# ############################# Val.source
# qcovidsource = pd.read_csv('data/input_data/Q-covid/val.source', names = ['question'])

# topK = 1 
# textfile = open("data/output_data/Q-covid-BM25/val.source", "w")
# for idx, r in tqdm(qcovidsource.iterrows(), total = qcovidsource.shape[0]): # Iterate each synthetic QA pair in QA.csv and convert into DPR data format above 
#     question = preprocess(r['question'])
#     docScores = BM25.get_scores(question)
#     idxTopKDocs = np.argsort(docScores)[::-1][:topK]
#     textfile.write(r['question'] + ' <BM25> ' + knowledgebase['text'][idxTopKDocs].to_list()[0] + "\n")
# textfile.close()