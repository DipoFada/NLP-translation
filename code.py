#Getting started
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

corpus_data = pd.read_csv('eng_-french.csv')
corpus_data.head()
#counting the number of words in each row
corpus_data['English words'] = corpus_data['English words/sentences'].str.split().str.len()
corpus_data ['French words'] = corpus_data['French words/sentences'].str.split().str.len()
corpus_data

#splitting the dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(corpus_data, test_size = 0.2)
#splitting the train data into training and validation
train_data, valid_data = train_test_split (train_data, test_size = 0.2)

freq_eng = corpus_data['English words/sentence'].str.split(expand = True).stack().value_counts().reset_index()
freq_fr = corpus_data['French words/sentences'].str.split(expand = True).stack().value_counts().reset_index()
freq_eng.to_csv('English_freq.csv', index = False)
freq_fr.to_csv('French_freq.csv', index = False)

freq_eng = pd.read_csv('English_freq.csv')
freq_fr = pd.read_csv ('French_freq.csv')

freq_eng
freq_fr

# preparing the data
def get_data(word_lines):
  text = []
  for line in word_lines:
    text.append('[start] ' + line + ' [end]')
  return text

eng_train = get_data(list(train_data['English words/sentences']))
eng_train
fr_train = get_data(list(train_data ['French words/sentences']))
fr_train

eng_valid = get_data (list(valid_data['English words/sentences']))
fr_valid = get_data (list(valid_data['English words/sentences']))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#input language tokenization and padding
fr_token = Tokenizer (filters = '', lower = False)
fr_token.fit_on_texts (fr_train)
fr_tokenized = fr_token.texts_to_sequences (fr_train)
fr_padded = pad_sequences (fr_tokenized, padding = 'post')

#target language tokenization and paddinf 
eng_token = Tokenizer (filters = '', lower = False)
eng_token.fit_on_texts (eng_train)
eng_tokenized = eng_token.texts_to_sequences (eng_train)
eng_padded = pad_sequences (eng_tokenized, padding = 'post')
eng_padded

#getting the unique tokens in input and output languages
output_tokens = len(fr_token.word_index)
input_tokens = len (eng_token.word_index)
print ('The number of unique output tokens is ', output_tokens)
print ('The number of unique input tokens is ', input_tokens)   

#maximum length of a sentence in both languages
max_len_op = fr_padded.shape[1]
max_len_ip = eng_padded.shape[1]
max_len_op

from ktext.preprocess import processor
eng_pp = processor (keep_n = 10407, padding_maxlen = 7)
eng_train_vecs = eng_pp.fit_transform (eng_train)
