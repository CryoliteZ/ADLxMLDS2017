import numpy as np
import pandas as pd
import os, sys
import pickle
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential, load_model, model_from_json




# TRAIN_LABEL_PATH = "MLDS_hw2_data/training_label.json"

def getWords(word_index, seq):
    for word in seq:
        idx = np.argmax(word)
        if idx == 0:
            continue
        print(list(word_index.keys())[list(word_index.values()).index(idx)])
    
def read_data(path):
    with open(path) as data_file:    
        y_data = json.load(data_file)
    videoId = []
    videoSeq = []
    for y in y_data:
        for idx, cap in enumerate(y['caption']):
            cap = "<bos> " + cap + " <eos>"
            videoId.append(y['id'])
            videoSeq.append(cap)
    return videoId, videoSeq

def getVId(path):
    TRAIN_LABEL_PATH = "MLDS_hw2_data/training_label.json"
    videoId, videoSeq = read_data(path)
    tokenize(videoId, videoSeq)
    curFilename = videoId[0]
    vCount = 0
    y_videoId = []
    for idx, seq in  enumerate(videoId):
            if(videoId[idx] == curFilename):
                vCount = vCount + 1
                if(vCount > 2):
                    continue
            else:
                vCount = 1
                curFilename = videoId[idx]
            y_videoId.append(videoId[idx])
    return videoId

def tokenize(videoId, videoSeq):
    MAX_WORDS = 1024
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(videoSeq)
    word_index = tokenizer.word_index
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(videoSeq)
    train_sequences = np.array(train_sequences)
    print(train_sequences[0])

    train_sequences = pad_sequences(train_sequences)
    print(train_sequences.shape)
    max_seq_length = train_sequences.shape[1]
    print(max_seq_length)
    y_data = []
    y_videoId= []
    curFilename = videoId[0]
    vCount = 0
    for idx, seq in  enumerate(train_sequences):
        if(videoId[idx] == curFilename):
            vCount = vCount + 1
            if(vCount > 2):
                continue
        else:
            vCount = 1
            curFilename = videoId[idx]
        y = to_categorical(seq, MAX_WORDS)
        y_data.append(y)
        y_videoId.append(videoId[idx])
    y_data = np.array(y_data)   
    print(y_data.shape)
  
    return y_videoId, y_data, tokenizer

def getX_data(path):
    x_data = {}
    for filename in os.listdir(path):
        f = np.load(os.path.join(path, filename))
        x_data[filename[:-4]] = f
    X_data = []
    for vid in y_videoId:
        X_data.append(x_data[vid])
    X_data = np.array(X_data)
    return X_data





        

    

data_directory = sys.argv[1]
output_file = sys.argv[2]
videoId, videoSeq = read_data(os.path.join(data_directory, "training_label.json"))
y_videoId, y_data, tokenizer= tokenize(videoId, videoSeq)
X_data = getX_data(os.path.join(data_directory, "training_data/feat/"))
print(X_data.shape)

