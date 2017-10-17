import numpy as np
import pandas as pd
import keras
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed, Bidirectional, Dense,Dropout, GRU
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys,os,time



MAX_SEQUENCE_NUM = 777
path, dirs, files = os.walk("./models/").__next__()
MODEL_NUM = len(files)+1


def mapPhones():
    labelList = []
    phone2num = dict()
    with open('./48phone_char.map') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')           
            phones = line.split('\t')
            phone2num[phones[0]] = phones[1]
            labelList.append(phones[2])
    phone39 = dict()
    with open('./phones/48_39.map') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            phones = line.split('\t')
            phone39[phones[0]] = phones[1]
    labelIdxList = []
    for key, value in phone39.items():
        label = phone2num[value]
        if(label not in labelIdxList):
            labelIdxList.append(label)
    return phone2num, phone39, labelIdxList, labelList


def loadData(mfcc_path, labels_path):
    phone2num, phone39, labelIdxList, labelList  = mapPhones()

    # fbank_features = ['fb_' + str(i) for i in range(0,69predict_classes)]
    mfcc_features = ['mfcc_' + str(i) for i in range(0,39)]
 
    # fbank_train = pd.read_csv('./fbank/train.ark', sep=' ', header = None, index_col=0, names = ['id'] + fbank_features)
    mfcc_data = pd.read_csv(mfcc_path, sep=' ', header = None, index_col=0, names = ['id'] + mfcc_features)
    labels = pd.read_csv(labels_path, sep=',', header = None, index_col=0, names = ['id', 'label'])
    df = pd.concat([mfcc_data, labels], axis=1)  
    # map label frome phone48 -> phone39 -> number -> index of number in labelList -> +1
    for key, value in phone39.items():        
        df.loc[df['label'] == key, 'label'] = labelIdxList.index(phone2num[value]) + 1
    
    df['f_id'] = df.index
    df['fid'] = df['f_id'].apply(lambda x: x.split('_')[2])
    df[['fid']]= df[['fid']].apply(pd.to_numeric)
    df['f_name'] =  df['f_id'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    del df['f_id']

    df = df.sort_values(by=['f_name', 'fid'])
    df_g = df.groupby('f_name')
    df_g = np.array(list(df_g))
    
    df_g = np.delete(df_g, 0, 1)
    
    X_data = []
    y_data = []
    for rows in (df_g):
        labels =  rows[0].as_matrix(['label'])        
        labels = to_categorical(labels, num_classes = 40)      
        mfcc = rows[0].as_matrix(mfcc_features)    
        mfcc = preprocessing.scale(mfcc)
        padding_num = MAX_SEQUENCE_NUM - mfcc.shape[0]
        padding_zeros = np.zeros((padding_num, 39))
        
        padding_labels = np.zeros(padding_num)
        padding_labels = to_categorical(padding_labels, num_classes = 40)
        
        mfcc = np.concatenate((mfcc, padding_zeros), axis = 0)
        labels = np.concatenate((labels, padding_labels), axis = 0)
        X_data.append(mfcc)
        y_data.append(labels)
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    # print(X_data[0].shape)
    # print(y_data[0].shape)
    return X_data, y_data, df_g
   

def genModel(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(512, return_sequences=True, activation='relu', dropout=0.4), input_shape=input_shape,))
    model.add(Bidirectional(GRU(512, return_sequences=True, activation='relu', dropout=0.4)))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.4))

    model.add(TimeDistributed(Dense(40, activation='softmax')))
    model.summary()
  

    opt = keras.optimizers.adam(lr = 0.001)
    model.compile(optimizer= opt,
                loss='categorical_crossentropy',
                metrics = ['accuracy']               
                )
    return model

def train():
    X_data, y_data, df = loadData('./mfcc/train.ark', './label/train.lab')
    
    input_shape = (X_data.shape[1], X_data.shape[2])

    # test size and random seed
    tsize = 0.1
    rnState = 42
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size= tsize, random_state=rnState)
    
    model = genModel(input_shape)
    
    model_json = model.to_json()
    with open('./models_checkpoint/model'+ str(MODEL_NUM) + '.json', "w") as json_file:
        json_file.write(model_json)

    earlystopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath=  './models_checkpoint/model'+ str(MODEL_NUM) + 'best.h5',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min')

    batchSize = 40
    epoch = 100
    model.fit(X_train, y_train,
        validation_data=(X_valid, y_valid), 
        batch_size = batchSize,
        epochs = epoch,
        callbacks=[earlystopping,checkpoint]) 

    scores = model.evaluate(X_valid, y_valid, verbose=0)
    benchmark = str(scores[0])[:8]
   

    model.save('models/model'+ str(MODEL_NUM) +'_' + benchmark + '.h5') 
    print(scores)

def loadTestData(mfcc_path):
    phone2num, phone39, labelIdxList, labelList   = mapPhones()

    mfcc_features = ['mfcc_' + str(i) for i in range(0,39)]
    df = pd.read_csv('./mfcc/test.ark', sep=' ', header = None,  names = ['id'] + mfcc_features)
    df['fid'] = df['id'].apply(lambda x: x.split('_')[2])
    df[['fid']]= df[['fid']].apply(pd.to_numeric)
    df['f_name'] =  df['id'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])

    df_g = df.groupby('f_name', sort=False)
    df_g = np.array(list(df_g))
    df_list = np.delete(df_g, 0, 1)
    X_test = []
    X_test_lens = []  
    for rows in (df_list):
        mfcc = rows[0].as_matrix(mfcc_features)    
        mfcc = preprocessing.scale(mfcc)
        padding_num = MAX_SEQUENCE_NUM - mfcc.shape[0]
        X_test_lens.append(mfcc.shape[0])

        padding_zeros = np.zeros((padding_num, 39))
        mfcc = np.concatenate((mfcc, padding_zeros), axis = 0)
        X_test.append(mfcc)
    X_test = np.asarray(X_test)
    print(df_g.shape)

    return X_test, X_test_lens, labelIdxList, labelList

def getFrameID():
    data = pd.read_csv('sample.csv', sep=',', header = 0,  names = ['id', 'phone_sequence'] )
    return data
    # print(data)

def test(path):
    X_test, X_test_lens, labelIdxList, labelList = loadTestData('./mfcc/test.ark')
    model = load_model(path)
    X_predict = []
    result = model.predict_classes(X_test)
    result.shape
    for idx, seq in enumerate(result):
        seq = seq[:X_test_lens[idx]]
        X_predict.append(seq)
    
    X_predict = np.array(X_predict)
    output = getFrameID()

    frames = []
    for x in X_predict:
        frame = ""
        for labelIdx in x:
            if(labelIdx > 0):
                frame += labelList[int(labelIdxList[labelIdx-1])]
        frame = frame.strip('L')

        newFrame = ""
        count = 1
        for i in range(1, len(frame)):
            preChar = frame[i-1]
            if(frame[i] == preChar):
                count = count + 1
            else:
                if(count >= 2):
                    newFrame += preChar
                count = 1
        if(count > 2):
            newFrame += frame[len(frame)-1]
        frames.append(newFrame)
    output['phone_sequence'] = frames
    output.to_csv('result'+ str(MODEL_NUM)+'.csv', header=True, index=False, sep=',', mode='w', columns=['id','phone_sequence' ])
    
    print(output)
   
if(len(sys.argv) > 1):
    if(sys.argv[1] == 'train'):
        train()
    else:
        test(sys.argv[2])
else:
    train()
# test()



