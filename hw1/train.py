import numpy as np
import pandas as pd
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed, Bidirectional, Dense,Dropout, GRU
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation

MAX_SEQUENCE_NUM = 777


def mapPhones():
    phone2num = dict()
    with open('./48phone_char.map') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')           
            phones = line.split('\t')
            phone2num[phones[0]] = phones[1]
    phone39 = dict()
    with open('./48_39.map') as f:
        
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            phones = line.split('\t')
            phone39[phones[0]] = phones[1]
    
    labelList = []
    for key, value in phone39.items():
        label = phone2num[value]
        if(label not in labelList):
            labelList.append(label)
   
    
    return phone2num, phone39, labelList


def loadData(mfcc_path, labels_path):
    phone2num, phone39, labelList  = mapPhones()

    # fbank_features = ['fb_' + str(i) for i in range(0,69predict_classes)]
    mfcc_features = ['mfcc_' + str(i) for i in range(0,39)]
 
    # fbank_train = pd.read_csv('./fbank/train.ark', sep=' ', header = None, index_col=0, names = ['id'] + fbank_features)
    mfcc_data = pd.read_csv(mfcc_path, sep=' ', header = None, index_col=0, names = ['id'] + mfcc_features)
    labels = pd.read_csv(labels_path, sep=',', header = None, index_col=0, names = ['id', 'label'])
    df = pd.concat([mfcc_data, labels], axis=1)  
    # map label frome phone48 -> phone39 -> number -> index of number in labelList -> +1
    for key, value in phone39.items():        
        df.loc[df['label'] == key, 'label'] = labelList.index(phone2num[value]) + 1
    
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
        # mfcc = preprocessing.scale(mfcc)
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
    model.add(Bidirectional(GRU(200, return_sequences=True, activation='relu', dropout=0.4), input_shape=input_shape,))
    model.add(Bidirectional(GRU(200, return_sequences=True, activation='relu', dropout=0.4)))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(40, activation='softmax')))
    model.summary()

    opt = keras.optimizers.adam(lr = 0.0001)
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
    rnState = 0
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size= tsize, random_state=rnState)
    
    model = genModel(input_shape)


    batchSize = 100
    epoch = 50
    model.fit(X_train, y_train, batch_size = batchSize,epochs = epoch)  
    model.save('model2.h5') 
    scores = model.evaluate(X_valid, y_valid, verbose=0)
    print(scores)

def loadTestData(mfcc_path):
    phone2num, phone39, labelList  = mapPhones()

    # fbank_features = ['fb_' + str(i) for i in range(0,69predict_classes)]
    mfcc_features = ['mfcc_' + str(i) for i in range(0,39)]
   

 
    # fbank_train = pd.read_csv('./fbank/train.ark', sep=' ', header = None, index_col=0, names = ['id'] + fbank_features)
    df = pd.read_csv(mfcc_path, sep=' ', header = None,  names = ['id'] + mfcc_features)
    # map label frome phone48 -> phone39 -> number -> index of number in labelList -> +1
  
  
    
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
        # mfcc = preprocessing.scale(mfcc)
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
    else:
        X_data = []
        for rows in (df_g):
            
            mfcc = rows[0].as_matrix(mfcc_features)    
            # mfcc = preprocessing.scale(mfcc)
            padding_num = MAX_SEQUENCE_NUM - mfcc.shape[0]
            padding_zeros = np.zeros((padding_num, 39))   
            mfcc = np.concatenate((mfcc, padding_zeros), axis = 0)
            X_data.append(mfcc)
            return X_data, df 

def testPredict():
    X_data, df = loadData('./mfcc/test.ark', None)
    print(X_data)
    print(df)

# def loadModel(path):
#     model = load_model(path)
#     pass


# train()
testPredict()
# model = load_model('models/model2.h5')

# result = model.predict(X_valid)




