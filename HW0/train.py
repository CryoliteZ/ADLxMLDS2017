import math, csv, random, copy
import numpy as np 
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import gzip, time

def load_mnist(path, kind='train'):   
    """Load MNIST data from `path`"""
    if(kind == 'train'):
        labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
        images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
   
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)
            labels = keras.utils.to_categorical(labels,  num_classes = 10)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 28, 28, 1)

        return images, labels
    else:
        images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
   
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(10000, 28, 28, 1)
            
        return images

def load_mnist_a(path, kind='train'):   
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                            '%s-labels-idx1-ubyte.gz'
                            % kind)
    images_path = os.path.join(path,
                            '%s-images-idx3-ubyte.gz'
                            % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
        labels = keras.utils.to_categorical(labels,  num_classes = 10)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 28, 28, 1)

    return images, labels

def genModelandCompile(X_train, X_valid, y_train, y_valid):
    # print("Create Model")

    model = Sequential() 
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dropout(0.1))
    model.add(Dense(256, activation = 'relu')) 
    model.add(Dense(10, activation = 'softmax'))
    model.summary()
    opt = keras.optimizers.adam(lr = 0.0003)


    # print("Start compile")
    model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['accuracy'])

    batchSize = 300
    epoch = 22
   
    model.fit(X_train, y_train, batch_size = batchSize,epochs = epoch)   
    scores = model.evaluate(X_valid, y_valid, verbose=0)
    model.save(str(int(curTime))+ '_' + str(scores[1]*100)+ 'model.h5')
    return model

X, Y = load_mnist('data', kind='train')
X_test, y_test = load_mnist_a('data', kind='t10k')

curTime = time.time()
tsize = 0.03
rnState = 42
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size= tsize, random_state=rnState)

model = genModelandCompile(X_train, X_valid, y_train, y_valid)
result = model.predict(X_test)
with open(str(curTime) + 'predict.csv' , "w", newline='') as mFile:
    writer = csv.writer(mFile)  
    writer.writerow(['id', 'label'])
    for i in range(0, len(result)):
        mFile.write(str(i) + ",")       
        mFile.write(str(np.argmax(result[i])))
        mFile.write("\n")



