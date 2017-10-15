import math, csv, random, copy
import numpy as np 
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import gzip
import time

def load_mnist(path, kind='train'):   
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
                            offset=16).reshape(len(labels), 784)

    return images, labels

X_test1, Y_test1 = load_mnist('./fashion-mnist/data/fashion', kind='t10k')
X_test2, Y_test2 = load_mnist('data', kind='t10k')

for i in range(0, 2):
    print(i)
    print(X_test1[i])
    print('----\n')
    print(X_test2[i])
    print('----\n')
     
