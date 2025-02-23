def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

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
  


    
   
