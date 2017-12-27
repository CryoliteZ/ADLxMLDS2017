import sys,os,csv, random
import skimage, skimage.io, skimage.transform
import numpy as np
from sklearn.externals import joblib
from keras.utils.vis_utils import plot_model
import skimage, skimage.io, skimage.transform
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Activation
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, Dropout, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import keras.layers.merge as merge
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.utils import shuffle
import time
import scipy
from PIL import Image
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
from keras.utils import to_categorical
   
    
def build_generator():
    # noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    # gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next
    # #gen_input = Input(shape = [noise_shape]) #if want to use with dense layer next
    # con_input = Input(shape = one_hot_vector_shape)
    # inputs = merge([gen_input, con_input], name='concat_input', mode='concat')
    latent_size = 100
    model = Sequential()
    
    model.add(Reshape((1, 1, 100), input_shape=(latent_size,)))
    model.add( Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init, ))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
        
    #model.add( bilinear2x,256,kernel_size=(4,4))
    #model.add( UpSampling2D(size=(2, 2)))
    #model.add( SubPixelUpscaling(scale_factor=2))
    #model.add( Conv2D(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    
    #model.add( bilinear2x,128,kernel_size=(4,4))
    #model.add( UpSampling2D(size=(2, 2)))
    #model.add( SubPixelUpscaling(scale_factor=2))
    #model.add( Conv2D(filters = 128, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    
    #model.add( bilinear2x,64,kernel_size=(4,4))
    #model.add( UpSampling2D(size=(2, 2)))
    #model.add( SubPixelUpscaling(scale_factor=2))
    #model.add( Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))    
    model.add( Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    
    model.add( Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    
    #model.add( bilinear2x,3,kernel_size=(3,3))
    #model.add( UpSampling2D(size=(2, 2)))
    #model.add( SubPixelUpscaling(scale_factor=2))
    #model.add( Conv2D(filters = 3, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Activation('tanh'))
#     model.summary()
        
    # gen_opt = Adam(lr=0.00015, beta_1=0.5)
    # generator_model = Model(input = [gen_input, con_input], output = generator)
    # generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    # generator_model.summary()
    latent_size = 100
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')
    # 10 classes in MNIST
    eyes = Flatten()(Embedding(num_class_eyes, int(latent_size/2),  init='glorot_normal')(eyes_class))
    hairs = Flatten()(Embedding(num_class_hairs, int(latent_size/2),  init='glorot_normal')(hairs_class))
    concat_style = merge([hairs, eyes], name='concat_style', mode='concat')
    h = merge([latent, concat_style], mode='mul')

    fake_image = model(h)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    
    m.summary()
    return m
def gen_noise(batch_size, latent_size):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,latent_size))
def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def generate_images(generator, latent_size, hair_color, eye_color):
    noise = gen_noise(16,latent_size)
    #using noise produced by np.random.uniform - the generator seems to produce same image for ANY noise - 
    #but those images (even though they are the same) are very close to the actual image - experiment with it later.
    
        
    hairs = np.full(16, hair_color, dtype=int)
    eyes = np.full(16,eye_color, dtype=int)
    fake_data_X = generator.predict([noise, hairs, eyes])
   
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,HAIRS[hair_color] +'_' + EYES[eye_color] +'.jpg'),bbox_inches='tight',pad_inches=0)
    # plt.show()
np.random.seed(42)
model_id = sys.argv[1]
gen_dir = sys.argv[2]
if not (os.path.exists(os.path.join("model/",  model_id, gen_dir))):
    os.makedirs(os.path.join("model/",  model_id, gen_dir))
model_dir = os.path.join('model', model_id)
save_dir = os.path.join("model/",  model_id, gen_dir)


HAIRS = [ 'orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = [  'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']
latent_size = 100
num_class_hairs = 12
num_class_eyes = 11
batch_size = 64
half_batch = 64
generator = build_generator()
generator.load_weights('./model/7/53000_GENERATOR_weights_and_arch.hdf5')

for i in range(num_class_hairs):
    for j in range(num_class_eyes):
        generate_images(generator, latent_size, i, j)




