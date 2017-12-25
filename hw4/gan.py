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

model_id = sys.argv[1]


def face_preprocess(data_path, y_index):   
    faces_dir = os.path.join(data_path,'faces')
    X_data = []
    for idx in y_index:
        filename = os.path.join(faces_dir, (str(idx) + '.jpg'))
        img = Image.open(filename)
        img = img.resize((64,64), Image.ANTIALIAS)
        # img.show()
        x = np.array(img)
        X_data.append(x)
    X_data = np.array(X_data)
    print(X_data.shape)
    with open(os.path.join(data_path,'X_data_filter.jlib' ),'wb') as file:
        joblib.dump(X_data, file)
    
    
def tag_preprocess(data_path):
#     HAIRS = ['hair', 'hairs', 'ponytail', 'tail', 'tails']
#     EYES = ['eye','eyes']
    HAIRS = [ 'orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
    EYES = [  'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    LABELS = [ 'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes','orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
    with open(os.path.join(data_path, 'tags_clean.csv'), 'r') as file:
        lines = csv.reader(file, delimiter=',')
        y_hairs = []
        y_eyes = []
        y_index = []
        for i, line in enumerate(lines):
            y = np.zeros(len(LABELS))
            idx = line[0]
            feats = line[1]
            feats = feats.split('\t')[:-1]
            flag_hair = False
            flag_eyes = False
            y_hair = []
            y_eye = []
            for feat in feats:
                feat = feat[:feat.index(':')]
                if(feat in HAIRS):
                    y_hair.append(HAIRS.index(feat))
                    flag_hair = True
                if(feat in EYES):
                    y_eye.append(EYES.index(feat))
                    flag_eyes = True
            if(flag_hair and flag_eyes):
                hair = random.choice(y_hair)
                eye = random.choice(y_eye)
                y_hairs.append(hair)
                y_eyes.append(eye)
                y_index.append(i)
            
        y_eyes = np.array(y_eyes)
        
            
        # y_eyes = to_categorical(y_eyes)
        y_hairs = np.array(y_hairs)
        # y_hairs = to_categorical(y_hairs)
        y_index = np.array(y_index)
        # print(y_hairs.shape)
        # print(y_eyes.shape)
        # print(y_index.shape)
        return y_hairs, y_eyes, y_index
        '''
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(words)  
        word_index = tokenizer.word_index
        train_sequences = tokenizer.texts_to_sequences(y_tags)
        print(word_index)
        print(train_sequences)
        '''
    pass

def load_data(data_path, y_hairs, y_eyes, y_index):
    with open(os.path.join(data_path, 'X_data_filter.jlib'), 'rb') as file:
        X_data = joblib.load(file)
        return X_data

              



np.random.seed(42)


def get_gen_normal(noise_shape):
    noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next
    #gen_input = Input(shape = [noise_shape]) #if want to use with dense layer next
    
    generator = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init)(gen_input)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
        
    #generator = bilinear2x(generator,256,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,128,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 128, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,64,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)    
    generator = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,3,kernel_size=(3,3))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 3, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Activation('tanh')(generator)
        
    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input = gen_input, output = generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model
    
#------------------------------------------------------------------------------------------

def get_disc_normal(image_shape=(64,64,3)):
    image_shape = image_shape
    
    dropout_prob = 0.4
    
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    dis_input = Input(shape = image_shape)
    
    discriminator = Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    discriminator = Flatten()(discriminator)
    
    #discriminator = MinibatchDiscrimination(100,5)(discriminator)
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)
    
    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input = dis_input, output = discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def gen_noise(batch_size, noise_shape):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

def sample_from_dataset(batch_size, image_shape, X_data, y_hairs, y_eyes):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    choice_indices = np.random.choice(len(X_data), batch_size)
    sample = []
    y_hair_label = []
    y_eyes_label = []
    for i in choice_indices:
        x = X_data[i]
        x = norm_img(x)
        y_hair_label.append(y_hairs[i])
        y_eyes_label.append(y_eyes[i])
        sample.append(x)
    sample = np.array(sample)
    y_hair_label = np.array(y_hair_label)
    y_eyes_label = np.array(y_eyes_label)
    return sample, y_hair_label, y_eyes_label

def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    #print(rand_indices)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    # for i in range(0,5):
    #     img = img_batch[i]
    #     img = denorm_img(img)
    #     scipy.misc.imsave(os.path.join('img', str(i) + '.png'), img)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    # plt.show()

if not (os.path.exists("model/" + model_id)):
    os.makedirs("model/" + model_id)

model_dir = os.path.join('model', model_id)


# np.random.seed(42)
num_steps = 10000
noise_shape = (1, 1, 100)
latent_size = 100
num_class_hairs = 12
num_class_eyes = 11
batch_size = 64
half_batch = 64
image_shape = (64,64,3)
img_save_dir = 'img'
save_model_dir = 'model'
log_dir = 'log'




y_hairs, y_eyes, y_index = tag_preprocess('data')
face_preprocess('data', y_index)
X_data = load_data('data', y_hairs, y_eyes, y_index )
# img = Image.fromarray(X_data[0], 'RGB')
# img.show()


print('X_data: {}, y_hairs: {},  y_eyes"{}'.format(X_data.shape, y_hairs.shape,y_eyes.shape ))
print(y_hairs[0], y_eyes[0], y_index[0])

generator = get_gen_normal(noise_shape)
discriminator = get_disc_normal()

discriminator.trainable = False

# plot_model(generator, to_file='generator_new.png', show_shapes=True)
# plot_model(discriminator, to_file='discriminator_new.png', show_shapes=True)

latent_size = 100
opt = Adam(lr=0.00015, beta_1=0.5) #same as gen
gen_inp = Input(shape=noise_shape)
GAN_inp = generator(gen_inp)
GAN_opt = discriminator(GAN_inp)
gan = Model(input = gen_inp, output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()


avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)

for step in range(num_steps): 
    tot_step = step

    step_begin_time = time.time() 
    # ---------------------
    #  Train Discriminator
    # ---------------------
    real_data_X, real_label_hairs,  real_label_eyes = sample_from_dataset(half_batch, image_shape, X_data, y_hairs, y_eyes)

  

    noise = gen_noise(half_batch,noise_shape)

    # sampled
  

    fake_data_X = generator.predict(noise)
    
    if (tot_step % 50) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_X,os.path.join(model_dir,step_num+"_image.png"))

    
    data_X = np.concatenate([real_data_X,fake_data_X])
    
    # valid
    real_data_Y = np.ones(half_batch) - np.random.random_sample(half_batch)*0.2
    fake_data_Y = np.random.random_sample(half_batch)*0.2
    
    # labels
    # real_label_hair
    # real_label_eye

   
    data_Y = np.concatenate((real_data_Y,fake_data_Y))
    
    
 
    discriminator.trainable = True
    generator.trainable = False
    
    dis_metrics_real = discriminator.train_on_batch(real_data_X,real_data_Y)   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch(fake_data_X,fake_data_Y)   #training seperately on fake
    
    
    
    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])
    
    # ---------------------
    #  Train Generator
    # ---------------------

    generator.trainable = True
    noise = gen_noise(half_batch,noise_shape)
   
 
    GAN_X = gen_noise(half_batch,noise_shape)
    GAN_Y = real_data_Y
    
    discriminator.trainable = False
    
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
   
    
    text_file = open(os.path.join(model_dir,"training_log.txt"), "a")
    if(tot_step % 100 == 0):
        print("Begin step: ", tot_step)
        print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
        
        print("GAN loss: %f" % (gan_metrics[0]))
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
    
        
    # end_time = time.time()
    # diff_time = int(end_time - step_begin_time)
    # print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))
    
    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        discriminator.trainable = True
        generator.trainable = True
        generator.save(os.path.join(model_dir,str(tot_step)+"_GENERATOR_weights_and_arch.hdf5"))
discriminator.save(os.path.join(model_dir,str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5"))

