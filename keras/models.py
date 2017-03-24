import theano.tensor as T

import numpy as np
np.random.seed(1337) # for reproducibility
import pdb
from metrics import Metrics

from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Input, merge
from keras.layers import Activation, Dropout, Flatten, Dense, Permute, Reshape, SeparableConv2D
from keras.regularizers import l2, activity_l2
from cityscapes_handler import CityScapesHandler
from keras import backend as K
from xception import Xception



class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]

def get_xception(input_width, input_height, nclasses):
    pdb.set_trace()
    model= Xception(include_top= False, weights= None, input_shape=(input_width,input_height,3), classes= nclasses)
    return model

def get_dilation_frontend(input_width, input_height, nclasses):
    model = models.Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(input_width, input_height, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6'))
    model.add(BatchNormalization())
    # TODO: dropout for training
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    model.add(BatchNormalization())
    # TODO: dropout for training
    # Note: this layer has linear activations, not ReLU
    model.add(Convolution2D(nclasses, 1, 1, name='fc-final'))
    model.add(BatchNormalization())


    curr_width, curr_height, curr_channels = model.layers[-1].output_shape[1:]
    model.add(Reshape((curr_width*curr_height, curr_channels)))
    model.add(Activation('softmax'))
    pdb.set_trace()
    return model



def get_dilation(input_width, input_height, nclasses):
    model = models.Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(input_width, input_height, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same'))
    model.add(BatchNormalization())

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3'))
    model.add(BatchNormalization())

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6'))
    model.add(BatchNormalization())
    # TODO: dropout for training
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    model.add(BatchNormalization())
    # TODO: dropout for training
    # Note: this layer has linear activations, not ReLU
    model.add(Convolution2D(nclasses, 1, 1, name='fc-final'))
    model.add(BatchNormalization())

    # Context module
    model.add(ZeroPadding2D(padding=(33, 33)))
    model.add(Convolution2D(nclasses*2, 3, 3, activation='relu', name='ct_conv1_1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(nclasses*2, 3, 3, activation='relu', name='ct_conv1_2'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(nclasses*4, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(nclasses*8, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(nclasses*32, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1'))
    model.add(BatchNormalization())
    model.add(AtrousConvolution2D(nclasses*32, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(nclasses*32, 3, 3, activation='relu', name='ct_fc1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(nclasses, 1, 1, name='ct_final'))

    # The softmax layer doesn't work on the (width, height, channel)
    # shape, so we reshape to (width*height, channel) first.
    # https://github.com/fchollet/keras/issues/1169
    curr_width, curr_height, curr_channels = model.layers[-1].output_shape[1:]
    model.add(Reshape((curr_width*curr_height, curr_channels)))
    model.add(Activation('softmax'))
    return model

#def get_fcn8s(input_width, input_height, nclasses):
#    inputs = Input((input_width, input_height, 3))
#    paded= ZeroPadding2D((100,100)(inputs)
#    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(paded)
#    conv1 = Convolution2D(base_nfilters, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



def get_segnet(data_shape, nclasses):
    autoencoder = models.Sequential()
    autoencoder.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(360, 480, 3), border_mode='same'))
    ##autoencoder.add(GaussianNoise(sigma=0.3))
    autoencoder.encoding_layers = create_encoding_layers()
    autoencoder.decoding_layers = create_decoding_layers()
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Convolution2D(12, 1, 1, border_mode='valid',))
    autoencoder.add(Reshape((data_shape, nclasses), input_shape=(360,480, 12)))
#    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    #from keras.optimizers import SGD
    #optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)

    #autoencoder= get_model(360,480)
    return autoencoder

def get_unet(input_width, input_height, nclasses):
    inputs = Input((input_width, input_height, 3))
    base_nfilters=64
    conv1 = Convolution2D(base_nfilters, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(inputs)
    conv1= BatchNormalization()(conv1)
    conv1 = Convolution2D(base_nfilters, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv1)
    conv1= BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(base_nfilters*2, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(pool1)
    conv2= BatchNormalization()(conv2)
    conv2 = Convolution2D(base_nfilters*2, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv2)
    conv2= BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(base_nfilters*4, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(pool2)
    conv3= BatchNormalization()(conv3)
    conv3 = Convolution2D(base_nfilters*4, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv3)
    conv3= BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(base_nfilters*8, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(pool3)
    conv4= BatchNormalization()(conv4)
    conv4 = Convolution2D(base_nfilters*8, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv4)
    conv4= BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(base_nfilters*16, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(pool4)
    conv5= BatchNormalization()(conv5)
    conv5 = Convolution2D(base_nfilters*16, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv5)
    conv5= BatchNormalization()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(base_nfilters*8, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(up6)
    conv6= BatchNormalization()(conv6)
    conv6 = Convolution2D(base_nfilters*8, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv6)
    conv6= BatchNormalization()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(base_nfilters*4, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(up7)
    conv7= BatchNormalization()(conv7)
    conv7 = Convolution2D(base_nfilters*4, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv7)
    conv7= BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(base_nfilters*2, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(up8)
    conv8= BatchNormalization()(conv8)
    conv8 = Convolution2D(base_nfilters*2, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv8)
    conv8= BatchNormalization()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(base_nfilters, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(up9)
    conv9= BatchNormalization()(conv9)
    conv9 = Convolution2D(base_nfilters, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1e-4))(conv9)
    conv9= BatchNormalization()(conv9)

    conv10 = Convolution2D(nclasses, 1, 1, border_mode='valid', W_regularizer=l2(1e-4))(conv9)
    res10 = Reshape((input_width*input_height, nclasses))(conv10)
    act10= Activation('softmax')(res10)

    model = Model(input=inputs, output=act10)

    return model


