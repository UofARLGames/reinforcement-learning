from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'
import time
import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
np.random.seed(1337) # for reproducibility
import pdb
from metrics import Metrics
import keras

from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Input, merge
from keras.layers import Activation, Dropout, Flatten, Dense, Permute, Reshape

from cityscapes_handler import CityScapesHandler
from keras import backend as K
from models import get_dilation_frontend, get_dilation, get_segnet, get_unet, get_xception

import cv2
import numpy as np

nclasses= 12
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
def eval(X, Y, model, valid_bs):
    for i in range(X.shape[0]//valid_bs):
        x= X[i*valid_bs:(i+1)*valid_bs,:,:,:]
        y= Y[i*valid_bs:(i+1)*valid_bs,:,:]
        preds= model.predict(x)
        met.update_metrics(np.argmax(preds,axis=2), np.argmax(y,axis=2), 0,0)

    met.compute_final_metrics(1, nonignore= None)
    return met


orig_size= [3,360,480]
#orig_size= [3,628,628]
#orig_size= [3,320, 448]
#autoencoder= get_dilation_frontend(orig_size[1], orig_size[2], 12)
#autoencoder= get_unet(orig_size[1], orig_size[2], 12)
#autoencoder= get_xception(orig_size[1], orig_size[2], 12)
autoencoder= get_segnet(orig_size[1]*orig_size[2], nclasses)
optim= keras.optimizers.Adam()
autoencoder.compile(loss="categorical_crossentropy", optimizer=optim, metrics= ['accuracy'])
current_dir = os.path.dirname(os.path.realpath(__file__))

nb_epoch = 200
BATCH_SIZE =2
VBATCH_SIZE =2
logfile= open('segnet_xception.txt', 'w')
directory= 'camvid_keras_unnorm.h5'
CHD = CityScapesHandler(directory, train_batchsize=BATCH_SIZE, valid_batchsize= VBATCH_SIZE)
valid_every= 1
best_val = np.inf
for epoch in np.arange(0, nb_epoch):
    print('epoch ', epoch)
    logfile.write(str(epoch)+'\n')
    n_batches = 0

    for iter in range(CHD.n_train_samples//BATCH_SIZE):

        data, target = CHD.getMiniBatch('train');
        target = target.astype(np.int32)
        loss, acc= autoencoder.train_on_batch(data.astype(np.float32), target)#, class_weight= class_weighting)
        print ("at iter {}, loss={}, acc={}".format(iter,loss,acc))
        logfile.write("at iter {}, loss={}, acc={}\n".format(iter,loss,acc))

    if epoch % valid_every == 0:
        # since the validation generator does not loop around we need to reinstantiate it for every epoch
        met = Metrics(nclasses)
        for iter in range(CHD.n_valid_samples//VBATCH_SIZE):
            print('validation iteration ',iter)
            data, target = CHD.getMiniBatch('test');
            target = target.astype(np.int32)
            target= np.argmax(target, axis=2)
            first= time.time()
            preds= np.argmax(autoencoder.predict(data), axis=2)
            end= time.time()
            print('Covered time is ', end-first)
            met.update_metrics(preds, target, 0, 0)

        met.compute_final_metrics(CHD.n_valid_samples//VBATCH_SIZE)
        if best_val<met.mean_iou_index:
            best_val= met.mean_iou_index
            autoencoder.save_weights('model_weight_ep100.hdf5')

        print ("cost = %s, precision = %s, recall = %s, fmeasure =%s, MIOU= %s, IOU=%s"%(
               met.cost, met.prec, met.rec, met.fmes, met.mean_iou_index, met.iou))
        logfile.write("cost = %s, precision = %s, recall = %s, fmeasure =%s, MIOU= %s, IOU=%s\n"%(
               met.cost, met.prec, met.rec, met.fmes, met.mean_iou_index, met.iou))
    logfile.flush()

logfile.close()
