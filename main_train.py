"""
Version: v1.2
Date: 2021-01-12
Author: Mullissa A.G.
Description: This script trains a complex-valued multistream fully convolutional network for despeckling a 
polarimetric SAR covariance matrix as discussed in 
our paper A. G. Mullissa, C. Persello and J. Reiche, 
"Despeckling Polarimetric SAR Data Using a Multistream Complex-Valued Fully Convolutional Network," 
in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2021.3066311. 
Some utility functions are adopted from https://github.com/cszn/DnCNN
"""
# =============================================================================
import complexnn
import helper 
import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input, Add
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import keras.backend as K


## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cv-despecknet', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_image', default='data/Train2', type=str, help='path of train data real')
parser.add_argument('--train_label', default='data/Label2', type=str, help='path of label data')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model) 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def cv_deSpeckNet(depth,filters=48,image_channels=6, use_bnorm=True):
    #FCN noise
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, CV-Conv+Crelu
    layer_count += 1
    x0 = complexnn.conv.ComplexConv2D(filters=filters, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same',name = 'conv'+str(layer_count))(inpt)
    # depth-2 layers, CV-Conv+CV-BN+Crelu
    for i in range(depth-2):
        layer_count += 1
        x0 = complexnn.conv.ComplexConv2D(filters=filters, kernel_size=(3,3), strides=(1,1),activation='relu', padding='same',name = 'conv'+str(layer_count))(x0)
        if use_bnorm:
            layer_count += 1
        x0 = complexnn.bn.ComplexBatchNormalization(name = 'bn'+str(layer_count))(x0)
    # last layer, CV-Conv
    layer_count += 1
    x0 = complexnn.conv.ComplexConv2D(filters=3, kernel_size=(3,3), strides=(1,1),padding='same',name = 'speckle'+str(1))(x0)
    layer_count += 1
    
    #FCN clean
    # 1st layer, CV-Conv+Crelu
    x = complexnn.conv.ComplexConv2D(filters=filters, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same',name = 'conv'+str(layer_count))(inpt)
    # depth-2 layers, CV-Conv+CV-BN+Crelu
    for i in range(depth-2):
        layer_count += 1
        x = complexnn.conv.ComplexConv2D(filters=filters, kernel_size=(3,3), strides=(1,1),activation='relu', padding='same',name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
        x = complexnn.bn.ComplexBatchNormalization(name = 'bn'+str(layer_count))(x)
    # last layer, CV-Conv
    layer_count += 1
    x = complexnn.conv.ComplexConv2D(filters=3, kernel_size=(3,3), strides=(1,1),padding='same',name = 'clean'+str(1))(x)
    layer_count += 1

    x_orig = Add(name = 'noisy' +  str(1))([x0,x])    
    model = Model(inputs=inpt, outputs=[x,x_orig])
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(args,kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),args,kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    #log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=64,data_dir=args.train_image,label_dir=args.train_label):
    while(True):
        n_count = 0
        if n_count == 0:
            #print(n_count)
            xs = helper.make_dataTensor(data_dir)
            xy = helper.make_dataTensor(label_dir)
            assert len(xs)%batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')
            xy = xy.astype('float32')
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                batch_y = xy[indices[i:i+batch_size]]
                yield batch_x, [batch_y, batch_x]
        
# sum square error loss function
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2
    
if __name__ == '__main__':
    # model selection

    model = cv_deSpeckNet(depth=17,filters=48,image_channels=6,use_bnorm=True)
    model.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), custom_objects={'ComplexConv2D': complexnn.conv.ComplexConv2D, 'ComplexBatchNormalization': complexnn.bn.ComplexBatchNormalization, 'sum_squared_error': sum_squared_error})
    
    loss_funcs = {
        'clean1': sum_squared_error,
        'noisy1' : sum_squared_error}
    
    loss_weights = {'clean1': 100.0, 'noisy1': 1.0}
    
    # compile the model
    model.compile(optimizer=Adam(0.001), loss=loss_funcs, loss_weights=loss_weights)
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=1)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # numer of steps per epoch
    nsteps = helper.get_steps(args.train_image, batch_size=64)
    
    history = model.fit_generator(train_datagen(batch_size=64),
                steps_per_epoch=nsteps, epochs=51, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])

















