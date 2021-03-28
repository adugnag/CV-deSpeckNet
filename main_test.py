"""
Version: v1.2
Date: 2021-01-12
Author: Mullissa A.G.
Description: This script tests a pre-trained cv-despecknet model on  a properly formatted polarimetric SAR covariance matrix
"""

# run this to test the model
import complexnn
import argparse
import os, time
import numpy as np
from keras.models import  load_model
import keras.backend as K
from tifffile import imread, imwrite

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='Train', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=[''], type=list, help='name of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('models','cv_despecknet'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_050.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return img[np.newaxis,...,]

def from_tensor(img):
        return np.squeeze(img[0,...])

    
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2


if __name__ == '__main__':        
    args = parse_args()

model = load_model(os.path.join(args.model_dir, args.model_name), custom_objects={'ComplexConv2D': complexnn.conv.ComplexConv2D, 'ComplexBatchNormalization': complexnn.bn.ComplexBatchNormalization, 'sum_squared_error': sum_squared_error})

if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        
for set_cur in args.set_names:  
        
        if not os.path.exists(os.path.join(args.result_dir,set_cur)):
            os.mkdir(os.path.join(args.result_dir,set_cur))
        
        for im in os.listdir(os.path.join(args.set_dir,set_cur)): 
            if im.endswith(".tif") :
                y = np.array(imread(os.path.join(args.set_dir,set_cur,im)), dtype=np.float32)
                np.random.seed(seed=0) # for reproducibility
                y = y.astype(np.float32)
                y_  = to_tensor(y)
                start_time = time.time()
                x_ = model.predict(y_) # filter
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_=from_tensor(x_[0])
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    imwrite(os.path.join(args.result_dir,set_cur,name+'_cv-despecknet'+ext) , x_, planarconfig='CONTIG')
        
        


