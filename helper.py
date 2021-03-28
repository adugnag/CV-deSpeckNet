"""
Version: v1.2
Date: 2021-01-12
Author: Mullissa A.
Description: This script contains helper functions that are used in synthesizing data tensors.
"""


import glob
import os
import cv2
import numpy as np
import tifffile


#from multiprocessing import Pool


patch_size, stride = 40, 9
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def gen_patches(file_name):

    # read image
    img = tifffile.imread(file_name) #comment out when using RGB imaes
    img = np.array(img)
    h, w, d = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size,:]
                patches.append(x)        
                
    return patches

def make_dataTensor(data_dir,verbose=False):
    
    file_list = glob.glob(data_dir+'/*.tif')  # get name list of all .tif files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')
    data = np.array(data)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],6))
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data

def get_steps(data_dir, batch_size=128):
    if os.path.isfile(data_dir):
        noisy_files = [data_dir]
    else:
        noisy_files = glob.glob(data_dir + '/*.tif')
    num = 0
    #get number of steps per epoch to use in training
    for data_file in noisy_files:
        xs = make_dataTensor(data_dir)
        if xs is not None: 
            num += len(xs)
    print("total number of patches: {}".format(num))
    print("steps per epoch: {}".format(num//batch_size))
    print("")
    return num // batch_size

if __name__ == '__main__':   

    data = make_dataTensor(data_dir='data/Train2')
    label = make_dataTensor(data_dir='data/Label2')
