# Despeckling Polarimetric SAR Data Using a Multistream Complex-Valued Fully Convolutional Network

Abstract
A polarimetric synthetic aperture radar (PolSAR) sensor is able to collect images in different polarization states, making it a rich source of information for target characterization. PolSAR images are inherently affected by speckle. Therefore, before deriving ad hoc products from the data, the polarimetric covariance matrix needs to be estimated by reducing speckle. In recent years, deep learning-based despeckling methods have started to evolve from single-channel SAR images to PolSAR images. To this aim, deep learning-based approaches separate the real and imaginary components of the complex-valued covariance matrix and use them as independent channels in standard convolutional neural networks (CNNs). However, this approach neglects the mathematical relationship that exists between the real and imaginary components, resulting in suboptimal output. Here, we propose a multistream complex-valued fully convolutional network (FCN) (CV-deSpeckNet1) to reduce speckle and effectively estimate the PolSAR covariance matrix. To evaluate the performance of CV-deSpeckNet, we used Sentinel-1 dual polarimetric SAR images to compare against its real-valued counterpart that separates the real and imaginary parts of the complex covariance matrix. CV-deSpeckNet was also compared against the state of the art PolSAR despeckling methods. The results show that CV-deSpeckNet was able to be trained with a fewer number of samples, has a higher generalization capability, and resulted in higher accuracy than its real-valued counterpart and state-of-the-art PolSAR despeckling methods. These results showcase the potential of complex-valued deep learning for PolSAR despeckling.

The pre-print for the paper describing cv-despecknet can be found here (https://arxiv.org/abs/2103.07394)

In this implementation, both the input noisy images and their reference images should be in log form. Therefore, in this implementation, the noisy image reconstruction doesnot need to be converted to the linear scale by taking the matrix exponent, This is the only difference between this implementation and the one used in the paper. 

![paper6_flowchart2](https://user-images.githubusercontent.com/48068921/112758977-4906ba00-8ff1-11eb-8e08-ce3cab3aaad7.png)

To run the scripts locally, you need to install the keras-complex library (https://pypi.org/project/keras-complex/) along with keras==2.23 and tensorflow-gpu==1.13.1. There is a requirements text file included. For usage in google colab the folder complexnn (supplied in this repo) should be uploaded to your google drive. 

The keras-complex library used in this paper and its documentation can be found in https://github.com/JesperDramsch/keras-complex

If you use this implementation please cite our work as follows

A. G. Mullissa, C. Persello and J. Reiche, (2021).   
Despeckling Polarimetric SAR Data Using a Multistream Complex-Valued Fully Convolutional Network  
IEEE Geoscience and Remote Sensing Letters, 1-5, doi:10.1109/LGRS.2021.3066311.

