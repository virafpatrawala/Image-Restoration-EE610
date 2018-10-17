
# coding: utf-8

# In[4]:


import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from scipy import fftpack
import cv2
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from __future__ import print_function
from scipy import signal
import tensorflow as tf
from skimage.measure import compare_ssim as ssim_function
from skimage.measure import compare_psnr as psnr_function


# In[5]:


def inverse_filtering(img1, kernel, R, ground_truth):
    img = img1/255.0 #Scaling image to [0,1]
    kernel = kernel/255.0 #Scaling kernel to [0,1]
    
    #Padding kernel to the size of the image
    kernel = np.pad(kernel, ((0,img.shape[0] - kernel.shape[0]),(0,img.shape[1] - kernel.shape[1])),
                    'constant', constant_values=kernel.min())
    
    #Taking FFT 
    kernel_fft = np.fft.fftshift(np.fft.fft2(kernel))
    ffti0=np.fft.fft2(img[...,0])
    ffti1=np.fft.fft2(img[...,1])
    ffti2=np.fft.fft2(img[...,2])

    #Defining the deblurring filter
    inv_filter = 1.0/kernel_fft

    #Creating a truncation radius
    for i in range(ffti0.shape[0]):
        for j in range(ffti1.shape[1]):
            if((i-ffti0.shape[0]/2)**2+(j-ffti0.shape[1]/2)**2>=R**2):
                inv_filter[i,j]=0
        
    output1 = np.zeros((img.shape[0],img.shape[1],3))
    output2 = np.zeros((img.shape[0],img.shape[1],3))

    #Multiplying the FFTs and then taking inverse
    output1[...,0] = np.real(np.fft.ifft2(1.0*ffti0*inv_filter))
    output1[...,1] = np.real(np.fft.ifft2(1.0*ffti1*inv_filter))
    output1[...,2] = np.real(np.fft.ifft2(1.0*ffti2*inv_filter))

    #Scaling the output back to the original pixel intensities
    output2[:,:,0]=np.interp(output1[:,:,0], (output1[:,:,0].min(), output1[:,:,0].max()), (img1[:,:,0].min(), img1[:,:,0].max()))
    output2[:,:,1]=np.interp(output1[:,:,1], (output1[:,:,1].min(), output1[:,:,1].max()), (img1[:,:,0].min(), img1[:,:,1].max()))
    output2[:,:,2]=np.interp(output1[:,:,2], (output1[:,:,2].min(), output1[:,:,2].max()), (img1[:,:,0].min(), img1[:,:,2].max()))
    
    output = Image.fromarray(output2.astype('uint8'))
    output.save('output.png')
    
    #Calculating the PSNR
    psnr_1 = psnr_function(ground_truth, output2)
#     print(R, ' PSNR:',psnr_1)
    
    #Calculating the SSIM
    ssim_1 = ssim_function(output2, ground_truth, multichannel=True)
#     print(R, ' SSIM:', ssim_1)
    
    return psnr_1, ssim_1


# In[6]:


def wiener_filtering(img1, kernel, K, ground_truth):
    img = img1/255.0 #Scaling image to [0,1]
    kernel = kernel/255.0 #Scaling kernel to [0,1]
    
    #Padding kernel to the size of the image and taking FFT
    kernel = np.pad(kernel, ((0,img.shape[0] - kernel.shape[0]),(0,img.shape[1] - kernel.shape[1])),'constant', constant_values=kernel.min())
    kernel_fft = np.fft.fftshift(np.fft.fft2(kernel))
    
    #Defining the deblurring filter
    wiener_filter = np.conj(kernel_fft) / (np.abs((kernel_fft)**2) + K)
    
    #Taking FFT of image channels
    ffti0=np.fft.fft2(img[...,0])
    ffti1=np.fft.fft2(img[...,1])
    ffti2=np.fft.fft2(img[...,2])

    output1 = np.zeros((img.shape[0],img.shape[1],3))
    output2 = np.zeros((img.shape[0],img.shape[1],3))

    #Multiplying the FFTs and filter and then taking inverse
    output1[...,0] = np.real(np.fft.ifft2(1.0*ffti0*wiener_filter))
    output1[...,1] = np.real(np.fft.ifft2(1.0*ffti1*wiener_filter))
    output1[...,2] = np.real(np.fft.ifft2(1.0*ffti2*wiener_filter))

    #Scaling the output back to the original pixel intensities
    output2[:,:,0]=np.interp(output1[:,:,0], (output1[:,:,0].min(), output1[:,:,0].max()),(img1[:,:,0].min(), img1[:,:,0].max()))                             
    output2[:,:,1]=np.interp(output1[:,:,1], (output1[:,:,1].min(), output1[:,:,1].max()),(img1[:,:,0].min(), img1[:,:,0].max()))                             
    output2[:,:,2]=np.interp(output1[:,:,2], (output1[:,:,2].min(), output1[:,:,2].max()),(img1[:,:,0].min(), img1[:,:,0].max()))                             

    output = Image.fromarray(output2.astype('uint8'))
    output.save('wiener_output.png')
    
    #Calculating the PSNR
    psnr_1 = psnr_function(ground_truth, output2)
#     print(K, ' PSNR:', psnr_1)
    
    #Calculating the SSIM
    ssim_1 = ssim_function(ground_truth, output2, multichannel=True)
#     print(K, ' SSIM:', ssim_1)
    
    return psnr_1, ssim_1


# In[7]:


def cls_filtering(img1, kernel, gamma, ground_truth):
    img = img1/255.0 #Scaling image to [0,1]
    kernel = kernel/255.0 #Scaling kernel to [0,1]
    
    #Padding kernel to the size of the image and taking FFT
    kernel = np.pad(kernel, ((0,img.shape[0] - kernel.shape[0]),(0,img.shape[1] - kernel.shape[1])),
                    'constant', constant_values=kernel.min())
    kernel_fft = np.fft.fftshift(np.fft.fft2(kernel))
    
    #Defining the deblurring filter
    p1 = np.array(([0, -1, 0],[-1, 4, -1],[0, -1, 0]))
    p1 = np.pad(p1, ((0,img.shape[0] - p1.shape[0]),(0,img.shape[1] - p1.shape[1])),
                    'constant', constant_values=0)
    P = np.fft.fft2(p1)
    cls_filter = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + (gamma*(np.abs(P)**2)))
    
    #Taking FFT of image channels
    ffti0=np.fft.fft2(img[...,0])
    ffti1=np.fft.fft2(img[...,1])
    ffti2=np.fft.fft2(img[...,2])

    output1 = np.zeros((img.shape[0],img.shape[1],3))
    output2 = np.zeros((img.shape[0],img.shape[1],3))

    #Multiplying the FFTs and filter and then taking inverse
    output1[...,0] = np.real(np.fft.ifft2(1.0*ffti0*cls_filter))
    output1[...,1] = np.real(np.fft.ifft2(1.0*ffti1*cls_filter))
    output1[...,2] = np.real(np.fft.ifft2(1.0*ffti2*cls_filter))

    #Scaling the output back to the original pixel intensities
    output2[:,:,0]=np.interp(output1[:,:,0], (output1[:,:,0].min(), output1[:,:,0].max()),
                             (img1[:,:,0].min(), img1[:,:,0].max()))
    output2[:,:,1]=np.interp(output1[:,:,1], (output1[:,:,1].min(), output1[:,:,1].max()),
                             (img1[:,:,0].min(), img1[:,:,1].max()))
    output2[:,:,2]=np.interp(output1[:,:,2], (output1[:,:,2].min(), output1[:,:,2].max()),
                             (img1[:,:,0].min(), img1[:,:,2].max()))

    output = Image.fromarray(output2.astype('uint8'))
    output.save('cls_output.png')
    
    #Calculating the PSNR
    psnr_1 = psnr_function(ground_truth, output2)
#     print(K, ' PSNR:', psnr_1)
    
    #Calculating the SSIM
    ssim_1 = ssim_function(ground_truth, output2, multichannel=True)
#     print(K, ' SSIM:', ssim_1)
    
    return psnr_1, ssim_1


# In[11]:


## Performing an experiment by blurring the Ground Truth image to get a blurred image, 
## then using the filtering methods to unblur it.

blur = Image.open('Kernel1.png')
kernel = np.array(blur)

ground_truth=Image.open('GroundTruth1_1_1.jpg')
img1=np.array(ground_truth)

img = img1/255.0
kernel = kernel/255.0
kernel = np.pad(kernel, ((0,img.shape[0] - kernel.shape[0]),(0,img.shape[1] - kernel.shape[1])),
                'constant', constant_values=kernel.min())
kernel_fft = np.fft.fftshift(np.fft.fft2(kernel))

ffti0=np.fft.fft2(img[...,0])
ffti1=np.fft.fft2(img[...,1])
ffti2=np.fft.fft2(img[...,2])

output1 = np.zeros((img.shape[0],img.shape[1],3))
output2 = np.zeros((img.shape[0],img.shape[1],3))

output1[...,0] = np.real(np.fft.ifft2(1.0*ffti0*kernel_fft))
output1[...,1] = np.real(np.fft.ifft2(1.0*ffti1*kernel_fft))
output1[...,2] = np.real(np.fft.ifft2(1.0*ffti2*kernel_fft))


output2[:,:,0]=np.interp(output1[:,:,0], (output1[:,:,0].min(), output1[:,:,0].max()),
                         (img1[:,:,0].min(), img1[:,:,0].max()))
output2[:,:,1]=np.interp(output1[:,:,1], (output1[:,:,1].min(), output1[:,:,1].max()),
                         (img1[:,:,0].min(), img1[:,:,1].max()))
output2[:,:,2]=np.interp(output1[:,:,2], (output1[:,:,2].min(), output1[:,:,2].max()),
                         (img1[:,:,0].min(), img1[:,:,2].max()))

output = Image.fromarray(output2.astype('uint8'))
output.save('test.png')

ground_truth=Image.open('GroundTruth1_1_1.jpg')
gt=np.array(ground_truth)

blur = Image.open('Kernel1.png')
blr = np.array(blur)

x=Image.open('test.png')
x=np.array(x)
psnr,ssim=inverse_filtering(x, blr, 1000, gt)

