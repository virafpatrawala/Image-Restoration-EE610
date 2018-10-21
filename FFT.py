
# coding: utf-8

# In[89]:


#Importing the required libraries
import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt

#########################
# 2 Dimensional FFT has been implemented using 1 Dimensional
# FFT on rows followed by columns and then compared to a standard
# library implementation.
###########################

#Simple 1 Dimensional DFT Implementation
def DFT(x):
    x = np.asarray(x, dtype=np.complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    expo = np.exp(-2j*np.pi*k*n/N) 
    return np.dot(expo, x) #Vectorized implementation

#Recursive 1 Dimensional FFT implementation
def FFT(x):
    x = np.asarray(x, dtype=np.complex)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  #End of recursion: this cutoff should be optimized
        return DFT(x)
    else:
        #Split the Transform into odd and even
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        n=np.arange(N)
        factor = np.exp(-2j*np.pi*n/N)
        return np.concatenate([X_even + factor[:int(N/2)]*X_odd, X_even + factor[int(N/2):]*X_odd])
    
#Vectorized 1 Dimensional FFT implementation
def FFT_vectorized(x):
    x = np.asarray(x, dtype=np.complex)
    N = x.shape[0]
    
    if(np.log2(N)%1 > 0):
        raise ValueError("size of x must be a power of 2")

    N_min = min(N, 32) #Decides the end of recursion: this cutoff should be optimized
    
    # Perform DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n.reshape((N_min,1))
    expo = np.exp(-2j*np.pi*n*k/N_min)
    X = np.dot(expo, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while(X.shape[0]<N):
        X_even = X[:,:int(X.shape[1]/2)]
        X_odd = X[:,int(X.shape[1]/2):]
        factor = np.exp(-1j*np.pi*np.arange(X.shape[0])/X.shape[0])[:, None]
        X = np.vstack([X_even + factor*X_odd, X_even - factor*X_odd])
    return X.ravel()


# In[90]:


def TwoDimensionalFFT(image):
    image = np.asarray(image, dtype=np.complex)
    for row in range(image.shape[0]):
        image[row,:] = FFT_vectorized(image[row,:])
    for column in range(image.shape[1]):
        image[:,column] = FFT_vectorized(image[:,column])
    return image


# In[91]:


image = plt.imread('Kernel1.png')[:,:,0]     # flatten=True gives a greyscale image
image = np.pad(image, ((0,11),(0,11)), 'constant', constant_values=0)

fft2 = (np.fft.fft2(image))
fft2_i = TwoDimensionalFFT(image)

plt.imshow(20*np.log10(abs(fft2)))
plt.show()

plt.imshow(20*np.log10(abs(fft2_i)))
plt.show()


# In[103]:


N_min=8
x=np.arange(16)
n = np.arange(N_min)
k = n.reshape((N_min,1))
expo = np.exp(-2j*np.pi*n*k/N_min)
X = np.dot(expo, x.reshape((N_min, -1)))


# In[105]:




