from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon

filename = "./data/test.png"

def to_sinogram(image, theta):
    #TODO radon
    sinogram = radon(image, theta=theta, circle=True) 
    return sinogram

def from_sinogram(image, theta):
    reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
    error = reconstruction_fbp - image
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))
    return reconstruction_fbp

def print_from_image(image, sinogram):
    
    image = rescale(image, scale=0.4, mode='reflect')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap='gray')
    
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap='gray',
            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()
    
def print_from_sinogram(reconstruction_fbp, theta):
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap='gray')
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(reconstruction_fbp - image, cmap='gray', **imkwargs)
    plt.show()

if __name__ == '__main__':
    image = imread(filename, as_grey=True)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = to_sinogram(image, theta)
    print_from_image(image, sinogram)
    
    reconstruction = from_sinogram(sinogram, theta)
    print_from_sinogram(reconstruction, theta)