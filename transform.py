import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import skimage.morphology as mp

from skimage import filters
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage import data_dir
from scipy import misc
from skimage.color import rgb2gray
from math import floor
from sklearn.metrics import mean_squared_error

filename = "./data/shepp_logan2.png"
width = 90
alpha = 2
detector_amount = 180

def bresenham_line(x1, y1, x2, y2):
    # zmienne pomocnicze
    d = dx = dy = ai = bi = 0
    xi = yi = 0
    x = x1
    y = y1
    line = []

    # ustalenie kierunku rysowania
    if (x1 < x2):
        xi = 1
        dx = x2 - x1
    else:
        xi = -1;
        dx = x1 - x2;

    # ustalenie kierunku rysowania
    if (y1 < y2):
        yi = 1
        dy = y2 - y1
    else:
        yi = -1
        dy = y1 - y2

    # pierwszy piksel
    line.append([x, y])
    # oś wiodąca OX
    if (dx > dy):
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        # pętla po kolejnych x
        while (x != x2):
            # test współczynnika
            if (d >= 0):
                x = x + xi
                y = y + yi
                d = d + ai
            else:
                d = d + bi
                x = x + xi
            line.append([x, y])
    # oś wiodąca OY
    else:
        ai = ( dx - dy ) * 2
        bi = dx * 2
        d = bi - dy
        # pętla po kolejnych y
        while (y != y2):
            #test współczynnika
            if (d >= 0):
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            line.append([x, y])

    return line

def save_snapshot(sinogram, directory):
    save_snapshot.counter += 1
    fig, plots = plt.subplots(1,1)
    plots.imshow(sinogram, cmap='gray')
    plt.savefig(directory + "/snapshot_sinogram_" + str(save_snapshot.counter) + ".png")

save_snapshot.counter = 0  

def coordinates(radius, detector, detector_amount, iteration, picture_size):
    x1 = radius * np.cos(iteration * np.pi / 180)
    y1 = radius * np.sin(iteration * np.pi / 180)                                    #ilosc detektorów
    x2 = radius * np.cos((iteration + 180 - (width / 2) + detector * (width / (detector_amount))) * np.pi / 180)
    y2 = radius * np.sin((iteration + 180 - (width / 2) + detector * (width / (detector_amount))) * np.pi / 180)
    x1 = int(x1) + np.floor(picture_size / 2)
    y1 = int(y1) + np.floor(picture_size / 2)
    x2 = int(x2) + np.floor(picture_size / 2)
    y2 = int(y2) + np.floor(picture_size / 2)
    return x1, y1, x2, y2
    
def get_normalised_pixel(image, line):
    count = int(0)
    average = np.float(0)
    value = np.float(0)
    for pos in line:
        if pos[0]>=0 and pos[1]>=0 and pos[0]<len(image) and pos[1]<len(image):
            value += float(image[int(pos[0]), int(pos[1])])
            count += 1
    average = value / count
    return average

def radon_transform(image):
    picture_size = len(image[0])
    radius = int(np.ceil(picture_size))

    sinogram = []   #stores next scanned lines
    lines = []      #stores coordinates

    for iteration in range(0, 360, alpha):
        sinogram.append([])
        lines.append([])
        for detector in range(0, detector_amount):
            #determination of the emitter/detectors coordinates
            x1, y1, x2, y2 = coordinates(radius, detector, detector_amount, iteration, picture_size)
            #assignation of pixels
            line = bresenham_line(x1, y1, x2, y2)
            #normalization of the pixel
            average = get_normalised_pixel(image, line)            
            #save results                         
            sinogram[-1].append(average)
            lines[-1].append([x1, y1, x2, y2])
        if(iteration % 30 == 0):
            print("Sinogram", save_snapshot.counter)    
            save_snapshot(sinogram, "out_sin")
    return sinogram, lines
    
#*****************REVERSE RADON***********************
def filtering_picture(img) :
    new_img = filters.gaussian(img, sigma=1)
    new_img = mp.dilation(mp.erosion(new_img))
    return new_img

def normalizing_picture(reconstructed, helper):
    normalized = np.copy(reconstructed)
    picture_shape = np.shape(normalized)
    width = picture_shape[0]
    height = picture_shape[1]

    for i in range (0, width, 1):
        for j in range (0, height, 1):
            if helper[i][j] != 0:
                normalized[i][j] = normalized[i][j]/helper[i][j]
    return normalized

def make_mask(detectors):
    mask_size = floor(detectors/2)
    mask = np.zeros(mask_size)
    center = floor(mask_size/2)
    for i in range(0, mask_size, 1):
        k = i - center
        if k % 2 != 0:
            mask[i] = (-4/np.pi**2)/k**2
    mask[center] = 1
    return mask

def filtering_sinogram(sinogram): # maska na sinogram convolve, http://www.dspguide.com/ch25/5.htm
    sinogram_shape = np.shape(sinogram)
    number_of_projections = sinogram_shape[0]
    number_of_detectors = sinogram_shape[1]
    filtered = np.zeros((number_of_projections, number_of_detectors))
    mask = make_mask(number_of_detectors)
    for projection in range (0, number_of_projections, 1):
        filtered[projection] = sig.convolve(sinogram[projection], mask, mode = 'same', method='direct')
    return filtered

def reverse_radon(image, sinogram_org, lines):
    # wymiary zdjęcia końcowego
    picture_shape = np.shape(image)
    width = picture_shape[0]
    height = picture_shape[1]
    # dane o projekcjach i detektorach
    sinogram_shape = np.shape(sinogram_org)
    number_of_projections = sinogram_shape[0]
    number_of_detectors = sinogram_shape[1]
    # dane do rekonstrukcji zdjęcia
    reconstructed = np.zeros(shape = picture_shape)
    reconstructed_nofilterd = np.zeros(shape = picture_shape)
    helper = np.zeros(shape = picture_shape)

    sinogram = filtering_sinogram(sinogram_org)

    # rekonstrukcja zdjęcia
    for projection in range (0, number_of_projections, 1):
        for detector in range (0, number_of_detectors, 1):
            x1, y1, x2, y2 = lines[projection][detector]
            line = bresenham_line(x1, y1, x2, y2)
            value = sinogram[projection][detector]
            value_nof = sinogram_org[projection][detector]
            for i in range (0, len(line), 1):
                    x, y = line[i]
                    if x >= 0 and y >= 0 and x < width and y < height:
                        reconstructed_nofilterd[int(x)][int(y)] += value_nof
                        reconstructed[int(x)][int(y)] += value
                        helper[int(x)][int(y)] += 1

        fragment = normalizing_picture(reconstructed, helper)
        fragment[fragment[:,:] < 0] = 0
        fragment = rescale_intensity(fragment)
        reconstructed2 = filtering_picture(fragment)

        if(projection % 30 == 0):
            print("Reconstructed", save_snapshot.counter)
            save_snapshot(reconstructed2, "out_rec")
            
    fragment = normalizing_picture(reconstructed, helper)
    fragment[fragment[:,:] < 0] = 0
    fragment = rescale_intensity(fragment)
    #reconstructed = filtering_picture(fragment)
    reconstructed = fragment
    save_snapshot(reconstructed, "out_fin")
    
    fragment = normalizing_picture(reconstructed_nofilterd, helper)
    fragment[fragment[:,:] < 0] = 0
    fragment = rescale_intensity(fragment)
    #reconstructed_nofilterd = filtering_picture(fragment)
    reconstructed_nofilterd = fragment
    save_snapshot(reconstructed_nofilterd, "out_fin")
    
    return reconstructed, reconstructed_nofilterd
    
    
def read_image(filename):
    #image = imread(filename, as_grey=True)
    image = np.zeros([200, 200])
    image[24:174, 24:174] = rgb2gray(imread(filename))
    return(image)

def mean_squared(picture, reconstructed, reconstructed_nofilterd):
    print("Mean squared error with filtering: ", mean_squared_error(picture, reconstructed))
    print("Mean squared error without filtering: ", mean_squared_error(picture, reconstructed_nofilterd))

if __name__ == '__main__':

    org_image = read_image(filename)
    
    sinogram, lines = radon_transform(org_image)
    
    reconst_image, reconstructed_nofilterd = reverse_radon(org_image, sinogram, lines)
    
    mean_squared(org_image, reconst_image, reconstructed_nofilterd)
    
