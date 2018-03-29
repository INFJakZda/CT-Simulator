import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from scipy import misc
from skimage.color import rgb2gray

filename = "./data/test.png"
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

def save_snapshot(sinogram):
    save_snapshot.counter += 1
    fig, plots = plt.subplots(1,1)
    plots.imshow(sinogram, cmap='gray')
    plt.savefig("out_sin/snapshot_sinogram_" + str(save_snapshot.counter) + ".png")

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
    return count, average, value

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
            count, average, value = get_normalised_pixel(image, line)            
            #save results                         
            sinogram[-1].append(average)
            lines[-1].append([x1, y1, x2, y2])        
        save_snapshot(sinogram)
    return sinogram, lines
    

def reverse_radon(image, sinogram, lines):
    return 0
    
def read_image(filename):
    #image = imread(filename, as_grey=True)
    image = np.zeros([200, 200])
    image[24:174, 24:174] = rgb2gray(imread(filename))
    return(image)

if __name__ == '__main__':

    org_image = read_image(filename)
    
    sinogram, lines = radon_transform(org_image)
    
    reconstruction = reverse_radon(org_image, sinogram, lines)
