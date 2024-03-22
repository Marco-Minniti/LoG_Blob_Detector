import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

img = cv2.imread('images/horse205.png', 0)
img1 = cv2.imread('images/horse205.png')
img = img/255


def sigmasAndFilters(number_scales,k,initial_sigma):
    sigmas = []
    for i in range(number_scales):
        sigma_i = initial_sigma*(k**i)
        sigmas.append(sigma_i)
    #creating the filter size for each sigmas
    filters_dim = [round(i*6) for i in sigmas]
    for i in range(len(filters_dim)):
        if filters_dim[i]%2 == 0: # if the size of the filter is even, then make it odd, useful for convolution
            filters_dim[i] = filters_dim[i]+1 
    return sigmas,filters_dim


def laplacianOfGaussian(k,sigmas,number_scales,filters_dim):
    LoG = [] # array that contains the LoG filters for each scale
    for i in range(number_scales):
        LoG_i = np.zeros((filters_dim[i],filters_dim[i])) # for each scale, create an empty LoG filter of size h[i] x h[i] where h[i] is the filter size for scale i
        # Defining the range of values that x and y can take when calculating the values of the LoG filter
        initial = int(-(math.floor(filters_dim[i]/2)))
        end = int(math.floor(filters_dim[i]/2))
        for x in range(initial,end+1):
            for y in range(initial,end+1):
                LoG_i[x+end,y+end] = (-1/(math.pi*(sigmas[i]**2)))*(1-(((x**2)+(y**2))/(2*(sigmas[i]**2))))*np.exp(-((x**2)+(y**2))/(2*(sigmas[i]**2)))
        LoG.append(LoG_i)
    return LoG


def convolution(image, LoG, number_scales, filters_dim):
    row = image.shape[0]
    col = image.shape[1]
    conv_imgs = []
    for i in range(number_scales):
        LoG[i] = np.flipud(np.fliplr(LoG[i])) # flip the LoG filter in both vertical and horizontal directions, useful for convolution
        matrix = np.zeros_like(image) 
        s = int(math.floor(filters_dim[i]/2))+1 # size of the padding that is added to the image before convolution
        padded = np.zeros((row + filters_dim[i]+1, col + filters_dim[i]+1))   
        padded[s:-s, s:-s] = image
        # Convolution
        for x in range(row): 
            for y in range(col):
                matrix[x,y]=(LoG[i]*padded[x:x+filters_dim[i],y:y+filters_dim[i]]).sum()  
        conv_imgs.append(matrix)
    return conv_imgs 


def find_Maxima(img,conv_imgs,sigmas,number_scales,threshold):
    row = img.shape[0]
    col = img.shape[1]
    blobs = [] # array that contains the location of the blobs in the image for each scale
    # For each scale, scrolls through each pixel in the convolved image. 
    # For each pixel, it checks whether the pixel's value is a local maximum by comparing it with the values of neighboring pixels in a 3x3x3 surround (which includes the current scale, the previous scale, and the next scale). 
    # If the pixel value is greater than the threshold and is a local maximum, then it is considered a blob.
    for n in range(number_scales):
        blob = []
        blob_radius = int(math.sqrt(2)*sigmas[n]) # radius of the circle that is drawn around the blob proportional to the sigma
        for x in range(row):
            for y in range(col):
                flag = 'min'
                for u in range(-1,2):
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if x+i >= 0 and y+j >= 0 and x+i < row and y+j < col and n+u >= 0 and n+u < number_scales:
                                if conv_imgs[n][x,y] > threshold:
                                    if conv_imgs[n][x,y] < conv_imgs[n+u][x+i,y+j]: flag = 'not min'   
                                else: flag = 'not min'
                if flag == 'min':
                    if x-blob_radius > 0 and y-blob_radius > 0 and x+blob_radius < row-1 and y+blob_radius < col-1:
                        blob.append([x,y])
                        #cv2.circle(img, (y, x), blob_radius, (0,0,255), 1)
                        cv2.circle(img, (y, x), blob_radius, (0,0,255), 2)
        blobs.append(blob)
    return img



number_scales = 5 # smaller scale will detect small details (small blobs), while a larger scale will detect larger details, used to generate several LoG filters_dim
k = 2 # scale factor
initial_sigma = 2 
threshold = 0.1

sigmas, filters_dim = sigmasAndFilters(number_scales,k,initial_sigma)
LoG = laplacianOfGaussian(k,sigmas,number_scales,filters_dim)
conv_imgs = convolution(img,LoG,number_scales,filters_dim)
img1 = find_Maxima(img1,conv_imgs,sigmas,number_scales,threshold)

plt.imshow(img1)
plt.show()
cv2.imwrite('results/out.png',img1)


