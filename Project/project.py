from numpy.core.records import array
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image

# ############################################################################
# Image Load
path_LDCT = "L506_QD_3_1.CT.0003.0105.2015.12.22.20.45.42.541197.358793241.IMA"
path_NDCT = "L506_FD_3_1.CT.0001.0105.2015.12.22.20.19.39.34094.358586575.IMA"
img_LDCT = pydicom.dcmread(path_LDCT).pixel_array
img_NDCT = pydicom.dcmread(path_NDCT).pixel_array

plt.imsave('NDCT image.jpg',img_NDCT, cmap=plt.cm.gray)
plt.imsave('LDCT image.jpg',img_LDCT, cmap=plt.cm.gray)

# ############################################################################
# Function Definition
def Box(img, size):
    kernel = np.ones((size,size), np.float32) / (size*size)
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def Gauss(img, size, sigma):
    kernel1d = cv2.getGaussianKernel(size, sigma) 
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    low_im_array = cv2.filter2D(img, -1, kernel2d)
    low_array = Image.fromarray(low_im_array)
    return low_array

def Sharp(img, Sharpen_value):
    kernel = np.array([[-1,-1,-1],[-1,Sharpen_value,-1],[-1,-1,-1]])
    Sharpen = cv2.filter2D(img, -1, kernel)
    return Sharpen

def Median(img, filter_size, stride):
    img_shape = np.shape(img)
    result_shape = tuple(np.int64(
        (np.array(img_shape)-np.array(filter_size))/stride+1
    ))
    result = np.zeros(result_shape)
    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            tmp = img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp = np.sort(tmp.ravel())
            result[h,w] = tmp[int(filter_size[0]*filter_size[1]/2)]
    return result

def psnr(ori_img, con_img):
    max_pixel = np.max(con_img)
    mse = np.mean((ori_img - con_img)**2)
    if mse ==0:
        return 100
    psnr = 20* math.log10(max_pixel / math.sqrt(mse))
    return psnr









# ############################################################################
# Perform denoising of "img_LDCT" via (a) box, (b) Gaussian, (c) sharpening, and (d) median filtering
FilterSize = 15
sigma = 5
Sharpen_value = 20
stride = 1

Box_img = Box(img_LDCT,FilterSize)
Gauss_img = Gauss(img_LDCT,FilterSize,sigma)
Sharp_img = Sharp(img_LDCT,Sharpen_value)
Median_img = Median(img_LDCT, (FilterSize, FilterSize), stride)

psnr_l = psnr(img_NDCT,img_LDCT)
psnr_a = psnr(img_NDCT,Box_img)
psnr_b = psnr(img_NDCT,Gauss_img)
psnr_c = psnr(img_NDCT,Sharp_img)
psnr_d = psnr(np.resize(img_NDCT, (512-FilterSize+stride,512-FilterSize+stride)),Median_img)

print(psnr_l, psnr_a, psnr_b, psnr_c, psnr_d)


# ############################################################################
# Display the images

plt.figure(figsize=(16, 8))

plt.subplot(241)
plt.title('LDCT Image')
plt.imshow(img_LDCT, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(242)
plt.title('NDCT Image')
plt.imshow(img_NDCT, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(245)
plt.title('(a) Box filtering')
plt.imshow(Box_img, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(246)
plt.title('(b) Gaussian filtering')
plt.imshow(Gauss_img, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(247)
plt.title('(c) Sharpening filtering')
plt.imshow(Sharp_img, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(248)
plt.title('(d) Median filtering')
plt.imshow(Median_img, cmap=plt.cm.gray)
plt.axis("off")

plt.show()