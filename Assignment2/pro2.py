from __future__ import print_function
import numpy as np
import os
import itk
import nrrd
import radiomics
from radiomics import featureextractor
from matplotlib import pyplot as plt
import logging
import SimpleITK as sitk
# ################################################################################
# 데이터 셋업

# Global values 및 ndarray 생성
n_train = 222
n_val = 29
n_test = 84
ch = 4
H = 240
W = 240
train_img = np.empty((n_train, ch, H, W))
train_sol = np.empty((n_train, 1))
val_img = np.empty((n_val, ch, H, W))
val_sol = np.empty((n_val, 1))
test_img = np.empty((n_test, ch, H, W))
test_sol = np.empty((n_test, 1))



Data_path = os.getcwd()


# train 데이터 리스트 생성
walking_train = os.path.join(Data_path, 'clf_w_mask', 'train')


i = 0
for path, dirs, files in os.walk(walking_train):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_label = os.path.join(path, 'label.npy')
        img = np.load(path_img)
        label = np.load(path_label)
        nrrd.write('img.nrrd', path_img)
        nrrd.write('label.nrrd', path_label)
        path_train_nrrd_img[i] = os.path.join(path, 'img.nrrd')
        path_train_nrrd_label[i] = os.path.join(path, 'label.nrrd')
        i += 1
 
 





"""
# 참고 코드 던져놓는 공간
for path, dirs, files in os.walk(Projectpath + '/train'):
train_img = np.load(Projectpath + './train/BraTS19_2013_0_1/img.npy')
train_label = np.load(Projectpath + './train/BraTS19_2013_0_1/label.npy')
train_seg = np.load(Projectpath + './train/BraTS19_2013_0_1/seg.npy')


img_ch1 = train_img[0]
img_ch2 = train_img[1]
img_ch3 = train_img[2]
img_ch4 = train_img[3]
img_seg = train_img[0]





img_itk = itk.GetImageFromArray(img_npy)
label_itk = itk.GetImageFromArray(label_npy)

itk.imwrite(img_itk, 'img.nrrd')
itk.imwrite(label_itk, 'label.nrrd')



img_nrrd_path = os.path.join(Data_path, "img.nrrd")
label_nrrd_path = os.path.join(Data_path, "label.nrrd")
print("npy 이미지 차원 : ", np.ndim(img_arr))
print("npy 이미지 모양 : ", np.shape(img_arr))
print("npy 레이블 차원 : ", np.ndim(label_arr))
print("npy 레이블 모양 : ", np.shape(label_arr))
print("nrrd 이미지 차원 : ", np.ndim(img_arr_nrrd))
print("nrrd 이미지 모양 : ", np.shape(img_arr_nrrd))
print("nrrd 레이블 차원 : ", np.ndim(label_arr_nrrd))
print("nrrd 레이블 모양 : ", np.shape(label_arr_nrrd))
print("nrrd 예제 이미지 차원 : ", np.ndim(brain_img_arr_nrrd))
print("nrrd 예제 이미지 모양 : ", np.shape(brain_img_arr_nrrd))
print("nrrd 예제 레이블 차원 : ", np.ndim(brain_label_arr_nrrd))
print("nrrd 예제 레이블 모양 : ", np.shape(brain_label_arr_nrrd))

plt.subplot(241)
plt.imshow(img_ch1)
plt.subplot(242)
plt.imshow(img_ch2)
plt.subplot(245)
plt.imshow(img_ch3)
plt.subplot(246)
plt.imshow(img_ch4)
plt.subplot(122)
plt.imshow(img_seg)
plt.show()

logger = radiomics.logger
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

settings = {}
settings['binWidth'] = 1
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

extractor.disableAllFeatures()

extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])

print("Calculating features")
featureVector = extractor.execute(imageName, maskName)
for featureName in featureVector.keys():
    print("Computed %s: %s" % (featureName, featureVector[featureName]))
"""