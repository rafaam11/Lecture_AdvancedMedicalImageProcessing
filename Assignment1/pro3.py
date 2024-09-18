print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pydicom
import cv2
import math
from time import time
from PIL import Image
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# #############################################################################
# psnr 함수
def psnr(ori_img, con_img):
    max_pixel = np.max(con_img)
    mse = np.mean((ori_img - con_img)**2)
    if mse ==0:
        return 100
    psnr = 20* math.log10(max_pixel / math.sqrt(mse))
    return psnr

# ############################################################################
# 이미지 로드
path_LDCT105 = "L506_QD_3_1.CT.0003.0105.2015.12.22.20.45.42.541197.358793241.IMA"
path_NDCT105 = "L506_FD_3_1.CT.0001.0105.2015.12.22.20.19.39.34094.358586575.IMA"


# 테스트와 트레이닝할 이미지 정의
print('Upload images...')
test_img = pydicom.dcmread(path_LDCT105).pixel_array
train_img = pydicom.dcmread(path_NDCT105).pixel_array


# 0과 1사이의 값으로 표현하기 위한 변환
train = train_img / np.max(train_img)
test = test_img / np.max(test_img)
height, width = train.shape


# 트레이닝 이미지로부터 모든 기준 패치들을 추출
print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(train[:, :], patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))

# #############################################################################
# 기준 패치로부터 딕셔너리 학습
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from NDCT patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


# #############################################################################
# 학습 전 이미지 디스플레이
def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))

    plt.subplot(1, 3, 1)
    plt.title('Training Image')
    plt.imshow(reference, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.subplot(1, 3, 2)
    plt.title('Test Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.subplot(1, 3, 3)
    difference = image - reference
    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


show_with_diff(test, train, 'Before Learning')

# #############################################################################
# 노이즈 패치들을 추출하고 딕셔너리를 사용하여 이미지 재구성
print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(train[:, :], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 5}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
psnr_dic = []
psnr_dic.append(psnr(train,test))
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = train.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
    reconstructions[title][:, :] = reconstruct_from_patches_2d(
        patches, (height, width))
    dt = time() - t0
    print('done in %.2fs.' % dt)
    show_with_diff(reconstructions[title], train,
                   title + ' (time: %.1fs)' % dt)

    psnr_value = psnr(train,reconstructions[title])
    psnr_dic.append(psnr_value)
print(psnr_dic[0], psnr_dic[1], psnr_dic[2], psnr_dic[3], psnr_dic[4])
plt.show()

