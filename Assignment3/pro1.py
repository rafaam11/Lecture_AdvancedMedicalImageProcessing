import numpy as np
import os
from matplotlib import pyplot as plt

# ################################################################################
# 데이터 셋업

# Global values 및 ndarray 생성
n_test = 84
ch = 4
H = 240
W = 240
test_img = np.empty((n_test, ch, H, W))
test_sol = np.empty((n_test, 1))
test_seg = np.empty((n_test, H, W))

Data_path = os.getcwd()

# test 데이터 리스트 생성
walking_test = os.path.join(Data_path, 'clf_w_mask', 'test')
i = 0
for path, dirs, files in os.walk(walking_test):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_label = os.path.join(path, 'label.npy')
        path_seg = os.path.join(path, 'seg.npy')
        img = np.load(path_img)
        label = np.load(path_label)
        seg = np.load(path_seg)
        test_img[i] = img
        test_sol[i] = label
        test_seg[i] = seg
        i += 1

# img의 intensity를 0~255사이 범위로 Normalization
print('Before test(min, max) :', np.min(test_img), np.max(test_img))
print('Before testseg(min, max) :', np.min(test_seg), np.max(test_seg))
test_img = ( test_img - np.min(test_img) ) / (np.max(test_img) - np.min(test_img)) * 255
test_seg = ( test_seg - np.min(test_seg) ) / (np.max(test_seg) - np.min(test_seg)) * 255
print('After test(min, max) :', np.min(test_img), np.max(test_img))
print('After testseg(min, max) :', np.min(test_seg), np.max(test_seg))


# binarization
test_img_bin = np.empty((H, W))
for x in range(H):
    for y in range(W):
        if test_img[5][2][x][y] >= 35:
            test_img_bin[x][y] = 255
        else:
            test_img_bin[x][y] = 0


# Region Growing 함수 구현
class Point(object):
    def __init__(self,x,y):
       self.x = x
       self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

    def getGrayDiff(img,currentPoint,tmpPoint):
        return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

    def selectConnects(p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img,seeds,thresh,p = 1):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = 255
        connects = Point.selectConnects(p)
        while(len(seedList)>0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x,currentPoint.y] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = Point.getGrayDiff(img,currentPoint,Point(tmpX,tmpY))

                if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                    seedMark[tmpX,tmpY] = label
                    seedList.append(Point(tmpX,tmpY))
        return seedMark

# region growing 구현
img1 = test_img_bin
seeds = [Point(104,181), Point(127,166)]
binaryImg = Point.regionGrow(img1,seeds,35)

# Region Growing할 이미지와 segmentation 정답을 열어 확인
plt.figure(figsize = (12,6))
plt.subplot(141)
plt.imshow(test_seg[5])
plt.subplot(142)
plt.imshow(test_img[5][2])
plt.subplot(143)
plt.imshow(test_img_bin)
plt.subplot(144)
plt.imshow(binaryImg)
plt.show()