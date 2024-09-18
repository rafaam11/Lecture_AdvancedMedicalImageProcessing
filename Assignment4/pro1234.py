import numpy as np
import cv2
from matplotlib import pyplot as plt

# ########################################################################################
# 데이터 준비
source_image = cv2.imread("source.png")
source_label = cv2.imread("source_label.png")
target_image = cv2.imread("target.png")
target_label = cv2.imread("target_label.png")


# ########################################################################################
# Q1. Annotate more than 3 corresponding points from the source and target images manually.

# 포인트들 정의
pts1 = np.float32([[64, 99], [155, 197], [195, 112]])
pts2 = np.float32([[87, 109], [170, 184], [211, 99]])

# Source 이미지에서 선택된 포인트들 빨간색으로 표시
cv2.circle(source_image, (64, 99), 3, (255,0,0), -1)
cv2.circle(source_image, (155, 197), 3, (255,0,0), -1)
cv2.circle(source_image, (195, 112), 3, (255,0,0), -1)

# Target 이미지에서 선택된 포인트들 파란색으로 표시
cv2.circle(target_image, (87, 109), 3, (0,0,255), -1)
cv2.circle(target_image, (170, 184), 3, (0,0,255), -1)
cv2.circle(target_image, (211, 99), 3, (0,0,255), -1)

# Original Source label, Target label 디스플레이
plt.figure(figsize=(16, 8))

plt.subplot(141)
plt.title('Original source image')
plt.imshow(source_image, 'gray')
plt.axis("off")

plt.subplot(142)
plt.title('Original target image')
plt.imshow(target_image, 'gray')
plt.axis("off")

# ########################################################################################
# Q2. Find an affine transformation matrix.

# Affine 변환행렬 출력
mat = cv2.getAffineTransform(pts1, pts2)
print(mat)
# Affine 역변환행렬 출력
inv_mat = cv2.getAffineTransform(pts2, pts1)
print(inv_mat)

# ########################################################################################
# 3. Transform the source image to the target image. Use the back-projection with bilinear interpolation. 

# dst (source => target) 정의
h,w,_ = target_image.shape
dst = cv2.warpAffine(source_image, mat, (w, h))

# dst 디스플레이
plt.subplot(143)
plt.title('dst \n (Source => Target)')
plt.imshow(dst, 'gray')
plt.axis("off")

# back-projection (dst => source) 정의. bilinear interpolation 옵션을 넣음
h1,w1,_ = source_image.shape
inv_dst = cv2.warpAffine(dst, inv_mat,  (w, h), flags=cv2.INTER_LINEAR)
 
plt.subplot(144)
plt.title('back-projection \n (dst => source)')
plt.imshow(inv_dst, 'gray')
plt.axis("off")

plt.show()

# ########################################################################################
# 4. Transform the source label to the target image. Use the back-projection with nearest neighbor interpolation. 

# Original Source label, Target image 디스플레이
plt.figure(figsize=(16, 8))

plt.subplot(151)
plt.title('Original source label')
plt.imshow(source_label, 'gray')
plt.axis("off")

plt.subplot(152)
plt.title('Original target image')
plt.imshow(target_image, 'gray')
plt.axis("off")

# dst2 (source label => target image) 정의
dst2 = cv2.warpAffine(source_label, mat, (w, h), )

# dst2 디스플레이
plt.subplot(153)
plt.title('dst \n (Source label => Target image)')
plt.imshow(dst2, 'gray')
plt.axis("off")

# back-projection 2 (dst2 => source label) 정의. nearest neighbor interpolation 옵션을 넣음.
inv_dst2 = cv2.warpAffine(dst2, inv_mat, (w, h), flags=cv2.INTER_NEAREST)
 
plt.subplot(154)
plt.title('Back-projection \n (dst => source label)')
plt.imshow(inv_dst2, 'gray')
plt.axis("off")

# Target label 디스플레이
plt.subplot(155)
plt.title('Target label')
plt.imshow(target_label, 'gray')
plt.axis("off")

plt.show()

# Compute the DSC score between the transformed label and the target label.
def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

true = np.array(target_label[:,:,0])
pred = np.array(inv_dst2[:,:,0])

print('Dice Similarity Score : ', dice_coef2(true, pred))
