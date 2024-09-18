import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ########################################################################################
# 데이터 준비
source_image = cv2.imread("source.png")
source_label = cv2.imread("source_label.png")
target_image = cv2.imread("target.png")
target_label = cv2.imread("target_label.png")

# ########################################################################################
# Q5. Extract object boundary points from the source and target labels.
source_label_gray = cv2.cvtColor(source_label, cv2.COLOR_RGB2GRAY)
target_label_gray = cv2.cvtColor(target_label, cv2.COLOR_RGB2GRAY)
contours1, hierarchy1 = cv2.findContours(source_label_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(target_label_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
boundary1 = cv2.drawContours(source_label, contours1, -1, (255,0,0), 3)
boundary2 = cv2.drawContours(target_label, contours2, -1, (255,0,0), 3)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title('Source Label Boundary')
plt.imshow(source_label, 'gray')
plt.axis("off")

plt.subplot(122)
plt.title('Target Label Boundary')
plt.imshow(target_label, 'gray')
plt.axis("off")

plt.show()

bd1 = np.array(contours1[0])
bd2 = np.array(contours2[0])



# ########################################################################################
# Q6. Implement Iterative Closest Point (ICP) method and find an affine transformation matrix.
indexes = [539, 538, 537, 536, 535, 534, 533]
bd1_modified = np.delete(bd1, indexes, 0)
bd1_modified = np.reshape(bd1_modified, (533,2))
bd2 = np.reshape(bd2, (533,2))

def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.001):  
    '''
    주어짐 점군 A, B에 대해 정합 행렬을 계산해 리턴함.
    Input:
        A: numpy 형태 Nxm 행렬. 소스(Src) mD points
        B: numpy 형태 Nxm 행렬. 대상(Dst) mD points
        init_pose: (m+1)x(m+1) 동차좌표계(homogeneous) 변환행렬
        max_iterations: 알고리즘 계산 중지 탈출 횟수
        tolerance: 수렴 허용치 기준
    Output:
        T: 최종 동차좌표계 변환 행렬. maps A on to B
        distances: 가장 가까운 이웃점 간 유클리드 오차 거리
        i: 수렴 반복 횟수
    '''

    assert A.shape == B.shape

    # 차원 획득
    m = A.shape[1]

    # 동차 좌표계 행렬을 만들고, 점군 자료를 추가
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # src 점군에 초기 자세 적용
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):  # 정합될때까지 반복 계산
        # 소스와 목적 점군 간에 가장 근처 이웃점 탐색. 계산량이 많음. 
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # 소스 점군에서 대상 점군으로 정합 시 필요한 변환행렬 계산
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # 소스 점군에 변환행렬 적용해 좌표 갱신
        src = np.dot(T, src)

        # 에러값 계산
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:  # 허용치보다 에러 작으면 탈출
            break
        prev_error = mean_error

    # 변환행렬 계산
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def nearest_neighbor(src, dst):
    '''
    소스와 목적 점군에 대한 가장 이웃한 점 계산
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: 가장 가까운 이웃점간 유클리드 거리
        indices: 목적 점군에 대한 가장 가까웃 이웃점의 인덱스들
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    '''
    m 공간 차원에서 점군 A에서 B로 맵핑을 위한 최소자승법 변환행렬 계산
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) 동차좌표 변환행렬. maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # 차원 얻기
    m = A.shape[1]

    # 각 점군 중심점 및 중심 편차 계산
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A    # A행렬 편차
    BB = B - centroid_B     # B행렬 편차

    # SVD이용한 회전 행렬 계산
    H = np.dot(AA.T, BB)    # 분산
    U, S, Vt = np.linalg.svd(H)   # SVD 계산
    R = np.dot(Vt.T, U.T)     # 회전 행렬

    # 반사된 경우 행렬 계산
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # 이동 행렬 계산
    t = centroid_B.T - np.dot(R, centroid_A.T)   # 

    # 동차 변환행렬 계산
    T = np.identity(m+1)
    T[:m, :m] = R # 회전 행렬
    T[:m, m] = t  # 이동 행렬

    return T, R, t


# ICP 실행
T, distances, iterations = icp(bd2, bd1_modified, tolerance=0.000001)

# Affine 변환 행렬, 반복 회수 출력
print('ICP algorithm affine transformation Matrix : \n', T)
print('iterations : ', iterations)


# ########################################################################################
# Q7. Transform the source image to the target image. Use the back-projection with bilinear interpolation. 

# Affine 역변환행렬 출력
mat = T.copy()
inv_mat = np.linalg.inv(mat)
indexes = [2]
inv_mat = np.delete(inv_mat, indexes, 0)
mat = np.delete(mat, indexes, 0)

# 디스플레이
plt.figure(figsize=(16, 8))

# Original source image & target image 디스플레이
plt.subplot(141)
plt.title('Original Source image')
plt.imshow(source_image, 'gray')
plt.axis("off")

plt.subplot(142)
plt.title('Original Target image')
plt.imshow(target_image, 'gray')
plt.axis("off")

# dst (source => target) 디스플레이
h,w,_ = target_image.shape
dst = cv2.warpAffine(source_image, inv_mat, (w, h))

plt.subplot(143)
plt.title('dst \n (Source => Target)')
plt.imshow(dst, 'gray')
plt.axis("off")

# back-projection (dst => source) 디스플레이, bilinear interpolation 옵션을 넣음
h1,w1,_ = source_image.shape
inv_dst = cv2.warpAffine(dst, mat,  (w, h), flags=cv2.INTER_LINEAR)

plt.subplot(144)
plt.title('back-projection \n (dst => source)')
plt.imshow(inv_dst, 'gray')
plt.axis("off")

plt.show()


# ########################################################################################
# Q8. Transform the source label to the target image. Use the back-projection with nearest neighbor interpolation. 


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
dst2 = cv2.warpAffine(source_label, inv_mat, (w, h) )

# dst2 디스플레이
plt.subplot(153)
plt.title('dst \n (Source label => Target image)')
plt.imshow(dst2, 'gray')
plt.axis("off")

# back-projection 2 (dst2 => source label) 정의. nearest neighbor interpolation 옵션을 넣음.
inv_dst2 = cv2.warpAffine(dst2, mat, (w, h), flags=cv2.INTER_NEAREST)
 
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

true = np.array(target_label[:,:,1])
pred = np.array(inv_dst2[:,:,1])

print('Dice Similarity Score : ', dice_coef2(true, pred))
