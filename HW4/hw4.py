import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

print('')
# ######################################## Data Importing ######################################## #
print('** Data importing')
print('')
# MNIST 데이터 불러오기
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
X, y = mnist['data'], mnist['target']
X = np.asarray(X)
y = np.asarray(y).astype(int)

print('MNIST data shape: ', X.shape, 'MNIST label shape: ', y.shape)

# 각 숫자에 해당하는 레이블의 인덱스를 분류
for k in range(10):
    globals()['label{}'.format(k)] = []

for i in range(len(y)):
    for k in range(10):
        if y[i] == k: globals()['label{}'.format(k)].append(i)

for k in range(10):
    print('label {} length : '.format(k), len(globals()['label{}'.format(k)]))

# 각 레이블의 인덱스를 랜덤하게 100개씩 추출
for k in range(10):
    globals()['label{}_100_idx'.format(k)] = random.sample(globals()['label{}'.format(k)], 100)
    # print('sampled label {} length : '.format(k), len(globals()['label{}_100_idx'.format(k)]))

# 100개씩 추출된 각 레이블 인덱스를 저장
for k in range(10):
    globals()['data{}_100'.format(k)] = np.zeros(shape=(100, 784))
    for s in range(100):
        globals()['data{}_100'.format(k)][s] = X[globals()['label{}_100_idx'.format(k)][s]]
    # print('sampled data {} shape: '.format(k), globals()['data{}_100'.format(k)].shape)

# 숫자 2에 대한 100개의 데이터 저장
X_train_2 = globals()['data{}_100'.format(2)]
y_train_idx_2 = globals()['label{}_100_idx'.format(2)]
# print('Digit-2 training data shape :', X_train_2.shape)
# print('Ground truth label : \n', y[y_train_idx_2])

# 숫자 0~9에 대한 1000개의 데이터 저장
X_train_09 = np.concatenate([globals()['data{}_100'.format(0)],
                             globals()['data{}_100'.format(1)],
                             globals()['data{}_100'.format(2)],
                             globals()['data{}_100'.format(3)],
                             globals()['data{}_100'.format(4)],
                             globals()['data{}_100'.format(5)],
                             globals()['data{}_100'.format(6)],
                             globals()['data{}_100'.format(7)],
                             globals()['data{}_100'.format(8)],
                             globals()['data{}_100'.format(9)]])
y_train_idx_09 = np.concatenate([globals()['label{}_100_idx'.format(0)],
                                 globals()['label{}_100_idx'.format(1)],
                                 globals()['label{}_100_idx'.format(2)],
                                 globals()['label{}_100_idx'.format(3)],
                                 globals()['label{}_100_idx'.format(4)],
                                 globals()['label{}_100_idx'.format(5)],
                                 globals()['label{}_100_idx'.format(6)],
                                 globals()['label{}_100_idx'.format(7)],
                                 globals()['label{}_100_idx'.format(8)],
                                 globals()['label{}_100_idx'.format(9)]])


print('')
# ######################################## Problem 1 ######################################## #
print('** Problem 1 : Run the PCA and kernel PCA functions on the Digit-2-Space')
print('')

# Mean image 추출
X_train_mean_2 = np.mean(X_train_2, axis=0)
print(X_train_mean_2.shape)
rec_X_train_mean_2 = X_train_mean_2.reshape(28, 28)

# PCA
pca_2 = PCA(n_components=10)
pca_2.fit(X_train_2)
eigen_vec_0 = pca_2.components_
print('(PCA) eigenvector shape: ', eigen_vec_0.shape)

# Kernel PCA (kernel= 'linear')
kpca_1 = KernelPCA(n_components=10, kernel='linear')
kpca_1.fit(X_train_2.T)
eigen_vec_1 = kpca_1.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_1.shape)

# Kernel PCA (kernel= 'poly')
kpca_2 = KernelPCA(n_components=10, kernel='poly')
kpca_2.fit(X_train_2.T)
eigen_vec_2 = kpca_2.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_2.shape)

# Kernel PCA (kernel= 'rbf')
kpca_3 = KernelPCA(n_components=10, kernel='rbf')
kpca_3.fit(X_train_2.T)
eigen_vec_3 = kpca_3.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_3.shape)

# Kernel PCA (kernel= 'sigmoid')
kpca_4 = KernelPCA(n_components=10, kernel='sigmoid')
kpca_4.fit(X_train_2.T)
eigen_vec_4 = kpca_4.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_4.shape)

# Kernel PCA (kernel= 'cosine')
kpca_5 = KernelPCA(n_components=10, kernel='cosine')
kpca_5.fit(X_train_2.T)
eigen_vec_5 = kpca_5.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_5.shape)

''' Visualization of eigenvectors '''
fig, axs = plt.subplots(7, 11)
axs[0, 0].text(0.2, 0.5, 'Mean image', fontsize='large')
axs[0, 0].axis("off")
axs[1, 0].text(0.2, 0.5, 'PCA', fontsize='large')
axs[1, 0].axis("off")
axs[2, 0].text(0.2, 0.5, 'linear', fontsize='large')
axs[2, 0].axis("off")
axs[3, 0].text(0.2, 0.5, 'poly', fontsize='large')
axs[3, 0].axis("off")
axs[4, 0].text(0.2, 0.5, 'rbf', fontsize='large')
axs[4, 0].axis("off")
axs[5, 0].text(0.2, 0.5, 'sigmoid', fontsize='large')
axs[5, 0].axis("off")
axs[6, 0].text(0.2, 0.5, 'cosine', fontsize='large')
axs[6, 0].axis("off")

# Mean image
axs[0, 1].imshow(rec_X_train_mean_2, cmap="binary")

# PCA & Kernel PCA
for i in range(10):
    # PCA
    test_digit_2 = eigen_vec_0[i, :]
    test_digit_image_2 = test_digit_2.reshape(28, 28)
    axs[1, i+1].imshow(test_digit_image_2, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_1[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs[2, i+1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_2[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs[3, i+1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_3[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs[4, i+1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_4[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs[5, i+1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_5[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs[6, i+1].imshow(test_digit_image_09, cmap="binary")

    # Remove axis
    axs[0, i+1].axis("off")
    axs[1, i+1].axis("off")
    axs[2, i+1].axis("off")
    axs[3, i+1].axis("off")
    axs[4, i+1].axis("off")
    axs[5, i+1].axis("off")
    axs[6, i+1].axis("off")
plt.show()


''' Visualization of eigenvalues '''
fig, axs = plt.subplots(1, 6)

eigen_val0 = pca_2.singular_values_ ** 2
eigen_val1 = kpca_1.eigenvalues_
eigen_val2 = kpca_2.eigenvalues_
eigen_val3 = kpca_3.eigenvalues_
eigen_val4 = kpca_4.eigenvalues_
eigen_val5 = kpca_5.eigenvalues_
axs[0].plot(np.arange(1, 11), eigen_val0)
axs[0].set_title('PCA')
axs[1].plot(np.arange(1, 11), eigen_val1)
axs[1].set_title('Linear')
axs[2].plot(np.arange(1, 11), eigen_val2)
axs[2].set_title('Poly')
axs[3].plot(np.arange(1, 11), eigen_val3)
axs[3].set_title('rbf')
axs[4].plot(np.arange(1, 11), eigen_val4)
axs[4].set_title('Sigmoid')
axs[5].plot(np.arange(1, 11), eigen_val5)
axs[5].set_title('Cosine')

plt.show()


print('')
# ######################################## Problem 2 ######################################## #
print('** Problem 2 : Run the PCA and kernel PCA functions on all 1000 training images')
print('')

# Mean image 추출
X_train_mean_09 = np.mean(X_train_09, axis=0)
print(X_train_mean_09.shape)
rec_X_train_mean_09 = X_train_mean_09.reshape(28, 28)

# PCA
pca_09 = PCA(n_components=10)
pca_09.fit(X_train_09)
eigen_vec_09_0 = pca_09.components_
print('(PCA) eigenvector shape: ', eigen_vec_09_0.shape)

# Kernel PCA (kernel= 'linear')
kpca_09_1 = KernelPCA(n_components=10, kernel='linear')
kpca_09_1.fit(X_train_09.T)
eigen_vec_09_1 = kpca_09_1.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_09_1.shape)

# Kernel PCA (kernel= 'poly')
kpca_09_2 = KernelPCA(n_components=10, kernel='poly')
kpca_09_2.fit(X_train_09.T)
eigen_vec_09_2 = kpca_09_2.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_09_2.shape)

# Kernel PCA (kernel= 'rbf')
kpca_09_3 = KernelPCA(n_components=10, kernel='rbf')
kpca_09_3.fit(X_train_09.T)
eigen_vec_09_3 = kpca_09_3.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_09_3.shape)

# Kernel PCA (kernel= 'sigmoid')
kpca_09_4 = KernelPCA(n_components=10, kernel='sigmoid')
kpca_09_4.fit(X_train_09.T)
eigen_vec_09_4 = kpca_09_4.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_09_4.shape)

# Kernel PCA (kernel= 'cosine')
kpca_09_5 = KernelPCA(n_components=10, kernel='cosine')
kpca_09_5.fit(X_train_09.T)
eigen_vec_09_5 = kpca_09_5.eigenvectors_.T
print('(Kernel PCA) eigenvector shape: ', eigen_vec_09_5.shape)

''' Eigenvectors '''
fig2, axs2 = plt.subplots(7, 11)
axs2[0, 0].text(0.2, 0.5, 'Mean image', fontsize='large')
axs2[0, 0].axis("off")
axs2[1, 0].text(0.2, 0.5, 'PCA', fontsize='large')
axs2[1, 0].axis("off")
axs2[2, 0].text(0.2, 0.5, 'linear', fontsize='large')
axs2[2, 0].axis("off")
axs2[3, 0].text(0.2, 0.5, 'poly', fontsize='large')
axs2[3, 0].axis("off")
axs2[4, 0].text(0.2, 0.5, 'rbf', fontsize='large')
axs2[4, 0].axis("off")
axs2[5, 0].text(0.2, 0.5, 'sigmoid', fontsize='large')
axs2[5, 0].axis("off")
axs2[6, 0].text(0.2, 0.5, 'cosine', fontsize='large')
axs2[6, 0].axis("off")

# Mean image
axs2[0, 1].imshow(rec_X_train_mean_09, cmap="binary")

# PCA & Kernel PCA
for i in range(10):
    # PCA
    test_digit_09 = eigen_vec_09_0[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[1, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_09_1[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[2, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_09_2[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[3, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_09_3[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[4, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_09_4[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[5, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Kernel PCA
    test_digit_09 = eigen_vec_09_5[i, :]
    test_digit_image_09 = test_digit_09.reshape(28, 28)
    axs2[6, i + 1].imshow(test_digit_image_09, cmap="binary")

    # Remove axis
    axs2[0, i+1].axis("off")
    axs2[1, i+1].axis("off")
    axs2[2, i+1].axis("off")
    axs2[3, i+1].axis("off")
    axs2[4, i+1].axis("off")
    axs2[5, i+1].axis("off")
    axs2[6, i+1].axis("off")
plt.show()


''' Eigenvalues '''
fig2, axs2 = plt.subplots(1, 6)

eigen_val_09_0 = pca_09.singular_values_ ** 2
eigen_val_09_1 = kpca_09_1.eigenvalues_
eigen_val_09_2 = kpca_09_2.eigenvalues_
eigen_val_09_3 = kpca_09_3.eigenvalues_
eigen_val_09_4 = kpca_09_4.eigenvalues_
eigen_val_09_5 = kpca_09_5.eigenvalues_
axs2[0].plot(np.arange(1, 11), eigen_val_09_0)
axs2[0].set_title('PCA')
axs2[1].plot(np.arange(1, 11), eigen_val_09_1)
axs2[1].set_title('Linear')
axs2[2].plot(np.arange(1, 11), eigen_val_09_2)
axs2[2].set_title('Poly')
axs2[3].plot(np.arange(1, 11), eigen_val_09_3)
axs2[3].set_title('rbf')
axs2[4].plot(np.arange(1, 11), eigen_val_09_4)
axs2[4].set_title('Sigmoid')
axs2[5].plot(np.arange(1, 11), eigen_val_09_5)
axs2[5].set_title('Cosine')

plt.show()


print('')
# ######################################## Problem 3 ######################################## #
print('** Problem 3 : k-means clustering')
print('')


def projection(eigen_vec, ori_train):
    comp = []
    for i in range(10):
        zi = 0
        for j in range(784):
            zi += eigen_vec[i, j] * ori_train[:, j]
        comp.append(zi)
    pca_res = np.vstack(comp).T

    return pca_res


def matching_label(true, pred):
    cm = confusion_matrix(true, pred)
    cm_argmax = cm.argmax(axis=0)
    pred = np.array([cm_argmax[i] for i in pred])

    return pred


# PCA를 통해 추출한 eigenvector에 원본 트레이닝 데이터를 프로젝션
pca_proj_09 = projection(eigen_vec_09_0, X_train_09)
kpca_1_proj_09 = projection(eigen_vec_09_1, X_train_09)
kpca_2_proj_09 = projection(eigen_vec_09_2, X_train_09)
kpca_3_proj_09 = projection(eigen_vec_09_3, X_train_09)
kpca_4_proj_09 = projection(eigen_vec_09_4, X_train_09)
kpca_5_proj_09 = projection(eigen_vec_09_5, X_train_09)
print('Projection with 10 eigenvectors of pca (shape): ', pca_proj_09.shape)
print('Projection with 10 eigenvectors of linear kernel pca (shape): ', kpca_1_proj_09.shape)
print('Projection with 10 eigenvectors of poly kernel pca (shape): ', kpca_2_proj_09.shape)
print('Projection with 10 eigenvectors of rbf kernel pca (shape): ', kpca_3_proj_09.shape)
print('Projection with 10 eigenvectors of sigmoid kernel pca (shape): ', kpca_4_proj_09.shape)
print('Projection with 10 eigenvectors of cosine kernel pca (shape): ', kpca_5_proj_09.shape)

# 프로젝션된 데이터를 kmeans로 분류
kmeans_pca = cluster.KMeans(n_clusters=10)
kmeans_kpca_1 = cluster.KMeans(n_clusters=10)
kmeans_kpca_2 = cluster.KMeans(n_clusters=10)
kmeans_kpca_3 = cluster.KMeans(n_clusters=10)
kmeans_kpca_4 = cluster.KMeans(n_clusters=10)
kmeans_kpca_5 = cluster.KMeans(n_clusters=10)
kmeans_labels_pca = kmeans_pca.fit_predict(pca_proj_09)
kmeans_labels_kpca_1 = kmeans_kpca_1.fit_predict(kpca_1_proj_09)
kmeans_labels_kpca_2 = kmeans_kpca_2.fit_predict(kpca_2_proj_09)
kmeans_labels_kpca_3 = kmeans_kpca_3.fit_predict(kpca_3_proj_09)
kmeans_labels_kpca_4 = kmeans_kpca_4.fit_predict(kpca_4_proj_09)
kmeans_labels_kpca_5 = kmeans_kpca_5.fit_predict(kpca_5_proj_09)


# Kmeans의 label과 True label을 매칭시키기
ground_truth = y[y_train_idx_09]
kmeans_labels_pca = matching_label(ground_truth, kmeans_labels_pca)
kmeans_labels_kpca_1 = matching_label(ground_truth, kmeans_labels_kpca_1)
kmeans_labels_kpca_2 = matching_label(ground_truth, kmeans_labels_kpca_2)
kmeans_labels_kpca_3 = matching_label(ground_truth, kmeans_labels_kpca_3)
kmeans_labels_kpca_4 = matching_label(ground_truth, kmeans_labels_kpca_4)
kmeans_labels_kpca_5 = matching_label(ground_truth, kmeans_labels_kpca_5)

# Rand index, Mutual information score 계산
print('')
print('* 1. Rand index')
print('')
rand_score_pca = adjusted_rand_score(ground_truth, kmeans_labels_pca)
rand_score_kpca_1 = adjusted_rand_score(ground_truth, kmeans_labels_kpca_1)
rand_score_kpca_2 = adjusted_rand_score(ground_truth, kmeans_labels_kpca_2)
rand_score_kpca_3 = adjusted_rand_score(ground_truth, kmeans_labels_kpca_3)
rand_score_kpca_4 = adjusted_rand_score(ground_truth, kmeans_labels_kpca_4)
rand_score_kpca_5 = adjusted_rand_score(ground_truth, kmeans_labels_kpca_5)
print('rand_score_pca :', rand_score_pca)
print('rand_score_kpca_1 :', rand_score_kpca_1)
print('rand_score_kpca_2 :', rand_score_kpca_2)
print('rand_score_kpca_3 :', rand_score_kpca_3)
print('rand_score_kpca_4 :', rand_score_kpca_4)
print('rand_score_kpca_5 :', rand_score_kpca_5)

print('')
print('* 2. Mutual information based score')
print('')
mutual_score_pca = adjusted_mutual_info_score(ground_truth, kmeans_labels_pca)
mutual_score_kpca_1 = adjusted_mutual_info_score(ground_truth, kmeans_labels_kpca_1)
mutual_score_kpca_2 = adjusted_mutual_info_score(ground_truth, kmeans_labels_kpca_2)
mutual_score_kpca_3 = adjusted_mutual_info_score(ground_truth, kmeans_labels_kpca_3)
mutual_score_kpca_4 = adjusted_mutual_info_score(ground_truth, kmeans_labels_kpca_4)
mutual_score_kpca_5 = adjusted_mutual_info_score(ground_truth, kmeans_labels_kpca_5)
print('mutual_score_pca :', mutual_score_pca)
print('mutual_score_kpca_1 :', mutual_score_kpca_1)
print('mutual_score_kpca_2 :', mutual_score_kpca_2)
print('mutual_score_kpca_3 :', mutual_score_kpca_3)
print('mutual_score_kpca_4 :', mutual_score_kpca_4)
print('mutual_score_kpca_5 :', mutual_score_kpca_5)

print('')
# ######################################## Problem 4 ######################################## #
print('** Problem 4 : Classifying the MNIST test data set using the center of each cluster')
print('')
# 테스트 데이터셋 생성 (앞에서부터 50000개의 데이터를 추출)
X_test = np.zeros(shape=(50000, 784))
y_test_idx = np.arange(50000)
for s in range(50000): X_test[s] = X[s]
y_test = y[y_test_idx]

# 테스트 데이터셋 차원 축소 ( 784 -> 10 )
X_test_proj_pca = projection(eigen_vec_09_0, X_test)
X_test_proj_kpca_1 = projection(eigen_vec_09_1, X_test)
X_test_proj_kpca_2 = projection(eigen_vec_09_2, X_test)
X_test_proj_kpca_3 = projection(eigen_vec_09_3, X_test)
X_test_proj_kpca_4 = projection(eigen_vec_09_4, X_test)
X_test_proj_kpca_5 = projection(eigen_vec_09_5, X_test)

# Kmeans의 각 클러스터 센터를 추출
kmeans_center_pca = kmeans_pca.cluster_centers_
kmeans_center_kpca_1 = kmeans_kpca_1.cluster_centers_
kmeans_center_kpca_2 = kmeans_kpca_2.cluster_centers_
kmeans_center_kpca_3 = kmeans_kpca_3.cluster_centers_
kmeans_center_kpca_4 = kmeans_kpca_4.cluster_centers_
kmeans_center_kpca_5 = kmeans_kpca_5.cluster_centers_

# 1-NN classifier 모델을 생성하여 클러스터링 센터 10개에 대한 디시전 바운더리를 생성함
kNN1 = KNeighborsClassifier(n_neighbors=1)
kNN2 = KNeighborsClassifier(n_neighbors=1)
kNN3 = KNeighborsClassifier(n_neighbors=1)
kNN4 = KNeighborsClassifier(n_neighbors=1)
kNN5 = KNeighborsClassifier(n_neighbors=1)
kNN6 = KNeighborsClassifier(n_neighbors=1)
kNN1.fit(kmeans_center_pca, np.arange(10))
kNN2.fit(kmeans_center_kpca_1, np.arange(10))
kNN3.fit(kmeans_center_kpca_2, np.arange(10))
kNN4.fit(kmeans_center_kpca_3, np.arange(10))
kNN5.fit(kmeans_center_kpca_4, np.arange(10))
kNN6.fit(kmeans_center_kpca_5, np.arange(10))

# 학습된 1-NN 모델을 통해 테스트 데이터셋을 예측
predict1 = kNN1.predict(X_test_proj_pca)
predict2 = kNN2.predict(X_test_proj_kpca_1)
predict3 = kNN3.predict(X_test_proj_kpca_2)
predict4 = kNN4.predict(X_test_proj_kpca_3)
predict5 = kNN5.predict(X_test_proj_kpca_4)
predict6 = kNN6.predict(X_test_proj_kpca_5)

# 우리가 아는 레이블과 예측된 값 사이의 정확도 출력
accuracy1 = accuracy_score(y_test, predict1)
accuracy2 = accuracy_score(y_test, predict2)
accuracy3 = accuracy_score(y_test, predict3)
accuracy4 = accuracy_score(y_test, predict4)
accuracy5 = accuracy_score(y_test, predict5)
accuracy6 = accuracy_score(y_test, predict6)
print('Accuracy of PCA using 1-NN : ', accuracy1)
print('Accuracy of Linear kernel PCA using 1-NN : ', accuracy2)
print('Accuracy of Poly kernel PCA center using 1-NN : ', accuracy3)
print('Accuracy of rbf kernel PCA using 1-NN : ', accuracy4)
print('Accuracy of Sigmoid kernel PCA using 1-NN : ', accuracy5)
print('Accuracy of Cosine kernel PCA using 1-NN : ', accuracy6)


print('')
# ######################################## Problem 5 ######################################## #
print('** Problem 5 : Visualize 3 correctly classified and 3 incorrectly classified images for each class')
print('')

fig3, axs3 = plt.subplots(8, 11)

# Extracting correct & incorrect images for each class
for i in range(10):
    globals()['cor_{}'.format(i)] = []
    globals()['incor_{}'.format(i)] = []

for i in range(50000):
    for j in range(10):
        if y_test[i] == j:
            if predict1[i] == j: globals()['cor_{}'.format(j)].append(i)
            else: globals()['incor_{}'.format(j)].append(i)

# Visualization of images
for i in range(10):
    if len(globals()['cor_{}'.format(i)]) >= 3:
        globals()['num{}_cor_{}'.format(i, 0)] = X_test[globals()['cor_{}'.format(i)][0]]
        globals()['num{}_cor_{}'.format(i, 1)] = X_test[globals()['cor_{}'.format(i)][1]]
        globals()['num{}_cor_{}'.format(i, 2)] = X_test[globals()['cor_{}'.format(i)][2]]
        test_digit_1 = globals()['num{}_cor_{}'.format(i, 0)]
        test_digit_2 = globals()['num{}_cor_{}'.format(i, 1)]
        test_digit_3 = globals()['num{}_cor_{}'.format(i, 2)]
        test_digit_image_1 = test_digit_1.reshape(28, 28)
        test_digit_image_2 = test_digit_2.reshape(28, 28)
        test_digit_image_3 = test_digit_3.reshape(28, 28)
        axs3[1, i + 1].imshow(test_digit_image_1, cmap="binary")
        axs3[2, i + 1].imshow(test_digit_image_2, cmap="binary")
        axs3[3, i + 1].imshow(test_digit_image_3, cmap="binary")

    if len(globals()['cor_{}'.format(i)]) == 2:
        globals()['num{}_cor_{}'.format(i, 0)] = X_test[globals()['cor_{}'.format(i)][0]]
        globals()['num{}_cor_{}'.format(i, 1)] = X_test[globals()['cor_{}'.format(i)][1]]
        test_digit_1 = globals()['num{}_cor_{}'.format(i, 0)]
        test_digit_2 = globals()['num{}_cor_{}'.format(i, 1)]
        test_digit_image_1 = test_digit_1.reshape(28, 28)
        test_digit_image_2 = test_digit_2.reshape(28, 28)
        axs3[1, i + 1].imshow(test_digit_image_1, cmap="binary")
        axs3[2, i + 1].imshow(test_digit_image_2, cmap="binary")

    if len(globals()['cor_{}'.format(i)]) == 1:
        globals()['num{}_cor_{}'.format(i, 0)] = X_test[globals()['cor_{}'.format(i)][0]]
        test_digit_1 = globals()['num{}_cor_{}'.format(i, 0)]
        test_digit_image_1 = test_digit_1.reshape(28, 28)
        axs3[1, i + 1].imshow(test_digit_image_1, cmap="binary")

    globals()['num{}_incor_{}'.format(i, 0)] = X_test[globals()['incor_{}'.format(i)][0]]
    globals()['num{}_incor_{}'.format(i, 1)] = X_test[globals()['incor_{}'.format(i)][1]]
    globals()['num{}_incor_{}'.format(i, 2)] = X_test[globals()['incor_{}'.format(i)][2]]
    test_digit_1 = globals()['num{}_incor_{}'.format(i, 0)]
    test_digit_2 = globals()['num{}_incor_{}'.format(i, 1)]
    test_digit_3 = globals()['num{}_incor_{}'.format(i, 2)]
    test_digit_image_1 = test_digit_1.reshape(28, 28)
    test_digit_image_2 = test_digit_2.reshape(28, 28)
    test_digit_image_3 = test_digit_3.reshape(28, 28)
    axs3[5, i + 1].imshow(test_digit_image_1, cmap="binary")
    axs3[6, i + 1].imshow(test_digit_image_2, cmap="binary")
    axs3[7, i + 1].imshow(test_digit_image_3, cmap="binary")

# Plot settings
axs3[1, 0].text(0.2, 0.5, 'Correct 1', fontsize='large')
axs3[2, 0].text(0.2, 0.5, 'Correct 2', fontsize='large')
axs3[3, 0].text(0.2, 0.5, 'Correct 3', fontsize='large')
axs3[5, 0].text(0.2, 0.5, 'Incorrect 1', fontsize='large')
axs3[6, 0].text(0.2, 0.5, 'Incorrect 2', fontsize='large')
axs3[7, 0].text(0.2, 0.5, 'Incorrect 3', fontsize='large')
for i in range(10):
    axs3[0, i+1].text(0.5, 0.5, '{}'.format(i), fontsize='large')
    axs3[4, i+1].text(0.5, 0.5, '{}'.format(i), fontsize='large')
for i in range(8):
    for j in range(11):
        axs3[i, j].axis("off")

plt.show()
