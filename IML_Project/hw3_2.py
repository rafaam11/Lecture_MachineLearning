import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cluster, mixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score


print('')
# ######################################## Problem 2 ######################################## #
print('** Problem 2 : Sample 100 images randomly from MNIST data set')
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
    print('sampled label {} length : '.format(k), len(globals()['label{}_100_idx'.format(k)]))

# 100개씩 추출된 각 레이블 인덱스에 대한 데이터셋을 저장
for k in range(10):
    globals()['data{}_100'.format(k)] = np.zeros(shape=(100, 784))
    for s in range(100):
        globals()['data{}_100'.format(k)][s] = X[globals()['label{}_100_idx'.format(k)][s]]
    print('sampled data {} shape: '.format(k), globals()['data{}_100'.format(k)].shape)

# 100개씩 추출된 데이터를 합쳐서 1000개의 훈련 데이터 만들기
X_train = np.concatenate([globals()['data{}_100'.format(0)],
                          globals()['data{}_100'.format(1)],
                          globals()['data{}_100'.format(2)],
                          globals()['data{}_100'.format(3)],
                          globals()['data{}_100'.format(4)],
                          globals()['data{}_100'.format(5)],
                          globals()['data{}_100'.format(6)],
                          globals()['data{}_100'.format(7)],
                          globals()['data{}_100'.format(8)],
                          globals()['data{}_100'.format(9)]])
y_train_idx = np.concatenate([globals()['label{}_100_idx'.format(0)],
                              globals()['label{}_100_idx'.format(1)],
                              globals()['label{}_100_idx'.format(2)],
                              globals()['label{}_100_idx'.format(3)],
                              globals()['label{}_100_idx'.format(4)],
                              globals()['label{}_100_idx'.format(5)],
                              globals()['label{}_100_idx'.format(6)],
                              globals()['label{}_100_idx'.format(7)],
                              globals()['label{}_100_idx'.format(8)],
                              globals()['label{}_100_idx'.format(9)]])

print('Final 1000 training data :', X_train.shape)
print('Ground truth label : \n', y[y_train_idx])

# random_idx = np.arange(0, 1000)
# np.random.shuffle(random_idx)
# X_train = X_train[random_idx]
# y_train_idx = y_train_idx[random_idx]

'''
# 데이터 확인
test_digit = globals()['data{}_100'.format(0)][50]
test_digit_image = test_digit.reshape(28, 28)
plt.imshow(test_digit_image, cmap="binary")
plt.axis("off")
plt.show()
'''

print('')
# ######################################## Problem 3 ######################################## #
print('** Problem 3 : Using data above, perform 4 kinds of clustering methods')
print('')
k = 10

print('* 1. Agglomerative clustering')
print('')
ward = cluster.AgglomerativeClustering(n_clusters=k, linkage="ward")
agglo_labels = ward.fit_predict(X_train)
print(agglo_labels.shape)
print(agglo_labels)

print('* 2. k-means clustering')
print('')
kmeans = cluster.KMeans(n_clusters=k)
kmeans_labels = kmeans.fit_predict(X_train)
print(kmeans_labels.shape)
print(kmeans_labels)

print('* 3. Gaussian mixture model')
print('')
gmm = mixture.GaussianMixture(n_components=k, covariance_type="full")
gmm_labels = gmm.fit_predict(X_train)
print(gmm_labels.shape)
print(gmm_labels)

print('* 4. Spectral clustering')
print('')
spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver="arpack", affinity="nearest_neighbors")
spectral_labels = spectral.fit_predict(X_train)
print(spectral_labels.shape)
print(spectral_labels)

print('')
# ######################################## Problem 4 ######################################## #
print('** Problem 4 : Rand index, mutual information')
print('')
ground_truth = y[y_train_idx]
agglomerative_pred = agglo_labels
kmeans_pred = kmeans_labels
gmm_pred = gmm_labels
spectral_pred = spectral_labels

print('')
print('* 1. Rand index')
print('')
rand_score1 = adjusted_rand_score(ground_truth, agglomerative_pred)
rand_score2 = adjusted_rand_score(ground_truth, kmeans_pred)
rand_score3 = adjusted_rand_score(ground_truth, gmm_pred)
rand_score4 = adjusted_rand_score(ground_truth, spectral_pred)
print('Rand score1 :', rand_score1)
print('Rand score2 :', rand_score2)
print('Rand score3 :', rand_score3)
print('Rand score4 :', rand_score4)

print('')
print('* 2. Mutual information based score')
print('')
mutual_score1 = adjusted_mutual_info_score(ground_truth, agglomerative_pred)
mutual_score2 = adjusted_mutual_info_score(ground_truth, kmeans_pred)
mutual_score3 = adjusted_mutual_info_score(ground_truth, gmm_pred)
mutual_score4 = adjusted_mutual_info_score(ground_truth, spectral_pred)
print('Mutual score1 :', mutual_score1)
print('Mutual score2 :', mutual_score2)
print('Mutual score3 :', mutual_score3)
print('Mutual score4 :', mutual_score4)

print('')
# ######################################## Problem 5 ######################################## #
print('** Problem 5 : Clustering the MNIST test data set using the center of each cluster')
print('')
# 테스트 데이터셋 생성 (앞에서부터 20000개의 데이터를 추출)
X_test = np.zeros(shape=(20000, 784))
y_test_idx = np.arange(20000)
for s in range(20000): X_test[s] = X[s]
y_test = y[y_test_idx]
print('test data shape: ', X_test.shape)
print('test label: \n', y_test, '(shape) :', y_test.shape)

# 각 클러스터의 평균을 계산하여 센터를 추출
agglo_centers = np.zeros(shape=(10, 784))
kmeans_centers = np.zeros(shape=(10, 784))
gmm_centers = np.zeros(shape=(10, 784))
spectral_centers = np.zeros(shape=(10, 784))
for k in range(10):
    agglo_avg_temp = np.zeros(shape=(1000, 784))
    kmeans_avg_temp = np.zeros(shape=(1000, 784))
    gmm_avg_temp = np.zeros(shape=(1000, 784))
    spectral_avg_temp = np.zeros(shape=(1000, 784))
    for i in range(1000):
        if agglomerative_pred[i] == k:
            agglo_avg_temp[i] = X_train[i]
        if kmeans_pred[i] == k:
            kmeans_avg_temp[i] = X_train[i]
        if gmm_pred[i] == k:
            gmm_avg_temp[i] = X_train[i]
        if spectral_pred[i] == k:
            spectral_avg_temp[i] = X_train[i]
    agglo_centers[k] = np.mean(agglo_avg_temp, axis=0)
    kmeans_centers[k] = np.mean(kmeans_avg_temp, axis=0)
    gmm_centers[k] = np.mean(gmm_avg_temp, axis=0)
    spectral_centers[k] = np.mean(spectral_avg_temp, axis=0)

# 1-NN classifier 모델을 생성하여 클러스터링 센터 10개에 대한 디시전 바운더리를 생성함
kNN1 = KNeighborsClassifier(n_neighbors=1)
kNN2 = KNeighborsClassifier(n_neighbors=1)
kNN3 = KNeighborsClassifier(n_neighbors=1)
kNN4 = KNeighborsClassifier(n_neighbors=1)
kNN1.fit(agglo_centers, np.arange(10))
kNN2.fit(kmeans_centers, np.arange(10))
kNN3.fit(gmm_centers, np.arange(10))
kNN4.fit(spectral_centers, np.arange(10))

# 학습된 1-NN 모델을 통해 테스트 데이터셋을 예측
predict1 = kNN1.predict(X_test)
predict2 = kNN2.predict(X_test)
predict3 = kNN3.predict(X_test)
predict4 = kNN4.predict(X_test)

# 우리가 아는 레이블과 예측된 값 사이의 정확도 출력
accuracy1 = adjusted_mutual_info_score(y_test, predict1)
accuracy2 = adjusted_mutual_info_score(y_test, predict2)
accuracy3 = adjusted_mutual_info_score(y_test, predict3)
accuracy4 = adjusted_mutual_info_score(y_test, predict4)
print('Accuracy of Agglomerative center using 1-NN : ', accuracy1)
print('Accuracy of Kmeans center using 1-NN : ', accuracy2)
print('Accuracy of GMM center using 1-NN : ', accuracy3)
print('Accuracy of Spectral center using 1-NN : ', accuracy4)

