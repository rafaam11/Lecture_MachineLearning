import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# ######################################## Data import ######################################## #
print('** Data import ')
print('')

# MNIST 데이터 불러오기
mnist = fetch_openml('mnist_784')
print('MNIST data shape: ', mnist.data.shape, 'MNIST label shape: ', mnist.target.shape)

# Train, Test 데이터 정의
X_train, X_trash, y_train, y_trash = train_test_split(mnist.data, mnist.target, test_size=0.95)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
print('train data shape: ', X_train.shape, 'train label shape: ', y_train.shape)
print('test data shape: ', X_test.shape, 'test label shape: ', y_test.shape)

# Regularization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('')
# ######################################## Problem 1 ######################################## #
print('** Problem 1 : Logistic regression models')
print('')

model1 = LogisticRegression(C=0.15, max_iter=100000)
model1.fit(X_train, y_train)
predict1 = model1.predict(X_test)

accuracy1 = accuracy_score(y_test, predict1)
mse1 = mean_squared_error(y_test, predict1)
rmse1 = np.sqrt(mse1)

print('accuracy :', accuracy1)
print('standard deviation :', rmse1)


print('')
# ######################################## Problem 2 ######################################## #
print('** Problem 2 : K-NN classifiers')
print('')

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
predict2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, predict2)
mse2 = mean_squared_error(y_test, predict2)
rmse2 = np.sqrt(mse2)

print('accuracy :', accuracy2)
print('standard deviation :', rmse2)

print('')
# ######################################## Problem 3 ######################################## #
print('** Problem 3 : SVM classifiers')
print('')

model3 = svm.SVC(C=10.0, kernel='poly', gamma=0.1)
model3.fit(X_train, y_train)
predict3 = model3.predict(X_test)

accuracy3 = accuracy_score(y_test, predict3)
mse3 = mean_squared_error(y_test, predict3)
rmse3 = np.sqrt(mse3)

print('accuracy :', accuracy3)
print('standard deviation :', rmse3)

print('')
# ######################################## Problem 4 ######################################## #
print('** Problem 4 : Random forest classifiers')
print('')

model4 = RandomForestClassifier(n_estimators=74, random_state=0)
model4.fit(X_train, y_train)
predict4 = model4.predict(X_test)

accuracy4 = accuracy_score(y_test, predict4)
mse4 = mean_squared_error(y_test, predict4)
rmse4 = np.sqrt(mse4)

print('accuracy :', accuracy4)
print('standard deviation :', rmse4)

