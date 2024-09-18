import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# ######################################## Problem 1 ######################################## #
plt.figure(figsize=(8,6))
print('----> Problem 1')

# Define 10 random sample data
datalen = 10
Sample_x = np.linspace(0, 1, datalen)
Sample_y = np.array([np.sin(2*np.pi*Sample_x[k]) + random.gauss(0.0, 0.05) for k in range(datalen)])
Sample_x = np.array([Sample_x]).T
Sample_y = np.array([Sample_y]).T

# Define sine graph
x = np.arange(0, 1, 0.001)
y = np.sin(2*np.pi*x)

# Plot
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')

plt.title('problem 1: Plot 10 samples with sin(2πx) with a Gaussian noise')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

# ######################################## Problem 2 ######################################## #
plt.figure(figsize=(8,6))
print('----> Problem 2')


def poly_generation(order, sam_x, sam_y):
    model = LinearRegression()

    poly = PolynomialFeatures(degree = order)
    x_poly = poly.fit_transform(sam_x)
    model.fit(x_poly, sam_y)
    y_p = model.predict(x_poly)

    print(order, '차항 변환 데이터 : ', x_poly.shape)

    return y_p


def poly_drawing(points_x, points_y, order):
    fit = np.polyfit(points_x, points_y, order)
    fi = np.poly1d(fit)

    return fi


# Regression with sample points
poly_1 = poly_generation(1, Sample_x, Sample_y)
poly_3 = poly_generation(3, Sample_x, Sample_y)
poly_5 = poly_generation(5, Sample_x, Sample_y)
poly_9 = poly_generation(9, Sample_x, Sample_y)
poly_15 = poly_generation(15, Sample_x, Sample_y)

# Flatten
Sample_x = Sample_x.flatten()
poly_1 = poly_1.flatten()
poly_3 = poly_3.flatten()
poly_5 = poly_5.flatten()
poly_9 = poly_9.flatten()
poly_15 = poly_15.flatten()

# Generate fitting line
poly_line_1 = poly_drawing(Sample_x, poly_1, 1)
poly_line_3 = poly_drawing(Sample_x, poly_3, 3)
poly_line_5 = poly_drawing(Sample_x, poly_5, 5)
poly_line_9 = poly_drawing(Sample_x, poly_9, 9)
poly_line_15 = poly_drawing(Sample_x, poly_15, 15)

# Plot
plt.plot(x, poly_line_1(x), label='Poly order=1')
plt.plot(x, poly_line_3(x), label='Poly order=3')
plt.plot(x, poly_line_5(x), label='Poly order=5')
plt.plot(x, poly_line_9(x), label='Poly order=9')
plt.plot(x, poly_line_15(x), label='Poly order=15')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')

plt.title('problem 2: Generating regression lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

# ######################################## Problem 3 ######################################## #
plt.figure(figsize=(8,6))
print('----> Problem 3')

# Define 3 outlier points
outlier_x = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
outlier_y = [random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)]

Sample_x2 = np.append(Sample_x, outlier_x)
Sample_y2 = np.append(Sample_y, outlier_y)
Sample_x2 = np.array([Sample_x2]).T
Sample_y2 = np.array([Sample_y2]).T

# Regression with sample points
poly_1 = poly_generation(1, Sample_x2, Sample_y2)
poly_3 = poly_generation(3, Sample_x2, Sample_y2)
poly_5 = poly_generation(5, Sample_x2, Sample_y2)
poly_9 = poly_generation(9, Sample_x2, Sample_y2)
poly_15 = poly_generation(15, Sample_x2, Sample_y2)

# Flatten
Sample_x2_Flatten = Sample_x2.flatten()
poly_1 = poly_1.flatten()
poly_3 = poly_3.flatten()
poly_5 = poly_5.flatten()
poly_9 = poly_9.flatten()
poly_15 = poly_15.flatten()

# Generate fitting line
poly_line_1 = poly_drawing(Sample_x2_Flatten, poly_1, 1)
poly_line_3 = poly_drawing(Sample_x2_Flatten, poly_3, 3)
poly_line_5 = poly_drawing(Sample_x2_Flatten, poly_5, 5)
poly_line_9 = poly_drawing(Sample_x2_Flatten, poly_9, 9)
poly_line_15 = poly_drawing(Sample_x2_Flatten, poly_15, 15)

# Plot
plt.plot(x, poly_line_1(x), label='Poly order=1')
plt.plot(x, poly_line_3(x), label='Poly order=3')
plt.plot(x, poly_line_5(x), label='Poly order=5')
plt.plot(x, poly_line_9(x), label='Poly order=9')
plt.plot(x, poly_line_15(x), label='Poly order=15')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')
plt.scatter(outlier_x, outlier_y, label='3 outlier points')

plt.title('problem 3: Adding 3 outliers and generating regression lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()
# ######################################## Problem 4 ######################################## #
plt.figure(figsize=(16,12))
print('----> Problem 4')


def poly_generation_with_l2_regularization(order, sam_x, sam_y, alpha_val):

    poly = PolynomialFeatures(degree=order)
    x_poly = poly.fit_transform(sam_x)

    model = Ridge(alpha=alpha_val)
    model.fit(x_poly, sam_y)
    y_p = model.predict(x_poly)

    print('Alpha value', alpha_val)

    return y_p


def poly_generation_with_l1_regularization(order, sam_x, sam_y, alpha_val):
    poly = PolynomialFeatures(degree=order)
    x_poly = poly.fit_transform(sam_x)

    model = Lasso(alpha=alpha_val)
    model.fit(x_poly, sam_y)
    y_p = model.predict(x_poly)

    print('Alpha value', alpha_val)

    return y_p


alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

plt.subplot(2, 2, 1)
for i in alpha_values:
    # Regression with sample points
    poly_9 = poly_generation_with_l2_regularization(9, Sample_x2, Sample_y2, i)
    # Flatten
    poly_9 = poly_9.flatten()
    # Generate fitting line
    poly_ridge_9 = poly_drawing(Sample_x2_Flatten, poly_9, 9)
    # Plot
    plt.plot(x, poly_ridge_9(x), label='alpha= {}'.format(i))

plt.plot(x, poly_line_9(x), label='No regularization')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')
plt.scatter(outlier_x, outlier_y, label='3 outlier points')
plt.title('problem 4: 9 order regression + L2-regularization')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

plt.subplot(2, 2, 2)
for i in alpha_values:
    # Regression with sample points
    poly_15 = poly_generation_with_l2_regularization(15, Sample_x2, Sample_y2, i)
    # Flatten
    poly_15 = poly_15.flatten()
    # Generate fitting line
    poly_ridge_15 = poly_drawing(Sample_x2_Flatten, poly_15, 15)
    # Plot
    plt.plot(x, poly_ridge_15(x), label='alpha= {}'.format(i))

plt.plot(x, poly_line_15(x), label='No regularization')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')
plt.scatter(outlier_x, outlier_y, label='3 outlier points')
plt.title('problem 4: 15 order regression + L2-regularization')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

plt.subplot(2, 2, 3)
for i in alpha_values:
    # Regression with sample points
    poly_9 = poly_generation_with_l1_regularization(9, Sample_x2, Sample_y2, i)
    # Flatten
    poly_9 = poly_9.flatten()
    # Generate fitting line
    poly_ridge_9 = poly_drawing(Sample_x2_Flatten, poly_9, 9)
    # Plot
    plt.plot(x, poly_ridge_9(x), label='alpha= {}'.format(i))

plt.plot(x, poly_line_9(x), label='No regularization')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')
plt.scatter(outlier_x, outlier_y, label='3 outlier points')
plt.title('problem 4: 9 order regression + L1-regularization')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

plt.subplot(2, 2, 4)
for i in alpha_values:
    # Regression with sample points
    poly_15 = poly_generation_with_l1_regularization(15, Sample_x2, Sample_y2, i)
    # Flatten
    poly_15 = poly_15.flatten()
    # Generate fitting line
    poly_ridge_15 = poly_drawing(Sample_x2_Flatten, poly_15, 15)
    # Plot
    plt.plot(x, poly_ridge_15(x), label='alpha= {}'.format(i))

plt.plot(x, poly_line_15(x), label='No regularization')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x, Sample_y, label='10 Sample points')
plt.scatter(outlier_x, outlier_y, label='3 outlier points')
plt.title('problem 4: 15 order regression + L1-regularization')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()

# ######################################## Problem 5 ######################################## #
plt.figure(figsize=(8,6))
print('----> Problem 5')
# Define 100 random sample data
datalen = 100
Sample_x3 = np.linspace(0, 1, datalen)
Sample_y3 = np.array([np.sin(2*np.pi*Sample_x3[k]) + random.gauss(0.0, 0.05) for k in range(datalen)])
Sample_x3 = np.array([Sample_x3]).T
Sample_y3 = np.array([Sample_y3]).T

# Regression with sample points
poly_1 = poly_generation(1, Sample_x3, Sample_y3)
poly_3 = poly_generation(3, Sample_x3, Sample_y3)
poly_5 = poly_generation(5, Sample_x3, Sample_y3)
poly_9 = poly_generation(9, Sample_x3, Sample_y3)
poly_15 = poly_generation(15, Sample_x3, Sample_y3)

# Flatten
Sample_x3_Flatten = Sample_x3.flatten()
poly_1 = poly_1.flatten()
poly_3 = poly_3.flatten()
poly_5 = poly_5.flatten()
poly_9 = poly_9.flatten()
poly_15 = poly_15.flatten()

# Generate fitting line
poly_line_1 = poly_drawing(Sample_x3_Flatten, poly_1, 1)
poly_line_3 = poly_drawing(Sample_x3_Flatten, poly_3, 3)
poly_line_5 = poly_drawing(Sample_x3_Flatten, poly_5, 5)
poly_line_9 = poly_drawing(Sample_x3_Flatten, poly_9, 9)
poly_line_15 = poly_drawing(Sample_x3_Flatten, poly_15, 15)

# Plot
plt.plot(x, poly_line_1(x), label='Poly order=1')
plt.plot(x, poly_line_3(x), label='Poly order=3')
plt.plot(x, poly_line_5(x), label='Poly order=5')
plt.plot(x, poly_line_9(x), label='Poly order=9')
plt.plot(x, poly_line_15(x), label='Poly order=15')
plt.plot(x, y, label='sin(2πx)', color='k')
plt.scatter(Sample_x3, Sample_y3, label='100 Sample points')

plt.title('problem 5: Plot 100 samples and generating the regression lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])
plt.legend()
plt.show()