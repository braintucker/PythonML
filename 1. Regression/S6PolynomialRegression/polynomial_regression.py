# Author @Brian Tucker
# Jan 2018

# polynomial regression

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 1:2 is set to keep X as a matrix
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(X, y)

# fitting polynomial regressions to the dataset
from sklearn.preprocessing import PolynomialFeatures
reg_poly = PolynomialFeatures(degree= 4)
X_poly = reg_poly.fit_transform(X)
reg_lin2 = LinearRegression()
reg_lin2.fit(X_poly, y)

# visualising the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, reg_lin.predict(X), color = 'blue')
plt.title('True Salary or Fabricated Salary [LinearRegression]')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, reg_lin2.predict(reg_poly.fit_transform(X_grid)), color = 'blue')
plt.title('True Salary or Fabricated Salary [PolynomialRegression]')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()