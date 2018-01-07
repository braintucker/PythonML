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
reg_poly = PolynomialFeatures(degree=2)
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
