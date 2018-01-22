# Apriori

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# training the apriori on the dataset
from apyori import apriori as ap
rules = ap(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = , min_length = 2)