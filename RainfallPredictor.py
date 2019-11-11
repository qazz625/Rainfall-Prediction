# KNN applied in barisal rainfall data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('data/barisal.csv')

dataset=dataset.fillna(0)

dataset['Month_Total']=dataset.iloc[:,5:].sum(axis=1)

#result of independentvariable
X = dataset.iloc[:,2:5].values


#result of dependent variable
y = dataset.iloc[:,36].values

#y=y.astype(int)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = 25)


neigh = KNeighborsRegressor(n_neighbors=8)
neigh.fit(X_train,y_train)

predicted1 = neigh.predict(X_test)
print("Prediction Result: ",predicted1)

print('\nR-squared test score: {:.3f}'.format(neigh.score(X_test,y_test))) # R-Squared test score

plt.plot(y_test,label='Actual')
plt.plot(predicted1,label='Predicted')
plt.legend()
plt.show()