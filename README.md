# Machine_Learning
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
columns= "sl sw pl pw".split()
df = pd.DataFrame(iris.data, columns=columns)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print ("Score:", model.score(X_test, y_test))
print ("Mean squared error: %.2f"
      % mean_squared_error(y_test, predictions))
print ("Variance score: %.2f" % r2_score(y_test, predictions))
