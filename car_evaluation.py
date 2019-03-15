#!/usr/bin/env python
author= "__Siddhant__"
import urllib.request
import numpy
from sklearn.tree import DecisionTreeRegressor
import random
import pandas as pd
from math import sqrt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plot
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
data = urllib.request.urlopen(target_url)
df = pd.read_csv(data)
xlist = []
labels = []
for line in data:
    row = line.decode().strip().split(",")
    labels.append(row[-1])
    row.pop()
    if (row[2]=="5more"):
        row[2]='5'
    if (row[3]=="more"):
        row[3]='5'
    arow= [num for num in row]
    xlist.append(arow)
labelencoder = LabelEncoder()
df_enco = df.apply(labelencoder.fit_transform)
print (df_enco)
onehotencoder = OneHotEncoder(categorical_features = [0])
de = df_enco.apply(onehotencoder.fit_transform)
print (de)
nrow=len(xlist)
ncol=len(xlist[0])
random.seed(1)
nsample = int(nrow*0.30)
idxtest =  random.sample(range(nrow),nsample)
idxtest.sort()
idxtrain = [idx for idx in range(nrow) if not(idx in idxtest)]
xTrain = [xlist[r] for r in idxtrain]
xtest = [xlist[r] for r in idxtest]
ytrain = [labels[r] for r in idxtrain]
ytest = [labels[r] for r in idxtest]
numtreesmax = 30
treedepth = 1
modelList = []
predlist = []
nbagsamples = int(len(xTrain)*0.5)
for itress in range (numtreesmax):
    idxbag=[]
    for i in range(nbagsamples):
        idxbag.append(random.choice(range(len(xTrain))))
    xtrainbag = [xTrain[i] for i in idxbag]
    ytrainbag = [ytrain[i] for i in idxbag]
    modelList.append(DecisionTreeRegressor(max_depth=treedepth))
    modelList[-1].fit(xtrainbag,ytrainbag)
    latestprediction = modelList[-1].predict(xtest)
    predlist.append(list(latestprediction))
mse = []
allpredictions = []
for imodels in range(len(modelList)):
    predictions = []
    for ipred in range(len(xtest)):
        predictions.append(sum([predlist[i][ipred] \
            for i in range(imodels+1)])/[imodels+1])
    allpredictions.append(predictions)
    errors = [(ytest[i] - predictions[i] for i in range(len(ytest)))]
    mse.append(sum([e*e for r in errors])/len(ytest))
nmodels = [i+1 for i in range(len(modelList))]
plot.plot(nomdels,mse)
plot.axis('tight')
plot.xlabel('Number of trees models in ensemble')
plot.ylabel("MSE")
plot.ylim((0.0,max(mse)))
plot.show()
print("Minimum mse")
print(min(mse))
for i in range(len(mse)):
    if (mse[i]==min(mse)):
        print(i+1)
