#!/usr/bin/env python
import urllib.request

import numpy
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import random
from math import sqrt
import matplotlib.pyplot as plot
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib.request.urlopen(target_url)
xlist = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.decode().strip().split(';')
        firstLine = False
    else:
        row = line.decode().strip().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xlist.append(floatRow)
nrow = len(xlist)
ncol = len(xlist[0])
random.seed(1)
nsample = int(nrow*0.30)
idxtest = random.sample(range(nrow),nsample)
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
nbagsamples = int(len(xTrain) * 0.5)
for itress in range (numtreesmax):
    idxbag = []
    for i in range (nbagsamples):
        idxbag.append(random.choice(range(len(xTrain))))
    xtrainbag = [xTrain[i] for i in idxbag]
    ytrainbag = [ytrain[i] for i in idxbag]
    modelList.append(DecisionTreeRegressor(max_depth=treedepth))
    modelList[-1].fit(xtrainbag,ytrainbag)
    latestprediction = modelList[-1].predict(xtest)
    predlist.append(list(latestprediction))
mse = []
allpredictions = []
for imodels in range(len(modelList)) :
    predictions = []
    for ipred in range(len(xtest)):
        predictions.append(sum([predlist[i][ipred] \
            for i in range(imodels+1)])/[imodels+1])
    allpredictions.append(predictions)
    errors = [(ytest[i] - predictions[i]) for i in range(len(ytest))]
    mse.append(sum([e*e for e in errors])/len(ytest))
nmodels = [i+1 for i in range(len(modelList))]
plot.plot(nmodels,mse)
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
