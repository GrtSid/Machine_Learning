import pandas as pd         #import pandas library for data manipulation and analysis
import numpy as np          # import numpy for high-level mathematical functions to operate on multi-dimensional arrays
from numpy.linalg import inv
data = pd.read_excel('/home/grtsid/Downloads/ENB2012_data.xlsx',index=0) # reading Dataset from Excel file using Pandas
colums= (data.columns[0])       # Number of columns
max= [data[c].max() for c in data.columns]      # computing max and min values in each column and storing in list
min= [data[c].min() for c in data.columns]
i=0
for c in data.columns:
    while(i<len(data.columns)):                     # Normalizing Dataset
        data[c]=(data[c]-min[i])/(max[i]-min[i])
        i=i+1
        break
arr = data.values
x_train=[]
y1=[]
y2=[]
a=data.shape
for i in range(a[0]):                       # spliting dataset into attributes and target values
    x_train.append((arr[i][:-2]).tolist())
    y2.append(arr[i][-1])
    y1.append(arr[i][-2])
m=np.ones((768,1))
b=np.matrix(x_train)
b=np.concatenate((m,b),axis=1)
d=b.T
e=np.linalg.inv(np.matmul(d,b))
y1=np.matrix(y1)
y1=y1.T
y2=np.matrix(y2)
y2=y2.T
f=np.matmul(e,d)
c1=np.matmul(f,y1)
c2=np.matmul(f,y2)
x_test=[[1],]
for j in range (8):
    inp=[float(input("Enter Value:"))]
    x_test.append(inp)
for i in range(8):
    x_test[i+1][0]=(x_test[i+1][0]-min[i])/(max[i]-min[i])  # Normalizing test data
x_test=np.matrix(x_test)        
Y1=np.matmul(c1.T,x_test)
Y2=np.matmul(c2.T,x_test)
print(Y1*(max[-2]-min[-2])+min[-2])
print((Y2*(max[-1]-min[-1]))+min[-1])
