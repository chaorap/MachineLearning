import numpy as np
import matplotlib.pyplot as plt

#Initail value Y = W*X
W = -8.6
Step = 0.00001

Ytrain = np.array([-2.1, -1.2, 2.2, 3.0], dtype="double")
Xtrain = np.array([-1.9, -1.3, 2.1, 3.1], dtype="double")

Size = Xtrain.size

plt.scatter(Xtrain,Ytrain)

def CalcError(w):
    err = 0
    for i in range(0, Size-1):
        err += ((Xtrain[i]*w - Ytrain[i])*(Xtrain[i]*w - Ytrain[i]))
    return err

def CalcErrorDaoShu(w):
    ds = 0;
    for i in range(0, Size-1):
        ds += 2*(Xtrain[i]*w - Ytrain[i])*Xtrain[i]
    return ds

NewError = 0
OldError = CalcError(W)

while 1>0:
    W = W - Step*CalcErrorDaoShu(W)
    print(NewError, W)
    OldError = NewError
    NewError = CalcError(W)
    if(abs(NewError - OldError) < 0.000000000001):
        break
    
print(W)

Xtest = np.arange(-2, 2, 0.001)
Ytest = W*Xtest
plt.plot(Xtest, Ytest)

plt.show()