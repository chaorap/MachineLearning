import numpy as np
import matplotlib.pyplot as plt

#Initail value Y = W*X + B
W = -8.6
B = 10
Step = 0.0001

Ytrain = np.array([-2.1, -1.2, 2.2, 3.0], dtype="double")
Xtrain = np.array([-1.9, -1.3, 2.1, 3.1], dtype="double")

Size = Xtrain.size

plt.scatter(Xtrain,Ytrain)

def CalcError(w, b):
    err = 0
    for i in range(0, Size-1):
        err += ((Xtrain[i]*w + b - Ytrain[i])*(Xtrain[i]*w + b - Ytrain[i]))
    return err

def CalcErrorWDaoShu(w, b):
    ds = 0
    for i in range(0, Size-1):
        ds += 2*(Xtrain[i]*w + b - Ytrain[i])*Xtrain[i]
    return ds

def CalcErrorBDaoShu(w, b):
    ds = 0
    for i in range(0, Size-1):
        ds += 2*(Xtrain[i]*w + b - Ytrain[i])
    return ds

NewError = 0
OldError = CalcError(W, B)

while 1>0:
    W = W - Step*CalcErrorWDaoShu(W, B)
    B = B - Step*CalcErrorBDaoShu(W, B)
    print(NewError, W, B)
    OldError = NewError
    NewError = CalcError(W, B)
    if(abs(NewError - OldError) < 0.000000000001):
        break
    
print(W, B)
for i in range(0, Size-1):
    ModelOutput = W*Xtrain[i] + B
    print(Ytrain[i], ModelOutput, (ModelOutput-Ytrain[i])/Ytrain[i])

Xtest = np.arange(-2, 2, 0.001)
Ytest = W*Xtest + B
plt.plot(Xtest, Ytest)

plt.show()