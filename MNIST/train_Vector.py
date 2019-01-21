import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os
import datetime

PreDefine_TrainingNumber = 60000
PreDefine_TestNumber = 10000
PreDefine_LearnRate = 0.01

Xtrain = np.zeros(shape=(28*28, PreDefine_TrainingNumber), dtype=float)
Ytrain = np.zeros(shape=(10, PreDefine_TrainingNumber), dtype=float)

Xtest = np.zeros(shape=(28*28, PreDefine_TestNumber), dtype=float)
Ytest = np.zeros(shape=(10, PreDefine_TestNumber), dtype=float)

W1 = np.random.rand(15, 28*28)
B1 = np.random.rand(15, 1)
W2 = np.random.rand(10, 15)
B2 = np.random.rand(10, 1)

def Load_MNIST_DataSet(LimitNumber, IsTrain=True):
    global Xtrain, Ytrain, Xtest, Ytest

    if IsTrain == True:
        MNIST_Files = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte"]
    else:
        MNIST_Files = ["t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"]

    for filename in MNIST_Files:
        try:
            f = open(filename, "rb")
            TrainMagic = struct.unpack(">i", f.read(4))[0]
            TrainNumber = struct.unpack(">i", f.read(4))[0]

            if "images" in filename:
                RowNumber = struct.unpack(">i", f.read(4))[0]
                ColNumber = struct.unpack(">i", f.read(4))[0]

                for i in range(0, LimitNumber):
                    print("%s - %d/%d" %(filename, i+1, LimitNumber))
                    for j in range(0, RowNumber * ColNumber):
                        if "train" in filename:
                            Xtrain[j][i] = struct.unpack("B", f.read(1))[0]
                        else:
                            Xtest[j][i] = struct.unpack("B", f.read(1))[0]
            else:
                for i in range(0, LimitNumber):
                    print("%s - %d/%d" % (filename, i+1, LimitNumber))
                    if "train" in filename:
                        Ytrain[struct.unpack("B", f.read(1))[0]][i] = 1;
                    else:
                        Ytest[struct.unpack("B", f.read(1))[0]][i] = 1;
        except:
            print("Cannot open " + filename)
            return False
    return True

def DisplayLoadedMNISTPicture(PictureNumber, IsTrain=True):
    global Xtrain, Ytrain, Xtest, Ytest
    img = np.zeros(shape=(28,28))
    label = -100
    for i in range(0, 28):
        for j in range(0, 28):
            if IsTrain == True:
                img[i][j] = Xtrain[i*28+j][PictureNumber]
            else:
                img[i][j] = Xtest[i*28+j][PictureNumber]
    for i in range(0, 10):
        if IsTrain == True:
            if Ytrain[i][PictureNumber] > 0:
                label = i;
                break;
        else:
            if Ytest[i][PictureNumber] > 0:
                label = i;
                break;

    if label >= 0:
        imgName = "Number_%d_Label_%d.tiff" %(PictureNumber, label)
        plt.imsave(imgName, img, format='tiff', cmap=plt.cm.gray)
        img=Image.open(imgName)
        plt.figure(imgName)
        plt.imshow(img)
        plt.show()

def sigmod(z, derivative=False):
    sigmoid = np.nan_to_num(1.0/(1.0+np.exp(-z)))
    if (derivative==True):
        return sigmoid * (1-sigmoid)
    return sigmoid

def LeakyRelu(z, derivative=False):
    LeakyRate = 0.1
    if derivative == False:
        return np.where(z < 0, LeakyRate*z, z)
    else:
        return np.where(z < 0, LeakyRate, 1)

def CalcualteLabelValue(y):
    for i in range(0, 10):
        if y[i] >= 0.7:
            return i

def CalculateCrossEnt(a, y):
    return np.nan_to_num(-(y*np.log(a) + (1-y)*np.log(1-a)))

def TrainMNIST(inTrainNumber):
    global Xtrain, Ytrain, Xtest, Ytest
    global W1, B1, W2, B2

    OneXtrain = np.zeros(shape=(28*28, 1), dtype=float)
    OneYtrain = np.zeros(shape=(10, 1), dtype=float)

    if True == Load_MNIST_DataSet(inTrainNumber,IsTrain=True):

        Allstarttime = datetime.datetime.now()

        for itera in range(0, 1000):
            OneIteraStartTime = datetime.datetime.now()
            for j in range(0, inTrainNumber):
                Onestarttime = datetime.datetime.now()

                OneXtrain[:,0] = Xtrain[:,j]
                OneYtrain[:,0] = Ytrain[:,j]

                #FP
                Z1 = np.dot(W1, OneXtrain) + B1
                A1 = LeakyRelu(Z1)

                Z2 = np.dot(W2, A1) + B2
                A2 = sigmod(Z2)

                J = CalculateCrossEnt(A2, OneYtrain)
                L = np.sum(J, axis=1, keepdims=True)

                #BP
                dZ2 = A2 - OneYtrain
                dW2 = np.dot(dZ2, A1.T)
                dB2 = np.sum(dZ2, axis=1, keepdims=True)

                dA1 = np.dot(W2.T, dZ2)
                dZ1 = dA1
                dZ1 = dA1 * LeakyRelu(dZ1, True)
                dW1 = np.dot(dZ1, OneXtrain.T)
                dB1 = np.sum(dZ1, axis=1, keepdims=True)

                #Update Parameter
                W2 = W2 - PreDefine_LearnRate * dW2
                B2 = B2 - PreDefine_LearnRate * dB2
                W1 = W1 - PreDefine_LearnRate * dW1
                B2 = B2 - PreDefine_LearnRate * dB2

                endtime = datetime.datetime.now()
                os.system('cls')
                print(L, "\n", 
                      itera, "\n", 
                      j, "\n", 
                      endtime-Onestarttime, "\n", 
                      endtime-OneIteraStartTime, "\n",
                      endtime-Allstarttime)
        
        print("Finished !!!")
        np.savetxt("W1.txt", W1, fmt="%f", delimiter=",")
        np.savetxt("B1.txt", B1, fmt="%f", delimiter=",")
        np.savetxt("W2.txt", W2, fmt="%f", delimiter=",")
        np.savetxt("B2.txt", B2, fmt="%f", delimiter=",")

def TestMNIST(inTestNumber):
    if True == Load_MNIST_DataSet(inTestNumber,IsTrain=False):
        retrurn


if __name__ == "__main__":
    #Verify Loaded data
    #Load_MNIST_DataSet(100,IsTrain=True)
    #DisplayLoadedMNISTPicture(99,IsTrain=True)

    TrainMNIST(PreDefine_TrainingNumber)
    #TestMNIST(PreDefine_TestNumber)



