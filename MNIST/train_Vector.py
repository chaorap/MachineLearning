import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os
import datetime

PreDefine_IterationNumber = 1000
PreDefine_MiniBatchNumber = 60000

PreDefine_TrainingNumber = 60000
PreDefine_TestNumber = 10000

# 0 - Cancel L2
PreDefine_L2Reg = 0 #PreDefine_TrainingNumber 

PreDefine_LearnRateInit = 0.1
PreDefine_LearnRate = PreDefine_LearnRateInit
PreDefine_LearnRateDecayRate = 0.01

Xtrain = np.zeros(shape=(28*28, PreDefine_TrainingNumber), dtype=float)
Ytrain = np.zeros(shape=(10, PreDefine_TrainingNumber), dtype=float)

Xtest = np.zeros(shape=(28*28, PreDefine_TestNumber), dtype=float)
Ytest = np.zeros(shape=(10, PreDefine_TestNumber), dtype=float)

W1 = np.random.rand(15, 28*28)*np.sqrt(2/(28*28))/1000
B1 = np.random.rand(15, 1)*np.sqrt(2/(28*28))/1000
W2 = np.random.rand(10, 15)*np.sqrt(2/(15))/1000
B2 = np.random.rand(10, 1)*np.sqrt(2/(15))/1000

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

def softmax(z):
    maxz = np.max(z, axis=0, keepdims=True)
    ez = np.exp(z-maxz)
    sumez = np.sum(ez, axis=0, keepdims=True)
    return (ez)/sumez

def LeakyRelu(z, derivative=False):
    LeakyRate = 0.01
    if derivative == False:
        return np.where(z < 0, LeakyRate*z, z)
    else:
        return np.where(z < 0, LeakyRate, 1)

def CalcualteLabelValue(y):
    for i in range(0, 10):
        if y[i] >= 0.7:
            return i

def CalculateCrossEnt(a, y):
    return np.nan_to_num(-(y*np.log(a+1e-10) + (1-y)*np.log(1-a+1e-10)))

def TrainMNIST(inTrainNumber):
    global Xtrain, Ytrain, Xtest, Ytest
    global W1, B1, W2, B2

    OneXtrain = np.zeros(shape=(28*28, PreDefine_MiniBatchNumber), dtype=float)
    OneYtrain = np.zeros(shape=(10, PreDefine_MiniBatchNumber), dtype=float)

    if True == Load_MNIST_DataSet(inTrainNumber,IsTrain=True):

        Allstarttime = datetime.datetime.now()
        itera = -1
        OldValidJ = 10000000000000
        OldJ = 100000000000000
        PreDefine_LearnRate = PreDefine_LearnRateInit
        OldLearningRate = PreDefine_LearnRate
        OldW1 = W1
        OldW2 = W2
        OldB1 = B1
        OldB2 = B2

        #for itera in range(0, PreDefine_IterationNumber):
        IsExit = False
        while not IsExit:
            itera += 1
            OneIteraStartTime = datetime.datetime.now()
            for j in range(0, inTrainNumber, PreDefine_MiniBatchNumber):
                Onestarttime = datetime.datetime.now()

                OneXtrain[:,0:PreDefine_MiniBatchNumber] = Xtrain[:,j:j+PreDefine_MiniBatchNumber]
                OneYtrain[:,0:PreDefine_MiniBatchNumber] = Ytrain[:,j:j+PreDefine_MiniBatchNumber]

                #FP
                Z1 = np.dot(W1, OneXtrain) + B1
                A1 = LeakyRelu(Z1)

                Z2 = np.dot(W2, A1) + B2
                A2 = softmax(Z2)

                L0 = OneYtrain * np.log(A2 + 1e-10)
                L = -np.sum(L0, axis=0, keepdims=True)

                J = np.sum(L, axis=1, keepdims=True)/PreDefine_MiniBatchNumber

                if J > OldJ:
                    #PreDefine_LearnRate = (OldLearningRate + PreDefine_LearnRate)/2
                    PreDefine_LearnRate = PreDefine_LearnRate*0.9
                    OldJ = OldValidJ
                    W1 = OldW1
                    W2 = OldW2
                    B1 = OldB1
                    B2 = OldB2
                elif OldJ - J  <= 0.00000001:
                    IsExit = True
                else:
                    OldValidJ = OldJ
                    OldJ = J

                    #OldLearningRate = PreDefine_LearnRate
                    #PreDefine_LearnRate = PreDefine_LearnRateInit/(1 + PreDefine_LearnRateDecayRate*itera)
                    #PreDefine_LearnRate *= 2

                    #BP
                    dZ2 = A2 - OneYtrain
                    dW2 = np.dot(dZ2, A1.T)/PreDefine_MiniBatchNumber
                    dB2 = np.sum(dZ2, axis=1, keepdims=True)/PreDefine_MiniBatchNumber

                    #dA1 = np.dot(W2.T, dZ2)
                    #dZ1 = dA1
                    #dZ1 = dA1 * LeakyRelu(A1, True)
                    dZ1 = np.dot(W2.T, dZ2) * LeakyRelu(A1, True)
                    dW1 = np.dot(dZ1, OneXtrain.T)/PreDefine_MiniBatchNumber
                    dB1 = np.sum(dZ1, axis=1, keepdims=True)/PreDefine_MiniBatchNumber

                    #Update Parameter
                    OldW1 = W1
                    OldW2 = W2
                    OldB1 = B1
                    OldB2 = B2
                    L2RegPara = (1 - PreDefine_LearnRate*PreDefine_L2Reg/PreDefine_TrainingNumber)
                    W2 = L2RegPara*W2 - PreDefine_LearnRate * dW2
                    B2 = B2 - PreDefine_LearnRate * dB2
                    W1 = L2RegPara*W1 - PreDefine_LearnRate * dW1
                    B2 = B2 - PreDefine_LearnRate * dB2

                endtime = datetime.datetime.now()
                os.system('cls')
                print("Error: ", J, "\n", 
                        "Itear: ", itera, "\n", 
                        "LN: ", PreDefine_LearnRate, "\n",
                        j, "\n", 
                        endtime-Onestarttime, "\n", 
                        endtime-OneIteraStartTime, "\n",
                        endtime-Allstarttime)
        
        print("Finished !!!")
        np.savetxt("W1.txt", W1, fmt="%.20f", delimiter=",")
        np.savetxt("B1.txt", B1, fmt="%.20f", delimiter=",")
        np.savetxt("W2.txt", W2, fmt="%.20f", delimiter=",")
        np.savetxt("B2.txt", B2, fmt="%.20f", delimiter=",")

def TestMNIST(inTestNumber, IsTrain=True):
    global Xtrain, Ytrain, Xtest, Ytest
    global W1, B1, W2, B2

    CorrectNumber = 0
    FailedNumber = 0

    if True == Load_MNIST_DataSet(inTestNumber,IsTrain=IsTrain):
        #Step 1. Load trained parameter to Model
        W2 = np.loadtxt("W2.txt", dtype='float', delimiter=",")
        B2 = np.loadtxt("B2.txt", dtype='float', delimiter=",").reshape(10,1)
        W1 = np.loadtxt("W1.txt", dtype='float', delimiter=",")
        B1 = np.loadtxt("B1.txt", dtype='float', delimiter=",").reshape(15,1)

        OneXtest = np.zeros(shape=(28*28, 1), dtype=float)
        OneYtest = np.zeros(shape=(10, 1), dtype=float)

        for i in range(0, inTestNumber):
            if IsTrain == True:
                OneXtest[:,0] = Xtrain[:,i]
                OneYtest[:,0] = Ytrain[:,i]
            else:
                OneXtest[:,0] = Xtest[:,i]
                OneYtest[:,0] = Ytest[:,i]
            Z1 = np.dot(W1, OneXtest) + B1
            A1 = LeakyRelu(Z1)

            Z2 = np.dot(W2, A1) + B2
            A2 = softmax(Z2)

            CorrectResult = CalcualteLabelValue(OneYtest)
            PredictResult = CalcualteLabelValue(A2)
            if(CorrectResult == PredictResult):
                ResultString = "OK"
                CorrectNumber += 1
            else:
                ResultString = "Fail"
                FailedNumber += 1
            print(i, ResultString, CorrectResult, PredictResult, CorrectNumber, FailedNumber, CorrectNumber/inTestNumber)

            


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=20)
    #Verify Loaded data
    #Load_MNIST_DataSet(100,IsTrain=True)
    #DisplayLoadedMNISTPicture(99,IsTrain=True)

    TrainMNIST(PreDefine_TrainingNumber)
    #TestMNIST(PreDefine_TestNumber, IsTrain=False)



