import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os

FileName_TrainingImage = "train-images.idx3-ubyte"
FileName_TrainingLabel = "train-labels.idx1-ubyte"

FileName_TestingImage = "t10k-images.idx3-ubyte"
FileName_TestingLabel = "t10k-labels.idx1-ubyte"

PreDefine_Mode  = "Train" # Test
PreDefine_MNIST_Number  = 60000
PreDefine_MNIST_TrainPercent = 0.83
PreDefine_TrainingNumber = PreDefine_MNIST_Number * PreDefine_MNIST_TrainPercent
PreDefine_TestNumber = PreDefine_MNIST_Number - PreDefine_TrainingNumber
PreDefine_LearnRate = 0.01

Xtrain = np.zeros(shape=(28*28, PreDefine_TrainingNumber), dtype="float")
Ytrain = np.zeros(shape=(10, PreDefine_TrainingNumber), dtype="float")

Xtest = np.zeros(shape=(28*28, PreDefine_TestNumber), dtype="float")
Ytest = np.zeros(shape=(10, PreDefine_TestNumber), dtype="float")

W1 = np.random.rand(15, 28*28)
B1 = np.random.rand(15, 1)
W2 = np.random.rand(10, 15)
B2 = np.random.rand(10, 1)

def Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, LimiteNumber):

    if PreDefine_Mode == "Train":
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

                for i in range(0, LimiteNumber):
                    print("%s - %d/%d" %(filename, i+1, LimiteNumber))
                    for j in range(0, RowNumber * ColNumber):
                        if "train" in filename:
                            Xtrain[i][j] = struct.unpack("B", f.read(1))[0]
                        else:
                            Xtest[i][j] = struct.unpack("B", f.read(1))[0]
            else:
                for i in range(0, LimiteNumber):
                    print("%s - %d/%d" % (filename, i+1, LimiteNumber))
                    if "train" in filename:
                        Ytrain[i][struct.unpack("B", f.read(1))[0]] = 1;
                    else:
                        Ytest[i][struct.unpack("B", f.read(1))[0]] = 1;
        except:
            print("Cannot open " + filename)
            return False
    return True

def DisplayLoadedMNISTPicture(PictureNumber, IsTrain=True):
    img = np.zeros(shape=(28,28))
    label = -100
    for i in range(0, 28):
        for j in range(0, 28):
            if IsTrain == True:
                img[i][j] = Xtrain[PictureNumber][i*28+j]
            else:
                img[i][j] = Xtest[PictureNumber][i*28+j]
    for i in range(0, 10):
        if IsTrain == True:
            if Ytrain[PictureNumber][i] > 0:
                label = i;
                break;
        else:
            if Ytest[PictureNumber][i] > 0:
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
    sigmoid = 1.0/(1.0+np.exp(-z))
    if (derivative==True):
        return sigmoid * (1-sigmoid)
    return sigmoid

def LeakyRelu(z, derivative=False):
    LeakyRate = 0.01
    if derivative == False:
        return max(0, z) + LeakyRate*min(0,z)
    else:
        if z>=0:
            return 1
        else: 
            return LeakyRate

def CalcualteLabelValue(y):
    for i in range(0, 10):
        if y[i] >= 0.7:
            return i

if __name__ == "__main__":

    if PreDefine_Mode == "Train":
        if Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, PreDefine_TrainingNumber):

            DisplayLoadedMNISTPicture(675, True)

            oldJ = -32132;
            while True:
                Z1 = np.dot(W1.T, Xtrain.T) + B1.T          #15 x m
                A1 = sigmod(Z1)                             #15 x m
                Z2 = np.dot(W2.T, A1) + B2.T                #10 x m
                A2 = sigmod(Z2)                             #10 x m

                dz2 = A2 - Ytrain.T                         #10 x m
                J = np.sum(dz2, axis=1, keepdims=True) / PreDefine_TrainingNumber
                os.system("cls")
                print(J)
                if abs(np.max(J) - oldJ) <= 0.0000000001:
                    break;
                oldJ = np.max(J)

                dw2 = np.dot(dz2, A1.T)                     #[10 x m] . [m x 15] = [10 x 15]
                db2 = dz2                                   #[10 x m]
                db2 = np.sum(db2, axis=1, keepdims=True)    #[10 x 1]
 
                dA1 = np.dot(W2, dz2)
                dz1 = A1*(1-A1) * dA1

                dw1 = np.dot(dz1, Xtrain)
                db1 = dz1
                db1 = np.sum(db1, axis=1, keepdims=True)

                W2 = W2 - PreDefine_LearnRate * dw2.T / PreDefine_TrainingNumber
                B2 = B2 - PreDefine_LearnRate * db2.T / PreDefine_TrainingNumber

                W1 = W1 - PreDefine_LearnRate * dw1.T / PreDefine_TrainingNumber
                B1 = B1 - PreDefine_LearnRate * db1.T / PreDefine_TrainingNumber

            print("Finished !!!")
            np.savetxt("W1.txt", W1, fmt="%f", delimiter=",")
            np.savetxt("B1.txt", B1, fmt="%f", delimiter=",")
            np.savetxt("W2.txt", W2, fmt="%f", delimiter=",")
            np.savetxt("B2.txt", B2, fmt="%f", delimiter=",")
    else:
        if Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, PreDefine_TestNumber):

            DisplayLoadedMNISTPicture(675, False)

            #try:
                #Step 1. Load trained parameter to Model
                W2 = np.loadtxt("W2.txt", dtype='float', delimiter=",")
                B2 = np.loadtxt("B2.txt", dtype='float', delimiter=",")
                W1 = np.loadtxt("W1.txt", dtype='float', delimiter=",")
                B1 = np.loadtxt("B1.txt", dtype='float', delimiter=",")

                #Step 2. Use the model to predict the result and Verify
                CorrectNumber = 0
                FailedNumber = 0
                for i in range(0, PreDefine_TestNumber):

                    Xi = Xtest[i].T.reshape(28*28,1)
                    Z1 = np.dot(W1.T, Xi) + B1           #15 x m
                    A1 = sigmod(Z1)                                                 #15 x m
                    Z2 = np.dot(W2.T, A1) + B2                                  #10 x m
                    A2 = sigmod(Z2)                                                 #10 x m

                    #Xi = Xtest[i].T.reshape(28*28,1)
                    #t0 = np.dot(W1.T, Xi)
                    #t1 = t0 + B1
                    #H1 = sigmod(t1)

                    #p0 = np.dot(W2.T, H1)
                    #p1 = p0 + B2
                    #H2 = sigmod(p1)
                    #H1 = sigmod(np.dot(Xtest[i], W1) + B1)
                    #H2 = sigmod(np.dot(H1, W2) + B2)

                    CorrectResult = CalcualteLabelValue(Ytest[i])
                    PredictResult = CalcualteLabelValue(A2)
                    if(CorrectResult == PredictResult):
                        ResultString = "OK"
                        CorrectNumber += 1
                    else:
                        ResultString = "Fail"
                        FailedNumber += 1
                    print(i, ResultString, CorrectResult, PredictResult, CorrectNumber, FailedNumber, CorrectNumber/PreDefine_TestNumber)
            #except:
            #    print("Cannot find the Trained Model Parameter files")




