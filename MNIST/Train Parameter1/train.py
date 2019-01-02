import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

FileName_TrainingImage = "train-images.idx3-ubyte"
FileName_TrainingLabel = "train-labels.idx1-ubyte"

FileName_TestingImage = "t10k-images.idx3-ubyte"
FileName_TestingLabel = "t10k-labels.idx1-ubyte"

PreDefine_Mode  = "Test" # Test
PreDefine_TrainingNumber = 60000 #60000
PreDefine_TestNumber = 10000     #10000
PreDefine_LearnRate = 0.01

Xtrain = np.zeros(shape=(PreDefine_TrainingNumber, 28*28), dtype="float")
Ytrain = np.zeros(shape=(PreDefine_TrainingNumber, 10), dtype="float")

Xtest = np.zeros(shape=(PreDefine_TestNumber, 28*28), dtype="float")
Ytest = np.zeros(shape=(PreDefine_TestNumber, 10), dtype="float")

W1 = np.random.rand(28*28, 15)
B1 = np.random.rand(1, 15)
W2 = np.random.rand(15, 10)
B2 = np.random.rand(1, 10)

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
            return false
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

def CalcualteLabelValue(y):
    for i in range(0, 10):
        if y[i] >= 0.7:
            return i

if __name__ == "__main__":

    if PreDefine_Mode == "Train":
        if Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, PreDefine_TrainingNumber):
            for i in range(0, PreDefine_TrainingNumber):

                OldError = np.zeros(shape=(1,10))
                NewError = np.zeros(shape=(1,10))

                while 1>0:
                    H1 = sigmod(np.dot(Xtrain[i], W1) + B1) # 1x15      need T
                    H2 = sigmod(np.dot(H1, W2) + B2)        # 1x10      need T

                    DaoShuSquare = Ytrain[i] - H2           # 1x10      need T
                    DaoshuSigmodH2 = sigmod(H2, True)       # 1x10      need T
                    DaoshuSigmodH1 = sigmod(H1, True)       # 1x10      need T

                    DaoShuH2 = DaoShuSquare * DaoshuSigmodH2    # 1x10      need T
                    DaoShu1 = np.dot(DaoShuH2.T, H1)            # 1x10      need T

                    W2_Delta = DaoShuH2.T.dot(H1).T
                    B2_Delta = DaoShuH2

                    W1_Delta = DaoShuH2.dot(W2.T)               # 1x15
                    W1_Delta = Xtrain[i].reshape(1, 28*28).T.dot(W1_Delta)         # 784x15

                    B1_Delta = DaoShuH2.dot(W2.T)               # 1x15

                    W2 += PreDefine_LearnRate * W2_Delta
                    B2 += PreDefine_LearnRate * B2_Delta
                    W1 += PreDefine_LearnRate * W1_Delta
                    B1 += PreDefine_LearnRate * B1_Delta

                    OldError = NewError
                    NewError = H2_Error = np.abs(Ytrain[i] - H2)       # 1x10      need T
                    #NewError = H2_Error = Ytrain[i] - H2       # 1x10      need T
                    print(i, CalcualteLabelValue(Ytrain[i]), H2, end='\r')

                    if np.abs((NewError - OldError).mean()) < 0.00001:
                        break;

            print("Finished !!!")
            np.savetxt("W1.txt", W1, fmt="%f", delimiter=",")
            np.savetxt("B1.txt", B1, fmt="%f", delimiter=",")
            np.savetxt("W2.txt", W2, fmt="%f", delimiter=",")
            np.savetxt("B2.txt", B2, fmt="%f", delimiter=",")
    else:
        if Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, PreDefine_TestNumber):
        
            try:
                #Step 1. Load trained parameter to Model
                W2 = np.loadtxt("W2.txt", dtype='float', delimiter=",")
                B2 = np.loadtxt("B2.txt", dtype='float', delimiter=",")
                W1 = np.loadtxt("W1.txt", dtype='float', delimiter=",")
                B1 = np.loadtxt("B1.txt", dtype='float', delimiter=",")

                #Step 2. Use the model to predict the result and Verify
                CorrectNumber = 0
                FailedNumber = 0
                for i in range(0, PreDefine_TestNumber):
                    t0 = np.dot(Xtest[i], W1)
                    t1 = t0 + B1
                    H1 = sigmod(t1)

                    p0 = np.dot(H1, W2)
                    p1 = p0 + B2
                    H2 = sigmod(p1)
                    #H1 = sigmod(np.dot(Xtest[i], W1) + B1)
                    #H2 = sigmod(np.dot(H1, W2) + B2)

                    CorrectResult = CalcualteLabelValue(Ytest[i])
                    PredictResult = CalcualteLabelValue(H2)
                    if(CorrectResult == PredictResult):
                        ResultString = "OK"
                        CorrectNumber += 1
                    else:
                        ResultString = "Fail"
                        FailedNumber += 1
                    print(i, ResultString, CorrectResult, PredictResult, CorrectNumber, FailedNumber, CorrectNumber/PreDefine_TestNumber)
            except:
                print("Cannot find the Trained Model Parameter files")




