import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os

PreDefine_Mode  = "Train"
PreDefine_MNIST_Number  = 60000

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

def Load_MNIST_DataSet(LimitTrainNumber, LimitTestNumber, IsTrain=True):

    if IsTrain == "Train":
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

                for i in range(0, LimitTrainNumber):
                    print("%s - %d/%d" %(filename, i+1, LimitTrainNumber))
                    for j in range(0, RowNumber * ColNumber):
                        if "train" in filename:
                            Xtrain[j][i] = struct.unpack("B", f.read(1))[0]
                        else:
                            Xtest[j][i] = struct.unpack("B", f.read(1))[0]
            else:
                for i in range(0, LimitTestNumber):
                    print("%s - %d/%d" % (filename, i+1, LimitTestNumber))
                    if "train" in filename:
                        Ytrain[struct.unpack("B", f.read(1))[0]][i] = 1;
                    else:
                        Ytest[struct.unpack("B", f.read(1))[0]][i] = 1;
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
    #Load_MNIST_DataSet(100,100,False)
    #DisplayLoadedMNISTPicture(58,False)




