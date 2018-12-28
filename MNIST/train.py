import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

FileName_TrainingImage = "train-images.idx3-ubyte"
FileName_TrainingLabel = "train-labels.idx1-ubyte"

FileName_TestingImage = "t10k-images.idx3-ubyte"
FileName_TestingLabel = "t10k-labels.idx1-ubyte"

PreDefine_TrainingNumber = 60000
PreDefine_TestNumber = 10000

Xtrain = np.zeros(shape=(PreDefine_TrainingNumber, 28*28))
Ytrain = np.zeros(shape=(PreDefine_TrainingNumber, 10))

Xtest = np.zeros(shape=(PreDefine_TestNumber, 28*28))
Ytest = np.zeros(shape=(PreDefine_TestNumber, 10))

def Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, limitNumber):
    for filename in ["train-images.idx3-ubyte",  "train-labels.idx1-ubyte",  "t10k-images.idx3-ubyte",   "t10k-labels.idx1-ubyte"]:
        try:
            f = open(filename, "rb")
            TrainMagic = struct.unpack(">i", f.read(4))[0]
            TrainNumber = struct.unpack(">i", f.read(4))[0]

            if "images" in filename:
                RowNumber = struct.unpack(">i", f.read(4))[0]
                ColNumber = struct.unpack(">i", f.read(4))[0]

                for i in range(0, limitNumber):
                    print("%s - %d/%d" %(filename, i+1, TrainNumber))
                    for j in range(0, RowNumber * ColNumber):
                        if "train" in filename:
                            Xtrain[i][j] = struct.unpack("B", f.read(1))[0]
                        else:
                            Xtest[i][j] = struct.unpack("B", f.read(1))[0]
            else:
                for i in range(0, limitNumber):
                    print("%s - %d/%d" % (filename, i+1, TrainNumber))
                    if "train" in filename:
                        Ytrain[i][struct.unpack("B", f.read(1))[0]] = 1;
                    else:
                        Ytest[i][struct.unpack("B", f.read(1))[0]] = 1;
        except IOError:
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

if __name__ == "__main__":
    if Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest, 500):
        DisplayLoadedMNISTPicture(0, True)
        DisplayLoadedMNISTPicture(1, False)

