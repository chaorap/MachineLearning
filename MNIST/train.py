import numpy as np
import struct
import matplotlib.pyplot as plt

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

def Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest):
    for filename in ["train-images.idx3-ubyte",  "train-labels.idx1-ubyte",  "t10k-images.idx3-ubyte",   "t10k-labels.idx1-ubyte"]:
        try:
            f = open(filename, "rb")
            TrainMagic = struct.unpack(">i", f.read(4))[0]
            TrainNumber = struct.unpack(">i", f.read(4))[0]

            if "images" in filename:
                RowNumber = struct.unpack(">i", f.read(4))[0]
                ColNumber = struct.unpack(">i", f.read(4))[0]

                for i in range(0, TrainNumber - 1):
                    for j in range(0, RowNumber * ColNumber - 1):
                        if "train" in filename:
                            Xtrain[i][j] = struct.unpack("B", f.read(1))[0]
                        else:
                            Xtest[i][j] = struct.unpack("B", f.read(1))[0]
            else:
                for i in range(0, TrainNumber - 1):
                    if "train" in filename:
                        Ytrain[i][struct.unpack("B", f.read(1))[0]] = 1;
                    else:
                        Ytest[i][struct.unpack("B", f.read(1))[0]] = 1;

            #------------Debug code for display picture----------------
            #img = np.zeros(shape=(RowNumber,ColNumber))
            #for i in range(0, RowNumber):
            #    for j in range(0, ColNumber):
            #        img[i][j] = struct.unpack("B", f.read(1))[0]
            #plt.imsave('output.tiff', img, format='tiff', cmap=plt.cm.gray)
        except IOError:
            print("Cannot open " + filename)
            return 0

if __name__ == "__main__":
    Load_MNIST_DataSet(Xtrain, Ytrain, Xtest, Ytest)
    print(Xtrain)