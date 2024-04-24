# Use modules outside of directory
import sys
sys.path.append("../Modules")

# Import necessary libraries and modules
import numpy as np
from NNmodule import basicNN, loadCSV, splitTrainTest

# Maps output to number (will automate later)
outMap = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

# Split inputs and outputs
def splitInOut(data, outI, outSize, ignore=[]):
    inData = []
    outData = []
    outI %= len(data[0])
    if type(ignore) == int:
        ignore = [ignore]
    for line in range(len(data)):
        inData.append([])
        for i in range(len(data[line])):
            if (i != outI) and (i not in ignore):
                inData[line].append(float(data[line][i])/10)
        outVals = np.zeros(outSize)
        outVals[outMap[data[line][outI]]] = 1
        outData.append(outVals)
    return inData, outData

# Set up everything
irisData = loadCSV("Iris.csv", skipheader=True)
trainData, testData = splitTrainTest(irisData, 0.8)
trainIn, trainOut = splitInOut(trainData, -1, 3, ignore=0)
testIn, testOut = splitInOut(testData, -1, 3, ignore=0)

# Build and train a classifier
irisClassifier = basicNN([4, 10, 10, 3], "Leaky RelU")
acc = irisClassifier.train(trainIn, trainOut, method="minibatch", epochs=100,
                           earlystopping=False, testin=testIn, testout=testOut,
                           rate=0.1, returnacc=True)

# Accuracy of the neural network
print("\nThe model had an accuracy of ", acc, "% on the test dataset!")