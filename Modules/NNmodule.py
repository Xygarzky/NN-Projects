import numpy as np
import random
import csv
import math
from tqdm import tqdm

# Turns CSV into list
def loadCSV(file, skipheader=False):
    extracted = []
    with open(file, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        if skipheader: next(csvReader)
        for line in tqdm(csvReader, desc="Loading Data"):
            extracted.append(line)
    return extracted

# Splits data between training and testing
def splitTrainTest(data, trainratio, shuffle=True):
    if shuffle: random.shuffle(data)
    spliceI = round(len(data)*trainratio)
    trainData = []
    testData = []
    for line in range(spliceI): trainData.append(data[line])
    for line in range(spliceI, len(data)): testData.append(data[line])
    return trainData, testData
    

# Splits data into inputs and outputs
# Will add output map functionality
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
                inData[line].append(float(data[line][i]))
        outVals = np.zeros(outSize)
        outVals[int(data[line][outI])] = 1
        outData.append(outVals)
    return inData, outData

# Basic neural network with multiclass support
class basicNN:
    def __init__(self, layers, activ=[], alpha=0.05):
        # Number of nodes per layer
        self.layers = layers
        # Number of layers
        self.lCount = len(layers)
        # Deafult activation functions
        if activ == [] or type(activ) == str:
            if activ == []: func = "RelU"
            else: func = activ
            activ = [""]
            for i in range(1, self.lCount-1):
                activ.append(func)
            activ.append("softmax")
        self.activ = activ
        # Inputs and outputs to every hidden layer
        self.z = [[] for i in range(self.lCount)]
        self.a = [[] for i in range(self.lCount)]
        # He (Kaiming) initialization
        self.weights = [[]]
        self.dw = [[]]
        for i in range(1, self.lCount):
            stdev = math.sqrt(2/layers[i-1])
            dist = np.random.randn(layers[i-1], layers[i])
            self.weights.append(stdev*dist)
            self.dw.append(np.zeros((layers[i-1], layers[i])))
        # Initialize weights to 0
        self.bias = [np.zeros(layers[i]) for i in range(self.lCount)]
        self.db = [np.zeros(layers[i]) for i in range(self.lCount)]
        # Alpha value for Leaky RelU
        self.alpha = alpha
        # For early stopping
        self.best = -1
    
    def activate(self, layerI):
        layer = self.z[layerI]
        result = []
        # RelU: used in hidden layers
        if self.activ[layerI] == "RelU":
            return [max(i, 0) for i in layer]
        # Leaky RelU: tries to fix dying RelU problem
        if self.activ[layerI] == "Leaky RelU":
            for i in layer:
                if i > 0: result.append(i)
                else: result.append(self.alpha*i)
            return result
        # Softmax: used in output layer to convert it into probabilities
        if self.activ[layerI] == "softmax":
            maxNode = max(layer)
            denom = sum(pow(math.e, i-maxNode) for i in layer)
            return [pow(math.e, i-maxNode)/denom for i in layer]
    
    def predict(self, inputLayer):
        self.a[0] = inputLayer
        for i in range(1, self.lCount):
            # Multiply by the weights
            self.z[i] = np.matmul(self.a[i-1], self.weights[i])
            # Add the bias
            self.z[i] = np.add(self.z[i], self.bias[i])
            # Plug into activation function
            self.a[i] = self.activate(i)
        return self.a[self.lCount-1]
    
    def dzCalc(self, layerI):
        zLayer = self.z[layerI]
        aLayer = self.a[layerI]
        result = []
        # RelU: used in hidden layers
        if self.activ[layerI] == "RelU":
            return [int(zLayer[i]>0)*self.da[i] for i in range(len(self.da))]
        # Leaky RelU: tries to fix dying RelU problem
        if self.activ[layerI] == "Leaky RelU":
            for i in range(len(self.da)):
                if zLayer[i] > 0: result.append(self.da[i])
                else: result.append(self.alpha*self.da[i])
            return result
        # Softmax: used in output layer to convert it into probabilities
        if self.activ[layerI] == "softmax":
            dz = []
            # Optimization for last layer softmax
            if layerI == self.lCount-1:
                for i in range(len(aLayer)):
                    dzi = 0
                    for j in range(len(aLayer)):
                        dzi += self.outData[j]*(self.calcOut[i]-int(i==j))
                    dz.append(dzi)
                return dz
            # If some crazy person decides to use softmax in a hidden layer
            else:
                dzMatrix = [[] for i in aLayer]
                # a as rows, z as cols
                for i in range(len(aLayer)):
                    for j in range(len(aLayer)):
                        dzMatrix[i].append(aLayer[i]*((i==j)-aLayer[j]))
                return np.matmul(self.da, dzMatrix)
    
    def calcDerivs(self, inData, outData, noise):
        # Add some noise for better generalization
        if noise != 0:
            for i in range(len(inData)):
                inData[i] += noise*inData[i]*np.random.randn()
        # Backpropagation with matrices
        self.outData = outData
        self.calcOut = self.predict(inData)
        for i in range(self.lCount-1, 0, -1):
            dz = self.dzCalc(i)
            self.db[i] += dz
            self.dw[i] += np.matmul(np.transpose([self.a[i-1]]), [dz])
            self.da = np.matmul(dz, np.transpose(self.weights[i]))
    
    def updateParams(self, rate, batchsize):
        # Update weights and bias
        # Reset derivatives
        for i in range(1, self.lCount):
            for j in range(self.layers[i]):
                self.bias[i][j] -= self.db[i][j]*rate/batchsize
                self.db[i][j] = 0
                for k in range(self.layers[i-1]):
                    self.weights[i][k][j] -= self.dw[i][k][j]*rate/batchsize
                    self.dw[i][k][j] = 0
    
    def train(self, trainIn, trainOut, method="sgd", epochs=20, rate=0.05,
              optimizer="none", shuffle=True, batchsize=32, noise=0,
              earlystopping=False, stopdiff=0, testin=[], testout=[],
              showacc=False, returnacc=False, l1=0, l2=0):
        stopdiff *= len(testin)
        # Trains all parameters with batched data
        if method == "minibatch":
            trained = 0
            for epoch in tqdm(range(epochs), "Training"):
                # Shuffle training data
                trained += 1
                if shuffle:
                    allData = list(zip(trainIn, trainOut))
                    random.shuffle(allData)
                    trainIn, trainOut = zip(*allData)
                for samp in range(len(trainIn)):
                    self.calcDerivs(trainIn[samp], trainOut[samp], noise)
                    if (samp-1) % batchsize == 0 or samp == len(trainIn)-1:
                        self.updateParams(rate, batchsize)
                if showacc or earlystopping:
                    self.correct = 0
                    for i in range(len(testin)):
                        prediction = self.predict(testin[i])
                        if np.argmax(prediction) == np.argmax(testout[i]):
                            self.correct += 1
                if showacc or earlystopping:
                    print(" ", 100*self.correct/len(testin), "% Accuracy")
                if earlystopping:
                    if self.correct+stopdiff < self.best: break
                    self.best = max(self.best, self.correct)
        if returnacc:
            if not showacc:
                self.correct = 0
                for i in range(len(testin)):
                    prediction = self.predict(testin[i])
                    if np.argmax(prediction) == np.argmax(testout[i]):
                        self.correct += 1
            return 100*self.correct/len(testin)