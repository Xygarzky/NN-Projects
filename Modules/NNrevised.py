import numpy as np

class Module:
    def __call__(self, *args):
        return self.forward(*args)

class FCLayer(Module):
    def __init__(self, inSize, outSize, l1=0, l2=0):
        self.inSize = inSize
        self.outSize = outSize
        self.l1 = l1
        self.l2 = l2
        super().__init__()
    
class ReLU(Module):
    def __init__(self, var="standard", alpha=0.1):
        self.var = var
        self.alpha = alpha
    def forward(self, inData):
        self.inData = inData
        self.outData = np.array([])
        if self.var == "standard":
            for i in inData: np.append(self.outData, max(i, 0))
            return self.outData
        if self.var == "leaky" or self.var == "parametric":
            for i in inData:
                if i > 0: np.append(self.outData, i)
                else: np.append(self.outData, -self.alpha*i)
            return self.outData
    def train(self, da):
        dz = np.array([])
        if self.var == "standard":
            for i in self.inData:
                dz.append(int(i>0)*da[i])
            return dz
        if self.var == "leaky":
            for i in self.inData:
                mult = 1 if i > 0 else -self.alpha
                dz.append(mult*da[i])
            return dz
        
class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules
    def forward(self, data):
        for module in self.modules:
            data = module(data)
        return data
    
joey = ReLU()
print(joey([5, 3, 3]))