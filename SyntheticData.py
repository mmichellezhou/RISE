import random
import numpy as np
import matplotlib.pyplot as plt

class SyntheticData:

    def __init__(self, scale, numAdvs, numImpTypes, numImpPerType):
        self.scale = scale
        self.numAdvs = numAdvs
        self.numImpTypes = numImpTypes
        self.weights = []
        self.displayTimes = []
        self.numImpPerType = numImpPerType
        self.budgets = []
        self.impressions = []

    def listToMat(self, list):
        mat = np.mat(list)
        mat = mat.reshape(self.numAdvs, self.numImpTypes)
        return mat.tolist()

    # generates a matrix from exponential an distribution
    def expDistMat(self):
        self.weights = self.listToMat(np.random.exponential(self.scale, self.numAdvs * self.numImpTypes))
        return self.weights
    
    def gausDistList(self, sd = 0):
        self.displayTimes = np.random.normal(random.uniform(0, 1), sd, self.numImpTypes * self.numImpPerType)
        return self.displayTimes
    
    def sampleBudgets(self, min, max):
        if min == 0:
            print("Choose a min greater than 0.")
            return
        for adv in range(self.numAdvs):
            self.budgets.append(random.randrange(min, max + 1))
        return self.budgets

    def sampleImps(self):
        t = 0
        for i in range(self.numImpTypes):
            for j in range(self.numImpPerType):
                self.impressions.append((i, self.displayTimes[t]))
                t += 1
        self.impressions.sort(key=lambda a : a[1])
        return self.impressions

# count, bins, ignored = plt.hist(displayTimes, 100, density = True)
# plt.show()
