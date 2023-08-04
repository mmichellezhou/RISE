import random
import numpy as np
import matplotlib.pyplot as plt

class SyntheticData:

    def __init__(self, numAdvs, numImpTypes, numImpPerType, scale = 10):
        self.numAdvs = numAdvs
        self.numImpTypes = numImpTypes
        self.scale = scale
        self.weights = []
        self.displayTimes = []
        self.numImpPerType = numImpPerType
        self.numImpressions = self.numImpTypes * self.numImpPerType
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
    
    def gausDistList(self, sd = 10):
        self.displayTimes = np.random.normal(random.uniform(0, 1), sd, self.numImpressions)
        print(self.displayTimes)
        return self.displayTimes
    
    def sampleBudgets(self, min, max, oneEqualToMin = False):
        if min == 0:
            print("Min must be greater than 0.")
            return
        if min > max:
            print("Min is greater than max.")
            max += min
            return
        for adv in range(self.numAdvs):
            self.budgets.append(random.randrange(min, max + 1))
        if oneEqualToMin:
            self.budgets[random.randrange(0, len(self.budgets))] = min
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
