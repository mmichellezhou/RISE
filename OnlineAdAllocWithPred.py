import numpy as np
from cvxopt import matrix, spmatrix, solvers
import time
import random
import math

class OnlineAdAllocWithPred:

    def __init__(self, b, t, w, predictor = 1, epsilon = -1, threshMethod = 1):
        self.budgets = b
        self.impressions = t
        self.weights = w
        self.predictor = predictor
        self.threshMethod = threshMethod
        self.numAdvertisers = len(self.weights)
        self.numImpressions = len(self.impressions)
        self.advertisers = [[] for i in range(self.numAdvertisers)]
        self.dummyAdvertiser = []
        self.thresholds = [0] * self.numAdvertisers
        self.runtime = 0
        self.LPRuntime = 0
        if self.predictor == 1:
            print("OPT")
            self.predictions = self.PRDPrimal()
        elif self.predictor == 2:
            print("DualBase")
            self.predictions = self.PRDDual(epsilon)
    
    def maxIndex(self, list):
        max = 0
        maxIndex = 0
        for i in range(len(list)):
            if list[i] > max:
                max = list[i]
                maxIndex = i
        return maxIndex

    # dual
    def PRDDual(self, epsilon):
        if epsilon == -1:
            sampleImps = self.impressions
        else: 
            sampleImps = random.sample(self.impressions, epsilon)
        numSampleImps = len(sampleImps)
        
        x = np.array([])
        I = np.array([])
        J = np.array([])
        # A = [[0] * (self.numAdvertisers * self.numImpressions + self.numAdvertisers + self.numImpressions) for i in range(self.numAdvertisers + self.numImpressions)]
        # A = np.zeros((self.numAdvertisers + self.numImpressions, self.numAdvertisers * self.numImpressions + self.numAdvertisers + self.numImpressions))
        # b = [0] * (self.numAdvertisers * self.numImpressions + self.numAdvertisers + self.numImpressions)
        b = np.zeros(self.numAdvertisers * numSampleImps + self.numAdvertisers + numSampleImps)
        # c = [0] * (self.numAdvertisers + self.numImpressions)
        c = np.zeros(self.numAdvertisers + numSampleImps)
        
        # c[:self.numAdvertisers] = [B for B in self.budgets]
        # c[self.numAdvertisers:] = [1.0] * self.numImpressions
        for i in range(self.numAdvertisers):
            for j in range(numSampleImps):
                index = i * numSampleImps + j
                # A[i][index] = -1.0
                # A[i][self.numAdvertisers * self.numImpressions + i] = -1.0
                # A[self.numAdvertisers + j][index] = -1.0
                # A[self.numAdvertisers + j][self.numAdvertisers * self.numImpressions + self.numAdvertisers + j] = -1.0
                x = np.append(x, [-1.0] * 4)
                I = np.append(I, [index, self.numAdvertisers * numSampleImps + i, index, self.numAdvertisers * numSampleImps + self.numAdvertisers + j])
                J = np.append(J, [i, i, self.numAdvertisers + j, self.numAdvertisers + j])
                b[index] = -1.0 * self.weights[i][sampleImps[j][0]]

                c[self.numAdvertisers + j] = 1.0
            c[i] = self.budgets[i]
        
        x = x.astype(float).tolist()
        I = I.astype(int).tolist()
        J = J.astype(int).tolist()

        A = spmatrix(x, I, J)
        b = matrix(b.tolist())
        c = matrix(c.tolist())

        # print("A: " + str(A))
        # print("b: " + str(b))
        # print("c: " + str(c))

        sol = solvers.lp(c, A, b)
        thresholds = sol['x'][:self.numAdvertisers]
        discGains = sol['x'][self.numAdvertisers:]

        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            res[self.maxADiscGain(self.impressions[i], thresholds)].append(i)
        
        return res

    # primal
    def PRDPrimal(self):
        x = np.array([])
        I = np.array([])
        J = np.array([])
        b = np.zeros(self.numAdvertisers + self.numImpressions + self.numAdvertisers * self.numImpressions)
        c = np.zeros(self.numAdvertisers * self.numImpressions)

        b[self.numAdvertisers : self.numAdvertisers + self.numImpressions] = 1.0
        for i in range(self.numAdvertisers):
            for j in range(self.numImpressions):
                index = i * self.numImpressions + j
                x = np.append(x, [1.0, 1.0, -1.0])
                I = np.append(I, [i, self.numAdvertisers + j, self.numAdvertisers + self.numImpressions + i * self.numImpressions + j])
                J = np.append(J, [index, index, index])
                c[index] = -1.0 * self.weights[i][self.impressions[j][0]]
            b[i] = self.budgets[i]
        
        x = x.astype(float).tolist()
        I = I.astype(int).tolist()
        J = J.astype(int).tolist()

        A = spmatrix(x, I, J)
        b = matrix(b.tolist())
        c = matrix(c.tolist())

        start = time.time()
        sol = solvers.lp(c, A, b)
        end = time.time()
        self.LPRuntime += end - start

        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            ws = [sol['x'][a * self.numImpressions + i] for a in range(self.numAdvertisers)]
            res[self.maxIndex(ws)].append(i)
        
        return res

    def getPRD(self, i):
        for j in range(self.numAdvertisers):
            r = self.predictions[j]
            if i in r:
                return j
        print("PRD not found")
        return
    
    def maxADiscGain(self, t, thresholds):
        max = 0
        maxA = 0
        for a in range(self.numAdvertisers):
            discGain = self.weights[a][t[0]] - thresholds[a]
            if (discGain > max):
                max = discGain
                maxA = a
        return maxA
    
    def updateThresh(self, a, alpha):
        B = self.budgets[a]
        e = (1 + 1 / B) ** B
        sum = 0
        i = 1
        for t in self.advertisers[a]:
            sum += self.weights[a][t[0]] * e ** ((alpha * (B - i)) / B)
            i += 1
        return (e ** (alpha/B) - 1) * sum / (e ** alpha - 1)
    
    # inserts impression t into advertiser adv's list sorted by increasing weight
    # O(n) time complexity
    # TODO make more efficient
    def insert(self, a, t):
        list = self.advertisers[a]
        if len(list) == 0:
            list.append(t)
        elif len(list) == 1 and self.weights[a][list[0][0]] < self.weights[a][t[0]]:
            list.append(t)
        else:
            index = len(list)
            for i in range(len(list)):
                if self.weights[a][list[i][0]] > self.weights[a][t[0]]:
                    index = i
                    break
            list.insert(index, t)
        return

    def algorithm1(self, alpha):
        start = time.time()
        B = min(self.budgets)
        e = (1 + 1 / B) ** B
        alphaB = B * (e ** (alpha / B) - 1)
        for i in range(len(self.impressions)):
            t = self.impressions[i]
            aPRD = self.getPRD(i)
            aEXP = self.maxADiscGain(t, self.thresholds)
            discGainPRD = self.weights[aPRD][t[0]] - self.thresholds[aPRD]
            discGainEXP = self.weights[aEXP][t[0]] - self.thresholds[aEXP]
            # if discounted gain is negative, continue to next impression
            if (discGainPRD <= 0 and discGainEXP <= 0):
                self.dummyAdvertiser.append(t)
                continue
            if (alphaB * discGainPRD >= discGainEXP):
                a = aPRD
            else:
                a = aEXP
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[a]) == self.budgets[a]:
                self.dummyAdvertiser.append(self.advertisers[a].pop(0))
            self.insert(a, t)
            if self.threshMethod == 1:
                # thresholds as exponential averages of weights
                self.thresholds[a] = self.updateThresh(a, alpha)
            else:
                # thresholds as lowest weights
                self.thresholds[a] = self.advertisers[a][0][0]
        end = time.time()
        self.runtime = end - start

        self.robustness = (e ** alpha - 1)/(B * e ** alpha * (e ** (alpha/B) - 1))
        self.consistency = (1 + 1 / (e ** alpha - 1) * max((1 / alphaB) * (e ** alpha - (e ** alpha - 1) / alphaB), math.log(e ** alpha))) ** (-1)
        # print(self.robustness, self.consistency)
        
        return self.advertisers