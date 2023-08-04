import numpy as np
from cvxopt import matrix, spmatrix, solvers
import time
import random
import math

class OnlineAdAllocWithPred:

    def __init__(self, b, t, w, predictor = 0, epsilon = -1, prevImps = [], threshMethod = 0):
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
        self.ALG = 0
        self.OPT = 0
        self.PRD = 0
        if self.predictor == 0:
            print("OPT")
            sol = self.PRDPrimal()
            self.predictions = self.OptimumSolution(sol)
        elif self.predictor == 1:
            print("Dual Base")
            if epsilon == -1:
                epsilon = 0.1
            sol = self.PRDDual(epsilon)
            self.predictions = self.DualBase(sol[0])
            print("predictions: " + str(self.predictions))
        elif self.predictor == 2:
            print("Previous Day")
            sol = self.PRDDual(epsilon, prevImps)
            self.predictions = self.DualBase(sol[0])
            print("predictions: " + str(self.predictions))
        else:
            print("Invalid predictor.")

    # dual
    def PRDDual(self, epsilon, prevImps = []):
        # print("In PRDDual.")
        # sampleImps = self.impressions
        print(epsilon, prevImps)
        if epsilon == -1 and prevImps:
            sampleImps = prevImps
        elif epsilon != -1 and not prevImps:
            sampleImps = random.sample(self.impressions, math.ceil(self.numImpressions * epsilon))
        else:
            print("Specify either epsilon for Dual Base or previous impressions for Previous Day.")
        # print("sample impressions: " + str(sampleImps))
        numSampleImps = len(sampleImps)

        x = np.array([])
        I = np.array([])
        J = np.array([])
        b = np.zeros(self.numAdvertisers * numSampleImps + self.numAdvertisers + numSampleImps)
        c = np.zeros(self.numAdvertisers + numSampleImps)
        
        for i in range(self.numAdvertisers):
            for j in range(numSampleImps):
                index = i * numSampleImps + j
                if i == 0:
                    x = np.append(x, [-1.0] * 3)
                    I = np.append(I, [index, index, self.numAdvertisers * numSampleImps + self.numAdvertisers + j])
                    J = np.append(J, [i, self.numAdvertisers + j, self.numAdvertisers + j])
                else:
                    x = np.append(x, [-1.0] * 2)
                    I = np.append(I, [index, index])
                    J = np.append(J, [i, self.numAdvertisers + j])
                b[index] = self.weights[i][sampleImps[j][0]]
                c[self.numAdvertisers + j] = 1.0
            x = np.append(x, [-1.0])
            I = np.append(I, [self.numAdvertisers * numSampleImps + i])
            J = np.append(J, [i])
            c[i] = self.budgets[i]
        
        x = x.astype(float).tolist()
        I = I.astype(int).tolist()
        J = J.astype(int).tolist()

        A = spmatrix(x, I, J)
        # print("A: " + str(A))
        b = matrix(b.tolist())
        # print("b: " + str(b))
        c = matrix(c.tolist())
        # print("c: " + str(c))

        start = time.time()
        sol = solvers.lp(c, A, b)
        end = time.time()
        self.LPRuntime += end - start

        thresholds = sol['x'][:self.numAdvertisers]
        # print("thresholds: " + str(thresholds))
        discGains = sol['x'][self.numAdvertisers:]
        # print("discounted gains: " + str(discGains))

        return (thresholds, discGains)

    def DualBase(self, thresholds):
        # print("In DualBase.")
        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            a = self.maxADiscGain(self.impressions[i], thresholds)
            res[a].append(i)
            self.PRD += self.weights[a][self.impressions[i][0]]
        
        return res

    # primal
    def PRDPrimal(self):
        # print("In PRDPrimal.")
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
                J = np.append(J, [index] * 3)
                c[index] = -1.0 * self.weights[i][self.impressions[j][0]]
            b[i] = self.budgets[i]
        
        x = x.astype(float).tolist()
        I = I.astype(int).tolist()
        J = J.astype(int).tolist()

        A = spmatrix(x, I, J)
        # print("A: " + str(A))
        b = matrix(b.tolist())
        # print("b: " + str(b))
        c = matrix(c.tolist())
        # print("c: " + str(c))

        start = time.time()
        sol = solvers.lp(c, A, b)
        end = time.time()
        self.LPRuntime += end - start

        return sol['x']

    def OptimumSolution(self, sol):
        # print("In OptimumSolution.")
        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            ws = [sol[a * self.numImpressions + i] for a in range(self.numAdvertisers)]
            a = ws.index(max(ws))
            res[a].append(i)
            self.OPT += self.weights[a][self.impressions[i][0]]
        
        if self.predictor == 0:
            self.PRD = self.OPT
        
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
        return (e ** (alpha / B) - 1) * sum / (e ** alpha - 1)
    
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

    def Algorithm1(self, alpha):
        start = time.time()
        self.alpha = alpha
        B = min(self.budgets)
        e = (1 + 1 / B) ** B
        alpha_B = B * (e ** (alpha / B) - 1)
        # print("alpha_B: " + str(alpha_B))
        for i in range(len(self.impressions)):
            t = self.impressions[i]
            # print("impression " + str(i) + ": " + str(t))
            aPRD = self.getPRD(i)
            aEXP = self.maxADiscGain(t, self.thresholds)
            discGainPRD = self.weights[aPRD][t[0]] - self.thresholds[aPRD]
            discGainEXP = self.weights[aEXP][t[0]] - self.thresholds[aEXP]
            # print("\taPRD: " + str(aPRD) + ", weight: " + str(self.weights[aPRD][t[0]]) + ", threshold: " + str(self.thresholds[aPRD]) + ", discGain: " + str(discGainPRD))
            # print("\taEXP: " + str(aEXP) + ", weight: " + str(self.weights[aEXP][t[0]]) + ", threshold: " + str(self.thresholds[aEXP]) + ", discGain: " + str(discGainEXP))
            # if discounted gain is negative, continue to next impression
            if (discGainPRD <= 0 and discGainEXP <= 0):
                # print("\tBoth discGainPRD and discGainEXP are not greater than 0.")
                self.dummyAdvertiser.append(t)
                continue
            if (alpha_B * discGainPRD >= discGainEXP):
                # print("\tAllocating to aPRD")
                a = aPRD
            else:
                # print("\tAllocating to aEXP")
                a = aEXP
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[a]) == self.budgets[a]:
                self.dummyAdvertiser.append(self.advertisers[a].pop(0))
            self.insert(a, t)
            # print("\ta: " + str(self.advertisers[a]))
            self.ALG += self.weights[a][t[0]]
            if self.threshMethod == 0:
                # thresholds as exponential averages of weights
                self.thresholds[a] = self.updateThresh(a, self.alpha)
            else:
                # thresholds as lowest weights
                self.thresholds[a] = self.advertisers[a][0][0]
            # print("\tthreshold: " + str(self.thresholds[a]))
        end = time.time()
        self.runtime = end - start
        
        return self.advertisers
    
    def getRobAndCon(self):
        if self.ALG == 0:
            print("Run Algorithm 1 first.")
        if self.OPT == 0:
            self.OptimumSolution(self.PRDPrimal())
        # print("ALG: " + str(self.ALG) + ", OPT: " + str(self.OPT) + ", PRD: " + str(self.PRD))
        robustness = self.ALG/self.OPT
        consistency = self.ALG/self.PRD
        return (robustness, consistency)