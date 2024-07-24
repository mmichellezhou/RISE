import numpy as np
from cvxopt import matrix, spmatrix, solvers
import time
import random
import math

class OnlineAdAllocWithPred:
    """
    A class used to represent an Online Ad Allocation with Predictions.

    ...

    Attributes
    ----------
    budgets : list
        A list of advertiser budgets in which each index corresponds to an advertiser
    impressions : list
        A list of impressions sorted by display time
    weights : matrix
        A matrix of weights in which the rows are advertisers and the columns are impression types
    predictor : int
        The predictor type in which 0 is OPT, 1 is DualBase, 2 is PreviousDay, 3 is OPT Random Corruption, and 4 is OPT Biased Corruption
    epsilon : int
        The epsilon for the DualBase predictor
    prevImps : list
        A list of previous impressions sorted by display time
    p : int
        The p-fraction of corrupted allocations
    numAdvertisers : int
        The number of advertisers
    numImpressions : int
        The number of impressions
    advertisers : matrix
        The matrix of allocations in which the rows are advertisers and the columns are impressions
    dummyAdvertiser : list
        A list of discarded impressions
    thresholds : list
        A list of thresholds in which each index corresponds to an advertiser
    runtime : int
        The runtime
    LPRuntime : int
        The runtime of the linear program(s)
    ALG : int
        The objective value of Algorithm 1
    OPT : int
        The objective value of the Optimum Solution
    PRD : int 
        The objective value of the predictor

    Methods
    -------
    predict()
        Makes the predictions based on the predictor

    DualLP()
        Solves the dual linear program

    DualBase(thresholds)
        Returns the predictions based on the Dual Base predictor

    PrimalLP(p=-1)
        Solves the primal linear program for a (1-p)-fraction of allocations

    OptimumSolution()
        Returns the predictions based on the Optimum Solution predictor
    
    getPRD(i)
        Returns the predicted advertiser for impression of index, i

    maxADiscGain(t, thresholds)
        Returns the index of the advertiser that maximizes the discounted gain of impression, t

    updateThresh(a, alpha, threshMethod)
        Updates the thresholds of the advertiser of index a based on the robustness-consistency trade-off, alpha, and the threshold method, threshMethod

    insert(a, t)
        Inserts impression t into the advertiser of index a's list of allocations by increasing weight
    
    Algorithm1(alpha, threshMethod=0)
        Runs Algorithm 1 based on the robustness-consistency trade-off, alpha, and the threshold method, threshMethod

    randomMixture(alpha, q, threshMethod=0)
        Runs the random mixture algorithm based on the robustness-consistency trade-off, alpha; the worst-case probability, q; and the threshold method, threshMethod
    
    corrupt(rand=True)
        Corrupts a p-fraction of allocations under random corruption if rand is True and biased corruption otherwise
        
    reset()
        Resets attributes for a new run
    """

    def __init__(self, b, t, w, predictor = 0, epsilon = -1, prevImps = [], p = -1):
        self.budgets = b
        self.impressions = t
        self.weights = w
        self.predictor = predictor
        self.epsilon = epsilon
        self.prevImps = prevImps
        self.p = p
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
        self.predict()

    def predict(self):
        """
        Makes the predictions based on the predictor.

        Raises
        ------
        Exception
            If predictor is invalid.
        """
        if self.predictor == 0:
            print("OPT")
            self.solOPT = self.PrimalLP()
            self.predictions = self.OptimumSolution(self.solOPT)
        elif self.predictor == 1:
            print("DualLP Base")
            if self.epsilon == -1:
                self.epsilon = 0.1
            self.sampleImpsD = random.sample(self.impressions, math.ceil(self.numImpressions * self.epsilon))
            self.solDualBase = self.DualLP()
            self.predictions = self.DualBase()
        elif self.predictor == 2:
            print("Previous Day")
            if not len(self.prevImps):
                print("Previous Day requires an input of previous impressions.")
            self.sampleImpsD = self.prevImps
            self.solDualBase = self.DualLP()
            self.predictions = self.DualBase()
        elif self.predictor in [3, 4]:
            print("Optimum Solution Random Corruption") if self.predictor == 3 else print("Optimum Solution Biased Corruption")
            if self.p == -1:
                self.p = 0.9
            self.predictions = self.OptimumSolution(self.PrimalLP(self.p))
            if self.predictor == 3:
                self.corrupt()
            else:
                self.corrupt(False)
        raise Exception("Invalid predictor.")

    def DualLP(self):
        """
        Solves the dual linear program.
        """
        self.numSampleImpsD = len(self.sampleImpsD)

        x = np.array([])
        I = np.array([])
        J = np.array([])
        b = np.zeros(self.numAdvertisers * self.numSampleImpsD + self.numAdvertisers + self.numSampleImpsD)
        c = np.zeros(self.numAdvertisers + self.numSampleImpsD)
        
        for i in range(self.numAdvertisers):
            for j in range(self.numSampleImpsD):
                index = i * self.numSampleImpsD + j
                if i == 0:
                    x = np.append(x, [-1.0] * 3)
                    I = np.append(I, [index, index, self.numAdvertisers * self.numSampleImpsD + self.numAdvertisers + j])
                    J = np.append(J, [i, self.numAdvertisers + j, self.numAdvertisers + j])
                else:
                    x = np.append(x, [-1.0] * 2)
                    I = np.append(I, [index, index])
                    J = np.append(J, [i, self.numAdvertisers + j])
                b[index] = -1.0 * self.weights[i][self.sampleImpsD[j][0]]
                c[self.numAdvertisers + j] = 1.0
            x = np.append(x, [-1.0])
            I = np.append(I, [self.numAdvertisers * self.numSampleImpsD + i])
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

    def DualBase(self):
        """
        Returns the predictions based on the Dual Base predictor.
        """
        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            a = self.maxADiscGain(self.impressions[i], self.solDualBase[0])
            res[a].append(i)
            self.PRD += self.weights[a][self.impressions[i][0]]
        
        return res

    def PrimalLP(self, p = -1):
        """
        Solves the primal linear program for a (1-p)-fraction of allocations.

        Parameters
        ----------
        p : int
            The p-fraction of corrupted allocations.
        """
        if p != -1:
            self.numCorrupted = math.ceil(self.numImpressions * p)
            self.sampleImpsP = self.impressions[:self.numImpressions - self.numCorrupted]
        else:
            self.sampleImpsP = self.impressions
        self.numSampleImpsP = len(self.sampleImpsP)
        x = np.array([])
        I = np.array([])
        J = np.array([])
        b = np.zeros(self.numAdvertisers + self.numSampleImpsP + self.numAdvertisers * self.numSampleImpsP)
        c = np.zeros(self.numAdvertisers * self.numSampleImpsP)

        b[self.numAdvertisers : self.numAdvertisers + self.numSampleImpsP] = 1.0
        for i in range(self.numAdvertisers):
            for j in range(self.numSampleImpsP):
                index = i * self.numSampleImpsP + j
                x = np.append(x, [1.0, 1.0, -1.0])
                I = np.append(I, [i, self.numAdvertisers + j, self.numAdvertisers + self.numSampleImpsP + i * self.numSampleImpsP + j])
                J = np.append(J, [index] * 3)
                c[index] = -1.0 * self.weights[i][self.sampleImpsP[j][0]]
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
        """
        Returns the predictions based on the Optimum Solution predictor.

        Parameters
        ----------
        sol : list
            The solution of the linear program.
        """
        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numSampleImpsP):
            ws = [sol[a * self.numSampleImpsP + i] for a in range(self.numAdvertisers)]
            a = ws.index(max(ws))
            res[a].append(i)
            self.OPT += self.weights[a][self.sampleImpsP[i][0]]
        
        if self.predictor == 0:
            self.PRD = self.OPT
        elif self.predictor in [3, 4] and self.PRD == 0:
            self.PRD = self.OPT
            self.OPT = 0
        
        return res

    def getPRD(self, i):
        """
        Returns the predicted advertiser for impression of index, i.

        Parameters
        ----------
        i : int
            The impression of index, i.

        Raises
        ------
        Exception
            If prediction is not found for impression of index, i.
        """
        for j in range(self.numAdvertisers):
            r = self.predictions[j]
            if i in r:
                return j
        raise Exception("Prediction not found for impression of index " + str(i))
    
    def maxADiscGain(self, t, thresholds):
        """
        Returns the index of the advertiser that maximizes the discounted gain of impression, t.

        Parameters
        ----------
        t : int
            The impression, t.
        thresholds : list
            The list of thresholds.
        """
        max = 0
        maxA = 0
        for a in range(self.numAdvertisers):
            discGain = self.weights[a][t[0]] - thresholds[a]
            if (discGain > max):
                max = discGain
                maxA = a
        return maxA
    
    def updateThresh(self, a, alpha, threshMethod):
        """
        Updates the thresholds of the advertiser of index a based on the robustness-consistency trade-off, alpha, and the threshold method, threshMethod.

        Parameters
        ----------
        a : int
            The advertiser of index, a.
        alpha : int
            The robustness-consistency trade-off
        threshMethod: int
            The method for updating the tresholds.
        """
        if threshMethod == 0:
            B = self.budgets[a]
            e = (1 + 1 / B) ** B
            sum = 0
            i = 1
            for t in self.advertisers[a]:
                sum += self.weights[a][t[0]] * e ** ((alpha * (B - i)) / B)
                i += 1
            return (e ** (alpha / B) - 1) * sum / (e ** alpha - 1)
        else:
            self.thresholds[a] = self.weights[a][self.advertisers[a][0][0]]
    
    # TODO make more efficient
    def insert(self, a, t):
        """
        Inserts impression t into the advertiser of index a's list of allocations by increasing weight.

        Parameters
        ----------
        a : int
            The advertiser of index, a.
        t : int
            The impression, t.
        """
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

    def Algorithm1(self, alpha, threshMethod = 0):
        """
        Runs Algorithm 1 based on the robustness-consistency trade-off, alpha, and the threshold method, threshMethod.

        Parameters
        ----------
        alpha : int
            The robustness-consistency trade-off.
        threshMethod : int
            The method for updating the thresholds in which 0 is exponential averaging and 1 is mimimum value.
        """
        self.reset()
        start = time.time()
        self.alpha = alpha
        B = min(self.budgets)
        e = (1 + 1 / B) ** B
        alpha_B = B * (e ** (alpha / B) - 1)
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
            if (alpha_B * discGainPRD >= discGainEXP):
                a = aPRD
            else:
                a = aEXP
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[a]) == self.budgets[a]:
                temp = self.advertisers[a].pop(0)
                self.dummyAdvertiser.append(temp)
                self.ALG -= self.weights[a][temp[0]]
            self.insert(a, t)
            self.ALG += self.weights[a][t[0]]

            self.thresholds[a] = self.updateThresh(a, self.alpha, threshMethod)

        end = time.time()
        self.runtime = end - start
        
        return self.advertisers
    
    def randomMixture(self, alpha, q, threshMethod = 0):
        """
        Runs the random mixture algorithm based on the robustness-consistency trade-off, alpha; the worst-case probability, q; and the threshold method, threshMethod.

        Parameters
        ----------
        alpha : int
            The robustness-consistency trade-off.
        q : int
            The worst-case probability.
        threshMethod : int
            The method for updating the thresholds in which 0 is exponential averaging and 1 is mimimum value.
        """
        self.reset()
        start = time.time()
        self.alpha = alpha
        for i in range(len(self.impressions)):
            t = self.impressions[i]
            if random.random() <= q:
                a = self.maxADiscGain(t, self.thresholds)
            else:
                a = self.getPRD(i)
            discGainEXP = self.weights[a][t[0]] - self.thresholds[a]
            # if discounted gain is negative, continue to next impression
            if (discGainEXP <= 0):
                self.dummyAdvertiser.append(t)
                continue
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[a]) == self.budgets[a]:
                temp = self.advertisers[a].pop(0)
                self.dummyAdvertiser.append(temp)
                self.ALG -= self.weights[a][temp[0]]
            self.insert(a, t)
            self.ALG += self.weights[a][t[0]]
            
            self.thresholds[a] = self.updateThresh(a, self.alpha, threshMethod)

        end = time.time()
        self.runtime = end - start
        
        return self.advertisers
        
    def corrupt(self, rand = True):
        """
        Corrupts a p-fraction of allocations under random corruption if rand is True and biased corruption otherwise.

        Parameters
        ----------
        rand : bool
            True for random corruption, False for biased corruption.
        """
        if rand:
            for i in range(self.numImpressions - self.numCorrupted, self.numImpressions):
                a = random.randint(0, self.numAdvertisers - 1)
                self.predictions[a].append(i)
                self.PRD += self.weights[a][self.impressions[i][0]]
        else:
            corrupted = np.random.permutation([i for i in range(self.numImpressions - self.numCorrupted, self.numImpressions)]).reshape(self.numAdvertisers, -1)
            for i in range(self.numAdvertisers):
                self.predictions[i].extend(corrupted[i].tolist())
                self.PRD += sum([self.weights[i][self.impressions[corrupted[i][j]][0]] for j in range(len(corrupted[i]))])
    
    def reset(self):
        """
        Resets attributes for a new run.
        """
        self.advertisers = [[] for i in range(self.numAdvertisers)]
        self.dummyAdvertiser = []
        self.thresholds = [0] * self.numAdvertisers
        self.runtime = 0
        self.ALG = 0