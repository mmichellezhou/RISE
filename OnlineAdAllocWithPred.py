from cvxopt import matrix, solvers

class OnlineAdAllocWithPred:

    def __init__(self, b, t, w):
        self.budgets = b
        self.impressions = t
        self.weights = w
        self.a = 0
        self.numAdvertisers = len(self.weights)
        self.numImpressions = len(self.impressions)
        self.advertisers = [[] for i in range(self.numAdvertisers)]
        self.dummyAdvertiser = []
        self.thresholds = [0] * self.numAdvertisers
    
    def maxIndex(self, list):
        max = 0
        maxIndex = 0
        for i in range(len(list)):
            if list[i] > max:
                max = list[i]
                maxIndex = i
        return maxIndex

    # dual
    # def PRD(self):
    #     A = [[0] * (self.numAdvertisers * self.numImpressions + self.numAdvertisers + self.numImpressions) for i in range(self.numAdvertisers + self.numImpressions)]
    #     b = [0] * (self.numAdvertisers * self.numImpressions + self.numAdvertisers + self.numImpressions)
    #     c = [0] * (self.numAdvertisers + self.numImpressions)
    #     c[:self.numAdvertisers] = [float(B) for B in self.budgets]
    #     c[self.numAdvertisers:] = [1.0] * self.numImpressions
    #     for i in range(self.numAdvertisers):
    #         for j in range(self.numImpressions):
    #             A[i][i * self.numImpressions + j] = -1.0
    #             A[i][self.numAdvertisers * self.numImpressions + i] = -1.0
    #             A[self.numAdvertisers + j][i * self.numImpressions + j] = -1.0
    #             A[self.numAdvertisers + j][self.numAdvertisers * self.numImpressions + self.numAdvertisers + j] = -1.0
    #             b[i * self.numImpressions + j] = -1.0 * self.weights[i][j]
    #     # print("A: " + str(A))
    #     # print("b: " + str(b))
    #     # print("c: " + str(c))
    #     sol = solvers.lp(matrix(c), matrix(A), matrix(b))
    #     thresholds = sol['x'][:self.numAdvertisers]
    #     discGains = sol['x'][self.numAdvertisers:]
    #     # print(thresholds)
    #     # print(discGains)
    #     return

    # primal
    def PRD(self):
        # print("start PRD")
        A = [[0.0] * (self.numAdvertisers + self.numImpressions + self.numAdvertisers * 
                    self.numImpressions) for i in range(self.numAdvertisers * self.numImpressions)]
        b = [0.0] * (self.numAdvertisers + self.numImpressions + self.numAdvertisers * 
                    self.numImpressions)
        c = [0.0] * (self.numAdvertisers * self.numImpressions)

        b[:self.numAdvertisers] = [float(B) for B in self.budgets]
        b[self.numAdvertisers : self.numAdvertisers + self.numImpressions] = [1.0] * self.numImpressions
        for i in range(self.numAdvertisers):
            for j in range(self.numImpressions):
                index = i * self.numImpressions + j
                A[index][i] = 1.0
                A[index][self.numAdvertisers + j] = 1.0
                A[index][self.numAdvertisers + self.numImpressions + i * self.numImpressions + j] = -1.0
                c[index] = -1.0 * self.weights[i][self.impressions[j][0]]
        
        sol = solvers.lp(matrix(c), matrix(A), matrix(b))
        res = [[] for i in range(self.numAdvertisers)]
        for i in range(self.numImpressions):
            ws = [sol['x'][j * self.numImpressions + i] for j in range(self.numAdvertisers)]
            res[self.maxIndex(ws)].append(i)
        # print("end PRD")
        return res

        # sols = []
        # res = [[] for i in range(self.numAdvertisers)] 
        # for i in range(self.numAdvertisers):
        #     print("A: " + str(A))
        #     print("b: " + str(matrix(b)))
        #     print("c: " + str(matrix(c[i])))
        #     sol = solvers.lp(matrix(c[i]), matrix(A), matrix(b))
        #     sols.append(sol)
        #     # for j in range(len(sol['x'])):
        #     #     print(sol['x'][j])
        # for i in range(self.numImpressions):
        #     sol = [sols[j][i] for j in range(self.numAdvertisers)]
        #     print(sol)
        # return res

    def getPRD(self, i):
        res = self.PRD()
        for j in range(len(res)):
            r = res[j]
            if i in r:
                return j
        print("PRD not found")
        return
    
    def maxAdvDiscGain(self, t):
        max = 0
        maxAdv = 0
        for adv in range(self.numAdvertisers):
            discGain = self.weights[adv][t[0]] - self.thresholds[adv]
            if (discGain > max):
                max = discGain
                maxAdv = adv
        return maxAdv
    
    def updateThresh(self, adv, trust):
        B = self.budgets[adv]
        e = (1 + 1 / B) ** B
        sum = 0
        i = 1
        for t in self.advertisers[adv]:
            sum += self.weights[adv][t[0]] * e ** ((trust * (B - i)) / B)
            i += 1
        return (e ** (trust/B) - 1) * sum / (e ** trust - 1)
    
    # O(n) time complexity
    # inserts impression t into advertiser adv's list sorted by increasing weight
    # TODO make more efficient
    def insert(self, adv, t):
        list = self.advertisers[adv]
        if len(list) == 0:
            list.append(t)
        elif len(list) == 1 and self.weights[adv][list[0][0]] < self.weights[adv][t[0]]:
            list.append(t)
        else:
            index = len(list)
            for i in range(len(list)):
                if self.weights[adv][list[i][0]] > self.weights[adv][t[0]]:
                    index = i
                    break
            list.insert(index, t)
        return

    def algorithm1(self, trust):
        B = min(self.budgets)
        e = (1 + 1 / B) ** B
        self.a = B * (e ** (trust / B) - 1)
        for i in range(len(self.impressions)):
            t = self.impressions[i]
            advPRD = self.getPRD(i)
            advEXP = self.maxAdvDiscGain(t)
            discGainPRD = self.weights[advPRD][t[0]] - self.thresholds[advPRD]
            discGainEXP = self.weights[advEXP][t[0]] - self.thresholds[advEXP]
            # if discounted gain is negative, continue to next impression
            if (discGainPRD < 0 and discGainEXP < 0):
                self.dummyAdvertiser.append(t)
                continue
            if (self.a * discGainPRD >= discGainEXP):
                adv = advPRD
            else:
                adv = advEXP
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[adv]) == self.budgets[adv]:
                self.dummyAdvertiser.append(self.advertisers[adv].pop(0))
            self.insert(adv, t)
            self.thresholds[adv] = self.updateThresh(adv, trust)
        return self.advertisers
    
    def getDummyAdvertiser(self):
        return self.dummyAdvertiser

    def getThresholds(self):
        return self.thresholds