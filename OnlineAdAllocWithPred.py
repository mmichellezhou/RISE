import bisect

class OnlineAdAllocWithPred:

    def __init__(self, b, w):
        self.budgets = b
        self.weights = w
        self.numAdvertisers = len(self.weights)
        self.numImpressions = len(self.weights[0])
        self.advertisers = [[] for adv in range(self.numAdvertisers)]
        self.dummyAdvertiser = []
        self.thresholds = [0] * self.numAdvertisers
        
    # TODO
    def PRD(self, t):
        return 0
    
    def maxAdvDiscGain(self, t):
        max = 0
        maxAdv = 0
        for adv in range(self.numAdvertisers):
            discGain = self.weights[adv][t] - self.thresholds[adv]
            if (discGain > max):
                max = discGain
                maxAdv = adv
        return maxAdv
    
    def updateThresh(self, adv, trust):
        B = self.budgets[adv]
        e = (1 + 1 / B) ** B
        # print(adv, B, e)
        sum = 0
        i = 1
        for t in self.advertisers[adv]:
            sum += self.weights[adv][t] * e ** ((trust * (B - i)) / B)
            i += 1
        return (e ** (trust/B) - 1) * sum / (e ** trust - 1)
    
    def algorithm1(self, trust):
        B = min(self.budgets)
        e = (1 + 1 / B) ** B
        a = B * (e ** (trust / B) - 1)
        # print(B, e, a)
        for t in range(self.numImpressions):
            advPRD = self.PRD(t)
            advEXP = self.maxAdvDiscGain(t)
            discGainPRD = self.weights[advPRD][t] - self.thresholds[advPRD]
            discGainEXP = self.weights[advEXP][t] - self.thresholds[advEXP]
            # if discounted gain is negative, continue to next impression
            if (discGainPRD < 0 and discGainEXP < 0):
                self.dummyAdvertiser.append(t)
                continue
            if (a * discGainPRD >= discGainEXP):
                adv = advPRD
            else:
                adv = advEXP
            # if advertiser a has reached their maximum budget, remove least valuable impression assigned to a
            if len(self.advertisers[adv]) == self.budgets[adv]:
                self.dummyAdvertiser.append(self.advertisers[adv].pop(0))
            bisect.insort(self.advertisers[adv], t)
            self.thresholds[adv] = self.updateThresh(adv, trust)
        return self.advertisers
    
    def getDummyAdvertiser(self):
        return self.dummyAdvertiser

    def getThresholds(self):
        return self.thresholds