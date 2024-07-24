from OnlineAdAllocWithPred import OnlineAdAllocWithPred
from SyntheticData import SyntheticData
import time
import math
import matplotlib.pyplot as plt
import numpy as np

class TestOnlineAdAllocWithPred:
    """
    A class used to generate sythetic data.

    ...

    Attributes
    ----------
    OAAWP : OnlineAdAllocWithPred
        The Online Ad Allocation With Predictions object
    predictor : int
        The predictor type in which 0 is OPT, 1 is DualBase, 2 is PreviousDay, 3 is OPT Random Corruption, and 4 is OPT Biased Corruption
    runtime : float
        The runtime.
    
    Methods
    -------
    listToMat(list)
        Converts a list of numAdvs*numImpressions elements to a matrix of numAdvs rows and numImpressions columns
    
    expDistMat()
        Generates a matrix from an exponential distribution
    
    gausDistList(sd=10)
        Generates a list from a gaussian distribution
    
    sampleBudgets(min, max, oneEqualToMin=False)
        Generates a list of sample budgets in which each index corresponds to an advertiser
    
    sampleImps()
        Generates a list of sample impressions that are defined by (impression type, display time)
    """

    def __init__(self, b, t, w, predictor = 0, epsilon = -1, prevImps = [], p = -1):
        self.OAAWP = OnlineAdAllocWithPred(b, t, w, predictor, epsilon, prevImps, p)
        self.predictor = predictor
        self.runtime = 0

    def runAlgorithm1(self, alpha, threshMethod):
        """
        Test runs Algorithm 1 on OAAWP based on the robustness-consistency trade-off, alpha, and the threshold method, threshMethod.

        Parameters
        ----------
        alpha : int
            The robustness-consistency trade-off.
        threshMethod : int
            The method for updating the thresholds in which 0 is exponential averaging and 1 is mimimum value.
        """
        start = time.time()
        self.res = self.OAAWP.Algorithm1(alpha, threshMethod)
        end = time.time()
        self.runtime = end - start
        
    def runRandomMixture(self, alpha, q, threshMethod):
        """
        Test runs the random mixture algorithm on OAAWP based on the robustness-consistency trade-off, alpha; the worst-case probability, q; and the threshold method, threshMethod.

        Parameters
        ----------
        alpha : int
            The robustness-consistency trade-off.
        q : int
            The worst-case probability.
        threshMethod : int
            The method for updating the thresholds in which 0 is exponential averaging and 1 is mimimum value.
        """
        start = time.time()
        self.res = self.OAAWP.randomMixture(alpha, q, threshMethod)
        end = time.time()
        self.runtime = end - start
    
    def reset(self):
        """
        Resets attributes of OAAWP for a new test run.
        """
        self.OAAWP.reset()

    def getRobAndCon(self):
        """
        Returns the robustness and consistency of OAAWP.
        """
        if self.OAAWP.ALG == 0:
            print("Run algorithm 1 first.")
        if self.OAAWP.OPT == 0:
            self.OAAWP.OptimumSolution(self.OAAWP.PrimalLP())
        print("ALG: " + str(self.OAAWP.ALG) + ", OPT: " + str(self.OAAWP.OPT) + ", PRD: " + str(self.OAAWP.PRD))
        robustness = self.OAAWP.ALG/self.OAAWP.OPT
        consistency = self.OAAWP.ALG/self.OAAWP.PRD
        return (robustness, consistency)
    
    def checkLP(self):
        """
        Prints the objective function values of the Optimum Solution and Dual Base predictors.
        """
        sumOPT = 0
        for i in range(self.OAAWP.numAdvertisers):
            for j in range(self.OAAWP.numImpressions):
                sumOPT += self.OAAWP.weights[i][self.OAAWP.impressions[j][0]] * self.OAAWP.solOPT[i * self.OAAWP.numImpressions + j]
        print("OPT objective function value: " + str(sumOPT))
        sumDualBase = 0
        for i in range(len(self.OAAWP.solDualBase[0])):
            sumDualBase += self.OAAWP.budgets[i] * self.solDualBase[0][i]
        for i in range(len(self.OAAWP.solDualBase[1])):
            sumDualBase += self.OAAWP.solDualBase[1][i]
        print("DualBase objective function value: " + str(sumDualBase))
    
    def printResults(self, results):
        """
        Prints the 

        Parameters
        ----------
        results : bool
            True for , False for
        """
        if results:
            for i in range(len(self.res)):
                print("Advertiser " + str(i) + ": " + str(self.res[i]))
        else:
            print("Results: " + str([len(adv) for adv in self.res]) + " impressions")

    def printDummyAdvertiser(self, results):
        if results:
            print("Dummy advertiser: " + str(self.OAAWP.dummyAdvertiser))
        else:
            print("Dummy advertiser: " + str(len(self.OAAWP.dummyAdvertiser)) + " impressions")

    def printThresholds(self, results):
        if results:
            print("Thresholds: " + str(self.OAAWP.thresholds))

    def printRobustnessAndConsistency(self):
        print("Robustness: " + str(self.getRobAndCon()[0]) + ", Consistency: " + str(self.getRobAndCon()[1]))

    def printRuntime(self):
        print("Runtime: " + str(self.OAAWP.runtime + self.OAAWP.LPRuntime) + " s")
        print("LP Runtime: " + str(self.OAAWP.LPRuntime) + " s")

    def showRobVsCon(self):
        figure, axis = plt.subplots(1, 3)
        for B in [2, 10, 20, float('inf')]:
            alphas = []
            consistencies = []
            robustnesses = []
            e = (1 + 1 / B) ** B
            for alpha in range(3, 30):
                alpha /= 3
                if B == float('inf'):
                    consistency = (1 + 1 / (math.e ** alpha - 1) * max((math.e ** alpha - (math.e ** alpha - 1) / alpha) / alpha, alpha)) ** (-1)
                    robustness = (math.e ** alpha - 1) / (alpha * math.e ** alpha)
                else:
                    alphaB = B * (e ** (alpha / B) - 1)
                    consistency = (1 + 1 / (e ** alpha - 1) * max((1 / alphaB) * (e ** alpha - (e ** alpha - 1) / alphaB), math.log(e ** alpha))) ** (-1)
                    robustness = (e ** alpha - 1)/(B * e ** alpha * (e ** (alpha/B) - 1))
                alphas.append(alpha)
                consistencies.append(consistency)
                robustnesses.append(robustness)
            axis[0].plot(alphas, consistencies)
            axis[1].plot(alphas, robustnesses)
            axis[2].plot(consistencies, robustnesses, label = "B = " + str(B))

        axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
        axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')
        axis[2].set_aspect(1.0/axis[2].get_data_ratio(), adjustable='box')

        axis.flat[0].set(xlabel = "\u03B1", ylabel = "Consistency")
        axis.flat[1].set(xlabel = "\u03B1", ylabel = "Robustness")
        axis.flat[2].set(xlabel = "Consistency", ylabel = "Robustness")

        axis[2].legend()

        plt.show()

    def runAllTests(self, alpha, q = -1, threshMethod = 0, results = False, print = True):
        if q == -1:
            self.runAlgorithm1(alpha, threshMethod)
        else:
            self.runRandomMixture(alpha, q, threshMethod)
        if print:
            self.printResults(results)
            self.printDummyAdvertiser(results)
            self.printThresholds(results)
            self.printRobustnessAndConsistency()
            self.printRuntime()
    
if __name__ == "__main__":
    # manual tests
    # print("---------- manual test 1 ----------")
    # test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]])
    # test1.runAllTests(1)
    
    # synthetic tests
    # print("---------- synthetic test 1 ----------")
    # sData1 = SyntheticData(4, 6, 4)
    # weights1 = sData1.expDistMat()
    # sData1.gausDistList()
    # budgets1 = sData1.sampleBudgets(3, 12)
    # impressions1 = sData1.sampleImps()
    # # OPT
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    # sTest1.runAllTests(1)
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 0, -1, [], -1, 1)
    # sTest1.runAllTests(1)
    # # Dual Base
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 1)
    # sTest1.runAllTests(1)
    # # Previous Day
    # prevData1 = SyntheticData(10, 5, 10)
    # prevData1.gausDistList()
    # prevImps = prevData1.sampleImps()
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 2, -1, prevImps)
    # sTest1.runAllTests(1)

    # print("---------- synthetic test 2 ----------")
    # sData2 = SyntheticData(5, 10, 5)
    # weights2 = sData2.expDistMat()
    # sData2.gausDistList()
    # budgets2 = sData2.sampleBudgets(10, 40)
    # impressions2 = sData2.sampleImps()
    # sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2)
    # sTest2.runAllTests(5, False)

    # print("---------- synthetic test 3 ----------")
    # sData3 = SyntheticData(20, 20, 20)
    # weights3 = sData3.expDistMat()
    # sData3.gausDistList()
    # budgets3 = sData3.sampleBudgets(10, 40)
    # impressions3 = sData3.sampleImps()
    # sTest3 = TestOnlineAdAllocWithPred(budgets3, impressions3, weights3)
    # sTest3.runAllTests(2, False)

    # print("---------- synthetic test 4 ----------")
    # sData4 = SyntheticData(5, 30, 10)
    # weights4 = sData4.expDistMat()
    # sData4.gausDistList()
    # budgets4 = sData4.sampleBudgets(30, 120)
    # impressions4 = sData4.sampleImps()
    # sTest4 = TestOnlineAdAllocWithPred(budgets4, impressions4, weights4)
    # sTest4.runAllTests(1, False)

    # print("---------- synthetic test 5 ----------")
    # sData5 = SyntheticData(10, 40, 10, 5)
    # weights5 = sData5.expDistMat()
    # sData5.gausDistList()
    # budgets5 = sData5.sampleBudgets(20, 80)
    # impressions5 = sData5.sampleImps()
    # sTest5 = TestOnlineAdAllocWithPred(budgets5, impressions5, weights5)
    # sTest5.runAllTests(2, False)

    # graphs
    # figure, axis = plt.subplots(1, 3)
    # for B in [2, 10, 20, float('inf')]:
    #     alphas = []
    #     consistencies = []
    #     robustnesses = []
    #     e = (1 + 1 / B) ** B
    #     for alpha in range(3, 30):
    #         alpha /= 3
    #         if B == float('inf'):
    #             consistency = (1 + 1 / (math.e ** alpha - 1) * max((math.e ** alpha - (math.e ** alpha - 1) / alpha) / alpha, alpha)) ** (-1)
    #             robustness = (math.e ** alpha - 1) / (alpha * math.e ** alpha)
    #         else:
    #             alphaB = B * (e ** (alpha / B) - 1)
    #             consistency = (1 + 1 / (e ** alpha - 1) * max((1 / alphaB) * (e ** alpha - (e ** alpha - 1) / alphaB), math.log(e ** alpha))) ** (-1)
    #             robustness = (e ** alpha - 1)/(B * e ** alpha * (e ** (alpha/B) - 1))
    #         alphas.append(alpha)
    #         consistencies.append(consistency)
    #         robustnesses.append(robustness)
    #     axis[0].plot(alphas, consistencies)
    #     axis[1].plot(alphas, robustnesses)
    #     axis[2].plot(consistencies, robustnesses, label = "B = " + str(B))

    # axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
    # axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')
    # axis[2].set_aspect(1.0/axis[2].get_data_ratio(), adjustable='box')

    # axis.flat[0].set(xlabel = "\u03B1", ylabel = "Consistency")
    # axis.flat[1].set(xlabel = "\u03B1", ylabel = "Robustness")
    # axis.flat[2].set(xlabel = "Consistency", ylabel = "Robustness")

    # axis[2].legend()

    # plt.show()

    sData = SyntheticData(10, 10, 10, 1.5)
    weights1 = sData.expDistMat()
    sData.gausDistList()
    budgets1 = sData.sampleBudgets(5, 40)
    impressions1 = sData.sampleImps()

    prevData1 = SyntheticData(10, 10, 8, 1.5)
    prevData1.gausDistList()
    prevImps = prevData1.sampleImps()
    
    sTestOPT = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    sTestDualBase = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 1)
    sTestPrevDay = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 2, -1, prevImps)
    sTestOPTRC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 3, -1, [], 0.9)
    sTestOPTBC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 4, -1, [], 0.9)

    labels = ["OPT", "DualBase", "PreviousDay", "OPT random corruption p = 0.9", "OPT biased corruption p = 0.9", "", "", "", "", "", "Worst-Case Baseline"]
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:red", "black"]
    linestyles = ["solid", "solid", "solid", "solid", "solid", "dashed", "dashed", "dashed", "dashed", "dashed", "solid"]

    # shows robustness-consistency trade-off graphs
    sTestOPT.showRobVsCon()

    figure, axis = plt.subplots(1, 2)

    numPRD = 11

    consistencies = []
    robustnesses = []
    
    for PRD in range(numPRD):
        consistencies = []
        robustnesses = []
        prevImps1 = []

        if PRD in [0, 5, 10]:
            sTest = sTestOPT
        elif PRD in [1, 6]:
            sTest = sTestDualBase
        elif PRD in [2, 7]:
            sTest = sTestPrevDay
        elif PRD in [3, 8]:
            sTest = sTestOPTRC
        else:
            sTest = sTestOPTBC

        for i in range(5):
            alphas = []
            consistency = []
            robustness = []

            for alpha in range(2, 11):
                alpha /= 2
                if PRD < 5:
                    sTest.runAllTests(alpha, -1)
                elif PRD == 10:
                    sTest.runAllTests(alpha, 1, False, False)
                else:
                    sTest.runAllTests(alpha, 1 / alpha, False, False)
                alphas.append(alpha)
                robustness.append(sTest.getRobAndCon()[0])
                consistency.append(sTest.getRobAndCon()[1])
            robustnesses.append(robustness)
            consistencies.append(consistency)

        print("robustnesses: " + str(robustnesses) + ", consistencies: " + str(consistencies))

        avgRob = np.array([sum(rob) / len(rob) for rob in zip(*robustnesses)])
        avgCon = np.array([sum(con) / len(con) for con in zip(*consistencies)])

        print(avgRob, avgCon)

        errRob = max(max([max(rob) for rob in robustnesses]) - sum(avgRob) / len(avgRob), sum(avgRob) / len(avgRob) - min([min(rob) for rob in robustnesses]))
        errCon = max(max([max(con) for con in consistencies]) - sum(avgCon) / len(avgCon), sum(avgCon) / len(avgCon) - min([min(con) for con in consistencies]))

        print(errRob, errCon)

        if PRD != 10:
            axis[0].plot(alphas, avgCon, color = colors[PRD], linestyle = linestyles[PRD])        
        axis[1].plot(alphas, avgRob, color = colors[PRD], linestyle = linestyles[PRD], label = labels[PRD])
        
        if PRD < 5:
            axis[0].fill_between(np.array(alphas), avgCon - errCon, avgCon + errCon, facecolor = colors[PRD], edgecolor="none", alpha=.25)
            axis[1].fill_between(np.array(alphas), avgRob - errRob, avgRob + errRob, facecolor = colors[PRD], edgecolor="none", alpha=.25)
    
    axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
    axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')

    axis.flat[0].set(xlabel = "\u03B1", ylabel = "Consistency")
    axis.flat[1].set(xlabel = "\u03B1", ylabel = "Robustness")

    axis[1].legend()

    plt.show()