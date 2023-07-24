from OnlineAdAllocWithPred import OnlineAdAllocWithPred
from SyntheticData import SyntheticData
import time

class TestOnlineAdAllocWithPred:

    def __init__(self, b, t, w, predictor = 1, epsilon = -1, threshMethod = 1):
        self.OAAWP = OnlineAdAllocWithPred(b, t, w, predictor, epsilon, threshMethod)
        self.predictor = predictor
        self.runtime = 0

    def runAlgorithm1(self, alpha):
        start = time.time()
        self.res = self.OAAWP.algorithm1(alpha)
        end = time.time()
        self.runtime = end - start

    def runPRD(self, epsilon = -1):
        return self.OAAWP.PRDDual(epsilon)
    
    def printResults(self, results):
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
        print("Consistency: " + str(self.OAAWP.consistency) + ", Robustness: " + str(self.OAAWP.robustness))

    def printRuntime(self):
        print("Runtime: " + str(self.OAAWP.runtime + self.OAAWP.LPRuntime) + " s")
        if self.predictor == 1:
            print("LP Runtime: " + str(self.OAAWP.LPRuntime) + " s")

    def runAllTests(self, alpha, results):
        self.runAlgorithm1(alpha)
        self.printResults(results)
        self.printDummyAdvertiser(results)
        self.printThresholds(results)
        self.printRobustnessAndConsistency()
        self.printRuntime()
    
if __name__ == "__main__":
    # # manual tests
    # print("---------- manual test 1 ----------")
    # test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]], 1)
    # test1.runAllTests(1, True)

    # # small synthetic tests
    # print("---------- small synthetic test 1 ----------")
    # sData1 = SyntheticData(10, 10, 10)
    # weights1 = sData1.expDistMat()
    # sData1.gausDistList()
    # budgets1 = sData1.sampleBudgets(5, 20)
    # impressions1 = sData1.sampleImps()
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    # sTest1.runAllTests(4, False)
    # sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 2, 10)
    # sTest1.runAllTests(4, False)

    # print("---------- small synthetic test 2 ----------")
    # sData2 = SyntheticData(10, 20, 10)
    # weights2 = sData2.expDistMat()
    # sData2.gausDistList()
    # budgets2 = sData2.sampleBudgets(10, 40)
    # impressions2 = sData2.sampleImps()
    # sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2)
    # sTest2.runAllTests(2, False)

    # print("---------- small synthetic test 3 ----------")
    # sData3 = SyntheticData(10, 30, 10)
    # weights3 = sData3.expDistMat()
    # sData3.gausDistList()
    # budgets3 = sData3.sampleBudgets(15, 60)
    # impressions3 = sData3.sampleImps()
    # sTest3 = TestOnlineAdAllocWithPred(budgets3, impressions3, weights3)
    # sTest3.runAllTests(2, False)

    # print("---------- small synthetic test 4 ----------")
    # sData4 = SyntheticData(5, 30, 10)
    # weights4 = sData4.expDistMat()
    # sData4.gausDistList()
    # budgets4 = sData4.sampleBudgets(30, 120)
    # impressions4 = sData4.sampleImps()
    # sTest4 = TestOnlineAdAllocWithPred(budgets4, impressions4, weights4)
    # sTest4.runAllTests(1, False)

    # print("---------- small synthetic test 5 ----------")
    # sData5 = SyntheticData(10, 40, 10, 5)
    # weights5 = sData5.expDistMat()
    # sData5.gausDistList()
    # budgets5 = sData5.sampleBudgets(20, 80)
    # impressions5 = sData5.sampleImps()
    # sTest5 = TestOnlineAdAllocWithPred(budgets5, impressions5, weights5)
    # sTest5.runAllTests(2, False)

    # big synthetic tests
    print("---------- big synthetic test 1 ----------")
    sData6 = SyntheticData(10, 100, 10)
    weights6 = sData6.expDistMat()
    sData6.gausDistList()
    budgets6 = sData6.sampleBudgets(50, 400)
    impressions6 = sData6.sampleImps()
    sTest6 = TestOnlineAdAllocWithPred(budgets6, impressions6, weights6, 2, 50)
    sTest6.runAllTests(1.5, False)

    # print("---------- big synthetic test 2 ----------")
    # sData7 = SyntheticData(10, 100, 25)
    # weights7 = sData7.expDistMat()
    # sData7.gausDistList()
    # budgets7 = sData7.sampleBudgets(125, 500)
    # impressions7 = sData7.sampleImps()
    # sTest7 = TestOnlineAdAllocWithPred(budgets7, impressions7, weights7)
    # sTest7.runAllTests(2, False)

    # print("---------- big synthetic test 3 ----------")
    # sData8 = SyntheticData(10, 50, 100)
    # weights8 = sData8.expDistMat()
    # sData8.gausDistList()
    # budgets8 = sData8.sampleBudgets(250, 1000)
    # impressions8 = sData8.sampleImps()
    # sTest8 = TestOnlineAdAllocWithPred(budgets8, impressions8, weights8)
    # sTest8.runAllTests(1, False)

    # print("---------- big synthetic test 4 ----------")
    # sData9 = SyntheticData(500, 500, 100)
    # weights9 = sData9.expDistMat()
    # sData9.gausDistList()
    # budgets9 = sData9.sampleBudgets(50, 400)
    # impressions9 = sData9.sampleImps()
    # sTest9 = TestOnlineAdAllocWithPred(budgets9, impressions9, weights9)
    # sTest9.runAllTests(1.25, False)

    # print("---------- big synthetic test 5 ----------")
    # sData10 = SyntheticData(500, 1000, 100)
    # weights10 = sData10.expDistMat()
    # sData10.gausDistList()
    # budgets10 = sData10.sampleBudgets(50, 400)
    # impressions10 = sData10.sampleImps()
    # sTest10 = TestOnlineAdAllocWithPred(budgets10, impressions10, weights10)
    # sTest10.runAllTests(1.5, False)

    