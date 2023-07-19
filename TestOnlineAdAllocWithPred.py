from OnlineAdAllocWithPred import OnlineAdAllocWithPred
from SyntheticData import SyntheticData
import time

class TestOnlineAdAllocWithPred:

    def __init__(self, b, t, w):
        self.OAAWP = OnlineAdAllocWithPred(b, t, w)
        self.runtime = 0

    def runAlgorithm1(self, trust):
        start = time.time()
        res = self.OAAWP.algorithm1(trust)
        end = time.time()
        self.runtime = end - start
        for i in range(len(res)):
            print("Advertiser " + str(i) + ": " + str(res[i]))

    def runPRD(self):
        return self.OAAWP.PRD()

    def printDummyAdvertiser(self):
        print("Dummy advertiser: " + str(self.OAAWP.dummyAdvertiser))

    def printThresholds(self):
        print("Thresholds: " + str(self.OAAWP.thresholds))
    
    def printRuntime(self):
        print("Runtime: " + str(self.runtime) + " s")

    def runAllTests(self, trust):
        self.runAlgorithm1(trust)
        self.printDummyAdvertiser()
        self.printThresholds()
        self.printRuntime()
    
if __name__ == "__main__":
    # manual tests
    print("---------- manual test 1 ----------")
    test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]])
    test1.runAllTests(1)

    # synthetic tests
    print("---------- synthetic test 1 ----------")
    sData1 = SyntheticData(10, 10, 10, 10)
    weights1 = sData1.expDistMat()
    sData1.gausDistList()
    budgets1 = sData1.sampleBudgets(5, 20)
    impressions1 = sData1.sampleImps()
    sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    sTest1.runAllTests(1)
    # runtime: ~10-20s

    print("---------- synthetic test 2 ----------")
    sData2 = SyntheticData(10, 10, 20, 10)
    weights2 = sData2.expDistMat()
    sData2.gausDistList()
    budgets2 = sData2.sampleBudgets(10, 40)
    impressions2 = sData2.sampleImps()
    sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2)
    sTest2.runAllTests(2)
    # runtime: ~80-120s

    print("---------- synthetic test 3 ----------")
    sData3 = SyntheticData(10, 10, 30, 10)
    weights3 = sData3.expDistMat()
    sData3.gausDistList()
    budgets3 = sData3.sampleBudgets(10, 40)
    impressions3 = sData3.sampleImps()
    sTest3 = TestOnlineAdAllocWithPred(budgets3, impressions3, weights3)
    sTest3.runAllTests(2)
    # runtime: ~300s

    print("---------- synthetic test 4 ----------")
    sData4 = SyntheticData(10, 5, 30, 10)
    weights4 = sData4.expDistMat()
    sData4.gausDistList()
    budgets4 = sData4.sampleBudgets(30, 120)
    impressions4 = sData4.sampleImps()
    sTest4 = TestOnlineAdAllocWithPred(budgets4, impressions4, weights4)
    sTest4.runAllTests(1)
    # runtime: ~110-120s

    print("---------- synthetic test 5 ----------")
    sData5 = SyntheticData(5, 10, 40, 10)
    weights5 = sData5.expDistMat()
    sData5.gausDistList()
    budgets5 = sData5.sampleBudgets(20, 80)
    impressions5 = sData5.sampleImps()
    sTest5 = TestOnlineAdAllocWithPred(budgets5, impressions5, weights5)
    sTest5.runAllTests(2)
    # runtime: ~1320s

    
    