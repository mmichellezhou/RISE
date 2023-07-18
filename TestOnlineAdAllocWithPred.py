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
    # old manual tests
    # print("---------- test 1 ----------")
    # test1 = TestOnlineAdAllocWithPred([2, 1], [[2, 1, 10], [100, 2, 1]])
    # test1.runAllTests(1)
    # print("---------- test 2 ----------")
    # test2 = TestOnlineAdAllocWithPred([1, 1], [[0, 2, 50], [3, 10, 30]])
    # test2.runAllTests(1)
    # print("---------- test 3 ----------")
    # test3 = TestOnlineAdAllocWithPred([1, 1], [[0, 2, 50], [3, 10, 30]])
    # test3.runAllTests(2)
    # print("---------- test 4 ----------")
    # test4 = TestOnlineAdAllocWithPred([3, 2], [[2, 10, 4, 0], [1, 16, 0, 9]])
    # test4.runAllTests(1)
    # print("---------- test 5 ----------")
    # test5 = TestOnlineAdAllocWithPred([2, 1, 3, 1], [[5, 50, 7, 4], [1, 2, 8, 4], [10, 0, 9, 30], [3, 0, 13, 1]])
    # test5.runAllTests(3)

    # new manual tests
    print("---------- manual test 1 ----------")
    test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]])
    test1.runAllTests(1)

    # synthetic tests
    print("---------- synthetic test 1 ----------")
    sData1 = SyntheticData(10, 10, 10, 10)
    weights1 = sData1.expDistMat()
    sData1.gausDistList(10)
    budgets1 = sData1.sampleBudgets(5, 20)
    impressions1 = sData1.sampleImps()
    sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    sTest1.runAllTests(1)

    # print("---------- synthetic test 2 ----------")
    # sData2 = SyntheticData(10, 10, 20, 10)
    # weights2 = sData2.expDistMat()
    # sData2.gausDistList(10)
    # budgets2 = sData2.sampleBudgets(5, 40)
    # impressions2 = sData2.sampleImps()
    # sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2)
    # sTest2.runAllTests(2)

    
    