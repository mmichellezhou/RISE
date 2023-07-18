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
            print("\tweights: " + str([self.OAAWP.weights[i][j[0]] for j in res[i]]))

    def runPRD(self):
        return self.OAAWP.PRD()

    def printDummyAdvertiser(self):
        print("Dummy advertiser: " + str(self.OAAWP.getDummyAdvertiser()))

    def printThresholds(self):
        print("Thresholds: " + str(self.OAAWP.getThresholds()))
    
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
    print("---------- test 1 ----------")
    test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]])
    test1.runAllTests(1)
    print("---------- test 2 ----------")
    test2 = TestOnlineAdAllocWithPred([4, 3, 2], [(6, 0), (6, 1), (3, 2), (1, 3), (2, 4), (1, 5), (4, 6), (5, 7)], [[3, 10, 12, 4, 9, 2, 8, 1], [7, 22, 9, 6, 4, 5, 20, 16], [3, 3, 4, 16, 19, 1, 9, 5]])
    test2.runAllTests(1.2)
    print("---------- test 3 ----------")
    test3 = TestOnlineAdAllocWithPred([3, 2, 3, 4], [(3, 0), (4, 1), (6, 2), (5, 3), (7, 4), (1, 5), (2, 6), (7, 7), (8, 8), (4, 9)], [[5, 17, 18, 14, 3, 2, 1, 1, 9, 15], [4, 12, 19, 5, 5, 3, 2, 15, 9, 8], [14, 13, 10, 9, 5, 6, 1, 3, 7, 7], [3, 5, 8, 10, 15, 10, 11, 6, 2, 2]])
    test3.runAllTests(1.4)

    # synthetic tests
    sData1 = SyntheticData(10, 10, 10, 10)
    weights1 = sData1.expDistMat()
    print(len(weights1), len(weights1[0]))
    sData1.gausDistList(10)
    budgets1 = sData1.sampleBudgets(1, 25)
    impressions1 = sData1.sampleImps()
    sTest1 = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    sTest1.runAllTests(1)

    
    