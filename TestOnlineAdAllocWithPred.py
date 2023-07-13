from OnlineAdAllocWithPred import OnlineAdAllocWithPred
import time

class TestOnlineAdAllocWithPred:

    def __init__(self, b, w):
        self.OAAWP = OnlineAdAllocWithPred(b, w)
        self.runtime = 0

    def runAlgorithm1(self, trust):
        start = time.time()
        res = self.OAAWP.algorithm1(trust)
        end = time.time()
        self.runtime = end - start
        for i in range(len(res)):
            print("Advertiser " + str(i) + ": " + str(res[i]))

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
    print("---------- test 1 ----------")
    test1 = TestOnlineAdAllocWithPred([2, 1], [[2, 1, 10], [100, 2, 1]])
    test1.runAllTests(1)
    print("---------- test 2 ----------")
    test2 = TestOnlineAdAllocWithPred([1, 1], [[0, 2, 50], [3, 10, 30]])
    test2.runAllTests(1)
    print("---------- test 3 ----------")
    test3 = TestOnlineAdAllocWithPred([1, 1], [[0, 2, 50], [3, 10, 30]])
    test3.runAllTests(2)
    print("---------- test 4 ----------")
    test4 = TestOnlineAdAllocWithPred([3, 2], [[2, 10, 4, 0], [1, 16, 0, 9]])
    test4.runAllTests(1)
    print("---------- test 5 ----------")
    test5 = TestOnlineAdAllocWithPred([2, 1, 3, 1], [[5, 50, 7, 4], [1, 2, 8, 4], [10, 0, 9, 30], [3, 0, 13, 1]])
    test5.runAllTests(3)


    
    