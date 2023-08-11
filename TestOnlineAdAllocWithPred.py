from OnlineAdAllocWithPred import OnlineAdAllocWithPred
from SyntheticData import SyntheticData
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

class TestOnlineAdAllocWithPred:

    def __init__(self, b, t, w, predictor = 0, epsilon = -1, prevImps = [], p = -1, threshMethod = 0):
        self.OAAWP = OnlineAdAllocWithPred(b, t, w, predictor, epsilon, prevImps, p, threshMethod)
        self.predictor = predictor
        self.runtime = 0

    def runAlgorithm1(self, alpha):
        start = time.time()
        self.res = self.OAAWP.algorithm1(alpha)
        end = time.time()
        self.runtime = end - start
        
    def runRandomMixture(self, alpha, q):
        start = time.time()
        self.res = self.OAAWP.randomMixture(alpha, q)
        end = time.time()
        self.runtime = end - start

    def runPRD(self, epsilon = -1):
        return self.OAAWP.PRDDual(epsilon)
    
    def reset(self):
        self.OAAWP.reset()

    def checkLP(self):
        self.OAAWP.checkLP()
    
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
        print("Robustness: " + str(self.OAAWP.getRobAndCon()[0]) + ", Consistency: " + str(self.OAAWP.getRobAndCon()[1]))

    def printRuntime(self):
        print("Runtime: " + str(self.OAAWP.runtime + self.OAAWP.LPRuntime) + " s")
        print("LP Runtime: " + str(self.OAAWP.LPRuntime) + " s")

    def runAllTests(self, alpha, q = -1, results = False, print = True):
        if q == -1:
            self.runAlgorithm1(alpha)
        else:
            self.runRandomMixture(alpha, q)
        if print:
            self.printResults(results)
            self.printDummyAdvertiser(results)
            self.printThresholds(results)
            self.printRobustnessAndConsistency()
            self.printRuntime()
    
    def draw_error_band(self, ax, x, y, err, **kwargs):
        # Calculate normals via centered finite differences (except the first point
        # which uses a forward difference and the last point which uses a backward
        # difference).
        dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
        dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
        l = np.hypot(dx, dy)
        nx = dy / l
        ny = -dx / l

        # end points of errors
        xp = x + nx * err
        yp = y + ny * err
        xn = x - nx * err
        yn = y - ny * err

        vertices = np.block([[xp, xn[::-1]],
                            [yp, yn[::-1]]]).T
        codes = np.full(len(vertices), Path.LINETO)
        codes[0] = codes[len(xp)] = Path.MOVETO
        path = Path(vertices, codes)
        ax.add_patch(PathPatch(path, **kwargs))
    
if __name__ == "__main__":
    # manual tests
    # print("---------- manual test 1 ----------")
    # test1 = TestOnlineAdAllocWithPred([2, 1], [(0, 0), (3, 1), (1, 2), (2, 3)], [[10, 12, 2, 1], [3, 7, 21, 6]])
    # test1.runAllTests(1)
    # test1.checkLP()
    
    # small synthetic tests
    # print("---------- small synthetic test 1 ----------")
    # sData1 = SyntheticData(4, 6, 4)
    # weights1 = sData1.expDistMat()
    # print("weights: " + str(weights1))
    # sData1.gausDistList()
    # budgets1 = sData1.sampleBudgets(3, 12)
    # print("budgets: " + str(budgets1))
    # impressions1 = sData1.sampleImps()
    # print("impressions: " + str(impressions1))
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

    # ws = [[34.392907857516335, 4.937635070973732, 10.861528206585348, 12.113387174196507, 2.9546329165121534], [6.684741222420044, 7.019374162571511, 3.702827681813816, 19.44604987733711, 7.884809880743408], [0.18345691186634566, 36.49062574006497, 23.576273982047265, 1.58287578671112, 3.7844108989223546], [8.31710196153401, 0.8545402929224519, 5.5375845666601675, 2.886215010007308, 6.970421309530068], [3.71604218133018, 12.429765719766126, 19.464583520669354, 5.283365694268852, 2.5099323949492547], [21.942619867906554, 11.975372923393756, 5.757104260326971, 15.288685759784773, 17.96161240498456], [5.994578741429196, 0.6228164537989793, 6.867514125158856, 3.6257650805559773, 2.3741675782208187], [1.7550914635727666, 5.4766811350569995, 12.113705894769964, 7.191469926024503, 14.824010790960456], [18.11275423092607, 0.0612649147436142, 5.108350763804559, 10.146043286050494, 15.127982902855255], [18.625786503552092, 26.590812430655962, 1.6868843339162694, 0.45968495283383487, 7.434733114106936]]
    # Bs = [7, 5, 9, 5, 6, 6, 6, 5, 6, 5]
    # ts = [(4, -25.578812479722316), (0, -16.277336650882667), (1, -14.604124481375404), (1, -10.119558170342033), (1, -8.55865504150902), (3, -8.143780286821348), (1, -8.009884938020994), (2, -6.589586608254692), (4, -6.4976694034171665), (4, -6.35741885764892), (4, -5.395273141835508), (2, -5.0346860880972395), (0, -4.581522291137154), (3, -4.49506648495064), (3, -4.1049666291748865), (4, -2.472162360857505), (0, -2.213278882133949), (0, -2.0770232136032276), (4, 0.19171647168235417), (3, 0.8877998486018254), (3, 0.9651184437441762), (3, 1.2894975557791448), (1, 1.420040968544072), (3, 1.6321170129007234), (3, 2.548607301749165), (1, 2.6271535749016874), (0, 3.357637318428161), (4, 3.8095418449761516), (3, 3.886805775097536), (0, 3.980432173608072), (4, 5.101253669493715), (2, 5.285541330552501), (3, 5.3473411718728), (0, 5.360766538545228), (4, 5.648100651290671), (2, 5.773673998598326), (0, 7.157495717575832), (2, 7.714589376010443), (2, 8.007938270058341), (0, 8.218760709173438), (2, 10.160276215656403), (0, 11.414679473869862), (1, 12.310699650889255), (2, 12.65083420425796), (1, 13.561391926237324), (2, 13.849503535199686), (1, 13.99849500341589), (2, 16.451137891293964), (4, 16.654388981708415), (1, 25.251668636860444)]
    # sTest1 = TestOnlineAdAllocWithPred(Bs, ts, ws, 2)
    # sTest1.runAllTests(1)

    # print("---------- small synthetic test 2 ----------")
    # sData2 = SyntheticData(5, 10, 5)
    # weights2 = sData2.expDistMat()
    # sData2.gausDistList()
    # budgets2 = sData2.sampleBudgets(10, 40)
    # impressions2 = sData2.sampleImps()
    # sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2, 2)
    # sTest2.runAllTests(2, False)
    # sTest2 = TestOnlineAdAllocWithPred(budgets2, impressions2, weights2)
    # sTest2.runAllTests(100, False)

    # print("---------- small synthetic test 3 ----------")
    # sData3 = SyntheticData(20, 20, 20)
    # weights3 = sData3.expDistMat()
    # sData3.gausDistList()
    # budgets3 = sData3.sampleBudgets(10, 40)
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
    # sTest5 = TestOnlineAdAllocWithPred(budgets5, impressions5, weights5, 1)
    # sTest5.runAllTests(2, False)
    # sTest5 = TestOnlineAdAllocWithPred(budgets5, impressions5, weights5, 2)
    # sTest5.runAllTests(2, False)
    
    # big synthetic tests
    # print("---------- big synthetic test 1 ----------")
    # sData6 = SyntheticData(10, 100, 10)
    # weights6 = sData6.expDistMat()
    # sData6.gausDistList()
    # budgets6 = sData6.sampleBudgets(50, 200)
    # impressions6 = sData6.sampleImps()
    # sTest6 = TestOnlineAdAllocWithPred(budgets6, impressions6, weights6, 2, -1, 1)
    # sTest6.runAllTests(2, False)
    # sTest6 = TestOnlineAdAllocWithPred(budgets6, impressions6, weights6, 2, -1, 1)
    # sTest6.runAllTests(8, False)

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
    # sData9 = SyntheticData(50, 25, 50)
    # weights9 = sData9.expDistMat()
    # sData9.gausDistList()
    # budgets9 = sData9.sampleBudgets(12, 50)
    # impressions9 = sData9.sampleImps()
    # sTest9 = TestOnlineAdAllocWithPred(budgets9, impressions9, weights9, 2)
    # sTest9.runAllTests(3, False)

    # print("---------- big synthetic test 5 ----------")
    # sData10 = SyntheticData(100, 100, 500)
    # weights10 = sData10.expDistMat()
    # sData10.gausDistList()
    # budgets10 = sData10.sampleBudgets(25, 1000)
    # impressions10 = sData10.sampleImps()
    # sTest10 = TestOnlineAdAllocWithPred(budgets10, impressions10, weights10, 2)
    # sTest10.runAllTests(1.5, False)

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

    figure, axis = plt.subplots(1, 2)

    sData = SyntheticData(10, 10, 10, 1.5)
    weights1 = sData.expDistMat()
    sData.gausDistList()
    budgets1 = sData.sampleBudgets(5, 40)
    impressions1 = sData.sampleImps()
    # print("impressions: " + str(impressions1))

    prevData1 = SyntheticData(10, 10, 8, 1.5)
    prevData1.gausDistList()
    prevImps = prevData1.sampleImps()
    # print("previous impressions: " + str(prevImps))
    
    sTestOPT = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1)
    sTestDualBase = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 1)
    sTestPrevDay = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 2, -1, prevImps)
    sTestOPTRC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 3)
    sTestOPTBC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 4)

    # sTestOPT = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 0, -1, [], -1, 1)
    # sTestDualBase = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 1, -1, [], -1, 1)
    # sTestPrevDay = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 2, -1, prevImps, -1, 1)
    # sTestOPTRC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 3, -1, [], -1, 1)
    # sTestOPTBC = TestOnlineAdAllocWithPred(budgets1, impressions1, weights1, 4, -1, [], -1, 1)

    labels = ["OPT", "DualBase", "PreviousDay", "OPT random corruption p = 0.9", "OPT biased corruption p = 0.9", "", "", "", "", "", "Worst-Case Baseline"]
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:red", "black"]
    linestyles = ["solid", "solid", "solid", "solid", "solid", "dashed", "dashed", "dashed", "dashed", "dashed", "solid"]

    numPRD = 11

    # for PRD in range(numPRD):
    #     alphas = []
    #     consistency = []
    #     robustness = []
    #     prevImps1 = []

    #     if PRD in [0, 5, 10]:
    #         sTest = sTestOPT
    #     elif PRD in [1, 6]:
    #         sTest = sTestDualBase
    #     elif PRD in [2, 7]:
    #         sTest = sTestPrevDay
    #     elif PRD in [3, 8]:
    #         sTest = sTestOPTRC
    #     else:
    #         sTest = sTestOPTBC

    #     for alpha in range(2, 11):
    #         alpha /= 2
    #         if PRD < 5:
    #             sTest.runAllTests(alpha, -1)
    #         elif PRD == 10:
    #             sTest.runAllTests(alpha, 1, False, False)
    #         else:
    #             sTest.runAllTests(alpha, 1 / alpha, False, False)
    #         alphas.append(alpha)
    #         robustness.append(sTest.OAAWP.getRobAndCon()[0])
    #         consistency.append(sTest.OAAWP.getRobAndCon()[1])

    #     print("robustness: " + str(robustness) + ", consistency: " + str(consistency))
        
    #     if PRD != 10:
    #         axis[0].plot(alphas, consistency, color = colors[PRD], linestyle = linestyles[PRD])        
    #     axis[1].plot(alphas, robustness, color = colors[PRD], linestyle = linestyles[PRD], label = labels[PRD])
    
    # axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
    # axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')

    # axis.flat[0].set(xlabel = "\u03B1", ylabel = "Consistency")
    # axis.flat[1].set(xlabel = "\u03B1", ylabel = "Robustness")

    # axis[1].legend()

    # plt.show()

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
                robustness.append(sTest.OAAWP.getRobAndCon()[0])
                consistency.append(sTest.OAAWP.getRobAndCon()[1])
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
            # sTest.draw_error_band(axis[0], np.array(alphas), avgCon, err=errCon, facecolor=colors[PRD], edgecolor="none", alpha=.3)
            # sTest.draw_error_band(axis[1], np.array(alphas), avgRob, err=errRob, facecolor=colors[PRD], edgecolor="none", alpha=.3)
            axis[0].fill_between(np.array(alphas), avgCon - errCon, avgCon + errCon, facecolor = colors[PRD], edgecolor="none", alpha=.25)
            axis[1].fill_between(np.array(alphas), avgRob - errRob, avgRob + errRob, facecolor = colors[PRD], edgecolor="none", alpha=.25)
    
    axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
    axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')

    axis.flat[0].set(xlabel = "\u03B1", ylabel = "Consistency")
    axis.flat[1].set(xlabel = "\u03B1", ylabel = "Robustness")

    axis[1].legend()

    plt.show()