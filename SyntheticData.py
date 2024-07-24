import random
import numpy as np

class SyntheticData:
    """
    A class used to generate sythetic data.

    ...

    Attributes
    ----------
    numAdvs : int
        The number of advertisers
    numImptTypes : int
        The number of impression types
    scale : int
        The scale
    weights : matrix
        The matrix of weights in which the rows are advertisers and the columns are impressions
    displayTimes : list
        The list of display times in which each index corresponds to an impression
    numImpPerType : int
        The number of impressions per type of impression
    numImpressions : int
        The number of impressions
    budgets : list
        The list of budgets in which each index corresponds to an advertiser
    impressions : list
        The list of impressions
    
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

    def __init__(self, numAdvs, numImpTypes, numImpPerType, scale = 1):
        self.numAdvs = numAdvs
        self.numImpTypes = numImpTypes
        self.scale = scale
        self.weights = []
        self.displayTimes = []
        self.numImpPerType = numImpPerType
        self.numImpressions = self.numImpTypes * self.numImpPerType
        self.budgets = []
        self.impressions = []

    def listToMat(self, list):
        """
        Converts a list of numAdvs*numImpressions elements to a matrix of numAdvs rows and numImpressions columns

        Parameters
        ----------
        list : list
            The list to be converted into a matrix.
        """
        mat = np.mat(list)
        mat = mat.reshape(self.numAdvs, self.numImpTypes)
        return mat.tolist()

    def expDistMat(self):
        """
        Generates a matrix from an exponential distribution.
        """
        self.weights = self.listToMat(np.random.exponential(self.scale, self.numAdvs * self.numImpTypes))
        return self.weights
    
    def gausDistList(self, sd = 10):
        """
        Generates a list from a gaussian distribution.

        Parameters
        ----------
        sd : int
            The standard deviation of the normal distribution.
        """
        self.displayTimes = np.random.normal(random.uniform(0, 1), sd, self.numImpressions)
        return self.displayTimes
    
    def sampleBudgets(self, min, max, oneEqualToMin = False):
        """
        Generates a list of sample budgets in which each index corresponds to an advertiser.

        Parameters
        ----------
        min : int
            The minimum budget value.
        max : int
            The maximum budget value.
        min : int
            True if one budget value should be equal to the minimum, False otherwise.
        
        Raises
        ------
        Exception
            If minimum budget value is zero.
        """
        if min == 0:
            raise Exception("Min must be greater than 0.")
        if min > max:
            max += min
            return
        for adv in range(self.numAdvs):
            self.budgets.append(random.randrange(min, max + 1))
        if oneEqualToMin:
            self.budgets[random.randrange(0, len(self.budgets))] = min
        return self.budgets

    def sampleImps(self):
        """
        Generates a list of sample impressions that are defined by (impression type, display time).
        """
        t = 0
        for i in range(self.numImpTypes):
            for j in range(self.numImpPerType):
                self.impressions.append((i, self.displayTimes[t]))
                t += 1
        self.impressions.sort(key=lambda a : a[1])
        return self.impressions
