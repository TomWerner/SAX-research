import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

class SAX:
    def __init__(self, wordSize = 8, alphabetSize = 7, epsilon = 1e-6):
        self.wordSize = wordSize
        self.alphabetSize = alphabetSize
        self.epsilon = epsilon
        self.alphabet = np.array(range(0, alphabetSize))
        self.breakpoints = [norm.ppf(x / alphabetSize) for x in range(1, alphabetSize)]
        
    def normalize(self, data):
        # Gives data a mean of zero and a standard deviation of 1
        # If the standard dev is below epsilon, returns zeros to avoid amplification

        # if already an np array, doesn't change it
        array = np.array(data)
        standardDev = array.std()
        if standardDev < self.epsilon:
            return np.array([0 for x in array])
        return (array - array.mean()) / standardDev

    def toPAA(self, data):
        result = []
        n = len(data)
        numFrames = int(math.ceil(float(n) / float(self.wordSize)))
    
        for i in range(numFrames):
            startIndex = self.wordSize * i
            endIndex = min(self.wordSize * (i + 1), n) # wordsize or end of data
            # Take the mean of the chunk
            approx = np.mean(np.array(data[startIndex: endIndex]))
            result.append(approx)
        return np.array(result)

    def toGroupPAA(self, data):
        result = ""
        n = len(data)
        numFrames = int(math.ceil(float(n) / float(self.wordSize)))
    
        alphabet = "abcdefghijklmnopqrstuvwxyz"[0:self.alphabetSize]
        breakpoints = self.getBreakpoints()
        
        for i in range(numFrames):
            startIndex = self.wordSize * i
            endIndex = min(self.wordSize * (i + 1), n) # wordsize or end of data

            counts = np.zeros(self.alphabetSize)
            for point in data[startIndex: endIndex]:
                found = False
                for i in range(len(breakpoints)):
                    if point < breakpoints[i] and not found:
                        counts[i] += 1
                        found = True
                        break
                if not found:
                    counts[-1] += 1
            result += alphabet[counts.argmax()]
            
        return result

    def getBreakpoints(self):
        return self.breakpoints

    def toAlphabet(self, paaData):
        alphabet = "abcdefghijklmnopqrstuvwxyz"[0:self.alphabetSize]
        breakpoints = self.getBreakpoints()
        result = ""
        for point in paaData:
            found = False
            for i in range(len(breakpoints)):
                if point < breakpoints[i] and not found:
                    result += alphabet[i]
                    found = True
                    break
            if not found:
                result += alphabet[-1]
        return result
        
    def toSAX(self, data):
        self.originalLength = len(data)
        #normalizedData = self.normalize(data)
        paaData = self.toPAA(data)#normalizedData)
        alphaData = self.toAlphabet(paaData)

        return alphaData

    def toGroupSAX(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        alphaData = self.toGroupPAA(normalizedData)

        return alphaData

    def minDist(self, stringA, stringB):
        if len(stringA) != len(stringB):
            print("Strings are not equal length")
            print(stringA)
            print(stringB)
            return

        totalDistance = 0
        for i in range(len(stringA)):
            totalDistance += self.letterDistance(stringA[i], stringB[i]) ** 2

        scalingFactor = (self.originalLength / self.wordSize)
        #return math.sqrt(scalingFactor) * abs(totalDistance)
        return math.sqrt(scalingFactor * totalDistance)

    def letterDistance(self, letter1, letter2):
        if abs(ord(letter1) - ord(letter2)) <= 1:
            return 0
        first = max(ord(letter1), ord(letter2)) - ord('a') - 1
        second = min(ord(letter1), ord(letter2)) - ord('a')
        try:
            return self.getBreakpoints()[first] - self.getBreakpoints()[second]
        except:
            print(letter1, letter2, first, second, self.getBreakpoints(), self.alphabetSize)
            raise




UCR_DIRECTORY = '/Users/test/fall_2015/research/UCR_TS_Archive_2015/'

def loadUCRData(path):
    file = open(path, 'r')
    lines = file.readlines()
    rows = len(lines)
    cols = max([len(row.split(",")) - 1 for row in lines]) # the first col is the label

    labels = np.zeros((rows, 1))
    data = np.zeros((rows, cols))
    
    for i, row in enumerate(lines):
        rowData = row.split(",")
        labels[i] = float(rowData[0])
        for k, col in enumerate(rowData[1:]):
            data[i][k] = float(col)

    return data, labels

def euclideanDist(obj1, obj2):
    if len(obj1) != len(obj2):
        raise "Objects must be of equal size"
    return sum(np.square(np.asarray(obj1) - np.asarray(obj2)))

def classify(trainingData, trainingLabels, unknownObj):
    bestSoFar = float("inf")
    predictedClass = None

    for i, trainingRow in enumerate(trainingData):
        # Euclidean distance
        distance = euclideanDist(trainingRow, unknownObj)
        if distance < bestSoFar:
            bestSoFar = distance
            predictedClass = trainingLabels[i]

    return predictedClass

def classifySAX(sax, trainingSaxData, trainingLabels, unknownSaxObj):
    bestSoFar = float("inf")
    predictedClass = None
    for i, trainingRow in enumerate(trainingSaxData):
        # mindist distance
        distance = sax.minDist(trainingRow, unknownSaxObj)
        if distance < bestSoFar:
            bestSoFar = distance
            predictedClass = trainingLabels[i]

    return predictedClass

def determineCorrect(trainData, trainLabels, testData, testLabels, classifyMethod, argList = None):
    correct = 0
    for i, testRow in enumerate(testData):
        actualClass = testLabels[i]
        if classifyMethod == "Euclidean":
            predictedClass = classify(trainData, trainLabels, testRow)    
        elif classifyMethod == "SAX":
            predictedClass = classifySAX(argList[0], trainData, trainLabels, testRow)   
        else:
            raise "Unsupported classify method"
        
        if actualClass == predictedClass:
            correct += 1
    return correct
    
#testFiles = ['CBF', 'Coffee', 'ECG200', 'FaceAll', 'FaceFour', 'Fish',
#             'Gun_Point', 'Lighting2', 'Lighting7', 'OliveOil', 'OSULeaf',
#             'synthetic_control', 'SwedishLeaf', 'Trace', 'Two_Patterns', 'wafer', 'yoga']
#testFiles = ['wafer', 'yoga']
testFiles = ['coffee']
for testDataSet in testFiles:

    trainingData, trainingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TRAIN")
    testingData, testingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TEST")
    #testingData = testingData[0:100]
    #testingLabels = testingLabels[0:100]
    data = [trainingData, trainingLabels, testingData, testingLabels]
    

    euclideanCorrect = determineCorrect(*data, classifyMethod = "Euclidean")
    saxCorrect = []
    saxCorrect2 = []
    groupSaxCorrect = []
    groupSaxCorrect2 = []
    wordSizes = [1, 2, 4, 6, 8, 10, ]

    #alphabetSize = 5
    #alphabetSize2 = min(20, max(len(set(trainingLabels.flat)), 3))
    #alphabetSize = len(set(trainingLabels.flat))
    alphabetSize2 = 10 #len(set(trainingLabels.flat)) * 2
    alphabetSize = 20

    for wordSize in wordSizes:
        print(testDataSet, "with word size", wordSize)
        s = SAX(wordSize = wordSize, alphabetSize = alphabetSize)
        trainingDataSAX = [s.toSAX(data) for data in trainingData]
        testingDataSAX = [s.toSAX(data) for data in testingData]
        trainingDataGroupSAX = [s.toGroupSAX(data) for data in trainingData]
        testingDataGroupSAX = [s.toGroupSAX(data) for data in testingData]
        saxData = [trainingDataSAX, trainingLabels, testingDataSAX, testingLabels]
        groupSaxData = [trainingDataGroupSAX, trainingLabels, testingDataGroupSAX, testingLabels]

        saxCorrect.append(determineCorrect(*saxData, classifyMethod = "SAX", argList=[s]))
        groupSaxCorrect.append(determineCorrect(*groupSaxData, classifyMethod = "SAX", argList=[s]))


        s2 = SAX(wordSize=wordSize, alphabetSize=alphabetSize2)
        trainingDataSAX = [s2.toSAX(data) for data in trainingData]
        testingDataSAX = [s2.toSAX(data) for data in testingData]
        trainingDataGroupSAX = [s2.toGroupSAX(data) for data in trainingData]
        testingDataGroupSAX = [s2.toGroupSAX(data) for data in testingData]
        saxData2 = [trainingDataSAX, trainingLabels, testingDataSAX, testingLabels]
        groupSaxData2 = [trainingDataGroupSAX, trainingLabels, testingDataGroupSAX, testingLabels]

        saxCorrect2.append(determineCorrect(*saxData2, classifyMethod = "SAX", argList=[s2]))
        groupSaxCorrect2.append(determineCorrect(*groupSaxData2, classifyMethod = "SAX", argList=[s2]))

    euclideanCorrect = np.array([euclideanCorrect] * len(wordSizes)) / len(testingData)
    saxCorrect = np.array(saxCorrect) / len(testingData)
    groupSaxCorrect = np.array(groupSaxCorrect) / len(testingData)
    
    saxCorrect2 = np.array(saxCorrect2) / len(testingData)
    groupSaxCorrect2 = np.array(groupSaxCorrect2) / len(testingData)
    
    fig, ax = plt.subplots()
    ax.plot(wordSizes, euclideanCorrect, 'k:', label="Euclidean")
    ax.plot(wordSizes, saxCorrect, 'r', label="SAX a=" + str(alphabetSize))
    ax.plot(wordSizes, groupSaxCorrect, 'r--', label="Group SAX a=" + str(alphabetSize))
    ax.plot(wordSizes, saxCorrect2, 'b', label="SAX a=" + str(alphabetSize2))
    ax.plot(wordSizes, groupSaxCorrect2, 'b--', label="Group SAX a=" + str(alphabetSize2))

    plt.title(testDataSet)
    plt.xlabel('Word Size')
    plt.ylabel('Percent Correct')

    legend = ax.legend(loc='upper right', shadow=True)
    plt.savefig(testDataSet + '.png')
    #plt.show()
    







