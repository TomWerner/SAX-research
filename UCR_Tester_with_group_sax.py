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
        result = []
        n = len(data)
        numFrames = int(math.ceil(float(n) / float(self.wordSize)))
    
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
            result.append(self.alphabet[counts.argmax()])
        return np.array(result)

    def getBreakpoints(self):
        return self.breakpoints

    def toAlphabet(self, paaData):
        breakpoints = self.getBreakpoints()
        result = []
        for point in paaData:
            found = False
            for i in range(len(breakpoints)):
                if point < breakpoints[i] and not found:
                    result.append(self.alphabet[i])
                    found = True
                    break
            if not found:
                result.append(self.alphabet[-1])
        return np.array(result)
        
    def toSAX(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        paaData = self.toPAA(normalizedData)
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
        if abs(letter1 - letter2) <= 1:
            return 0
        first = max(letter1, letter2) - 1
        second = min(letter1, letter2)
        try:
            return self.breakpoints[first] - self.breakpoints[second]
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

def shiftSax(saxArray, amt):
    return np.right_shift(saxArray, amt)
    
#testFiles = ['CBF', 'Coffee', 'ECG200', 'FaceAll', 'FaceFour', 'Fish',
#             'Gun_Point', 'Lighting2', 'Lighting7', 'OliveOil', 'OSULeaf',
#             'synthetic_control', 'SwedishLeaf', 'Trace', 'Two_Patterns', 'wafer', 'yoga']
#testFiles = ['wafer', 'yoga']
testFiles = ['CBF', 'synthetic_control', 'coffee', 'Fish', 'Lighting2', 'Lighting7', 'Trace']
testFiles = ['MiddlePhalanxOutlineAgeGroup', 'ArrowHead', 'Beef', 'MiddlePhalanxOutlineCorrect',
             'BeetleFly', 'MoteStrain', 'BirdChicken', 'Car', 'OliveOil', 'Plane', 'ShapeletSim']
testFiles = ['50words', 'Adiac']
testFiles = ['ECG200']
testFiles = ['Ham']
testFiles = ['Meat']
testFiles = ['BirdChicken']
testFiles = ['CBF']


testFiles = ['Beef', 'OliveOil', 'Coffee']
testFiles += ['Earthquakes', 'ChlorineConcentration']
testFiles += ['SmallKitchenAppliances', 'LargeKitchenAppliances', 'TwoLeadECG', 'ECGFiveDays']#, 'FordA', 'FordB', 'ElectricDevices', 'ECG5000']
testFiles += ['ItalyPowerDemand', 'Plane', 'Car']
testFiles += ['ECG200']
testFiles += ['Computers', ]

for testDataSet in testFiles:

    trainingData, trainingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TRAIN")
    testingData, testingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TEST")
    data = [trainingData, trainingLabels, testingData, testingLabels]

    print("Train size:", len(trainingData))
    print("Test size:", len(testingData))

    euclideanCorrect = determineCorrect(*data, classifyMethod = "Euclidean")
    saxCorrect = {}
    groupSaxCorrect = {}
    wordSizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]
    alphabetSize = 64

    for wordSize in wordSizes:
        print(testDataSet, "with word size", wordSize)
        s = SAX(wordSize = wordSize, alphabetSize = alphabetSize)
        trainingDataSAX = [s.toSAX(data) for data in trainingData]
        testingDataSAX = [s.toSAX(data) for data in testingData]
        trainingDataGroupSAX = [s.toGroupSAX(data) for data in trainingData]
        testingDataGroupSAX = [s.toGroupSAX(data) for data in testingData]

        for i in range(0, 5): # 32, 16, 8, 4
            saxData = [shiftSax(trainingDataSAX, i), trainingLabels, shiftSax(testingDataSAX, i), testingLabels]
            groupSaxData = [shiftSax(trainingDataGroupSAX, i), trainingLabels, shiftSax(testingDataGroupSAX,i), testingLabels]

            
            
            key = "SAX a=%d" % (alphabetSize >> i)
            temp = saxCorrect.get(key, [])
            temp.append(determineCorrect(*saxData, classifyMethod = "SAX", argList=[s]))
            saxCorrect[key] = temp
        
            key = 'G-' + key
            temp = groupSaxCorrect.get(key, [])
            temp.append(determineCorrect(*groupSaxData, classifyMethod = "SAX", argList=[s]))
            groupSaxCorrect[key] = temp

    euclideanCorrect = np.array([euclideanCorrect] * len(wordSizes)) / len(testingData)
    for key in saxCorrect.keys():
        saxCorrect[key] = np.array(saxCorrect[key]) / len(testingData)
        groupSaxCorrect["G-" + key] = np.array(groupSaxCorrect["G-" + key]) / len(testingData)
    
    fig, ax = plt.subplots()
    ax.plot(wordSizes, euclideanCorrect, 'k:', label="Euclidean")

    colors = ['red', 'blue', 'darkgreen', 'cyan', 'black']
    for i, key in enumerate(sorted(saxCorrect.keys())):
        ax.plot(wordSizes, saxCorrect[key], color=colors[i], label=key)
        ax.plot(wordSizes, groupSaxCorrect['G-' + key], color=colors[i], ls='--', label='G-' + key)
    
    plt.title(testDataSet + " (%d classes)" % len(set(trainingLabels.flat)))
    plt.xlabel('Word Size')
    plt.ylabel('Percent Correct')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig("tuesday-" + testDataSet + '.png')
    #plt.show()
    







