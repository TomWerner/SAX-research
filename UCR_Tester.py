import numpy as np
import math
import time
import matplotlib.pyplot as plt

class SAX:
    def __init__(self, wordSize = 8, alphabetSize = 7, epsilon = 1e-6):
        self.wordSize = wordSize
        self.alphabetSize = alphabetSize
        self.epsilon = epsilon
        
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
        breakpoints = {'3' : [-0.43, 0.43],
                                '4' : [-0.67, 0, 0.67],
                                '5' : [-0.84, -0.25, 0.25, 0.84],
                                '6' : [-0.97, -0.43, 0, 0.43, 0.97],
                                '7' : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                                '8' : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                                '9' : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                                '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                                '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                                '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                                '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                                '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                                '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
                                '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
                                '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
                                '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
                                '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
                                '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
                                }
        return breakpoints[str(self.alphabetSize)]

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
        return self.getBreakpoints()[max(ord(letter1), ord(letter2)) - ord('a') - 1] -\
               self.getBreakpoints()[min(ord(letter1), ord(letter2)) - ord('a')]





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
        distance = s.minDist(trainingRow, unknownSaxObj)
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
    
testFiles = ['CBF', 'Coffee', 'ECG200', 'FaceAll', 'FaceFour', 'Fish',
             'Gun_Point', 'Lightning2', 'Lightning7', 'OliveOil', 'OSULeaf',
             'synthetic_control', 'SwedishLeaf', 'Trace', 'Two_Patterns', 'Wafter', 'yoga']

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

    alphabetSize = 5
    alphabetSize2 = min(20, max(len(set(trainingLabels.flat)), 3))


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


        s2 = SAX(wordSize = wordSize, alphabetSize = alphabetSize2)
        trainingDataSAX = [s2.toSAX(data) for data in trainingData]
        testingDataSAX = [s2.toSAX(data) for data in testingData]
        trainingDataGroupSAX = [s2.toGroupSAX(data) for data in trainingData]
        testingDataGroupSAX = [s2.toGroupSAX(data) for data in testingData]
        saxData2 = [trainingDataSAX, trainingLabels, testingDataSAX, testingLabels]
        groupSaxData2 = [trainingDataGroupSAX, trainingLabels, testingDataGroupSAX, testingLabels]

        saxCorrect2.append(determineCorrect(*saxData, classifyMethod = "SAX", argList=[s2]))
        groupSaxCorrect2.append(determineCorrect(*groupSaxData, classifyMethod = "SAX", argList=[s2]))

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

    plt.show()
    







