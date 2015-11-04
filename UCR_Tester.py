import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys


class SAX:
    def __init__(self, wordSize=8, alphabetSize=7, epsilon=1e-6):
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

    # Takes the average of each chunk
    def to_standard_paa(self, data):
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

    # Takes the max of a chunk if the average is above zero, the min otherwise
    def min_max_PAA(self, data):
        result = []
        n = len(data)
        numFrames = int(math.ceil(float(n) / float(self.wordSize)))
    
        for i in range(numFrames):
            startIndex = self.wordSize * i
            endIndex = min(self.wordSize * (i + 1), n) # wordsize or end of data
            # Take the mean of the chunk
            chunk = np.array(data[startIndex: endIndex])
            approx = np.mean(chunk)
            if approx > 0:
                result.append(chunk.max())
            else:
                result.append(chunk.min())
        return np.array(result)

    # if the avg is above zero, breaks ties to the max, else breaks ties to the min
    def min_max_group_tie_breaker(self, data):
        result = []
        n = len(data)
        numFrames = int(math.ceil(float(n) / float(self.wordSize)))
        breakpoints = self.getBreakpoints()

        for i in range(numFrames):
            startIndex = self.wordSize * i
            endIndex = min(self.wordSize * (i + 1), n) # wordsize or end of data
            # Take the mean of the chunk
            chunk = np.array(data[startIndex: endIndex])
            approx = np.mean(chunk)

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

            if approx > 0:
                # Take the max mode
                result.append(self.alphabet[counts.argmax()])
            else:
                # Take the min mode
                result.append(self.alphabet[::-1][counts[::-1].argmax()])

        return np.array(result)

    def group_sax(self, data):
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
        
    def to_standard_sax(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        paaData = self.to_standard_paa(normalizedData)
        alphaData = self.toAlphabet(paaData)

        return alphaData
        
    def to_min_max_sax(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        paaData = self.min_max_PAA(normalizedData)
        alphaData = self.toAlphabet(paaData)

        return alphaData

    def to_min_max_group_sax(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        alphaData = self.min_max_group_tie_breaker(normalizedData)

        return alphaData

    def to_group_sax(self, data):
        self.originalLength = len(data)
        normalizedData = self.normalize(data)
        alphaData = self.group_sax(normalizedData)

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
# UCR_DIRECTORY = r'C:\Users\Tom\Documents\fall_2015\research\UCR_TS_Archive_2015\\'

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

testFiles = ['Beef', 'OliveOil', 'Coffee']
testFiles += ['Earthquakes',]
testFiles += ['SmallKitchenAppliances', 'LargeKitchenAppliances', 'TwoLeadECG', 'ECGFiveDays']#, 'FordA', 'FordB', 'ElectricDevices', 'ECG5000']
testFiles += ['ItalyPowerDemand', 'Plane', 'Car']
testFiles += ['ECG200']
testFiles += ['Computers', ]
testFiles += ['ChlorineConcentration']

if len(sys.argv) > 1:
    print("Running on UCR dataset", sys.argv[1])
    testFiles = [sys.argv[1]]


def convert_train_data(sax_instance, dataset, test_type):
    test_dictionary = {
        "SAX": sax_instance.to_standard_sax,
        "Max G": sax_instance.to_group_sax,
        "MiMa G": sax_instance.to_min_max_group_sax,
        "MiMa SAX": sax_instance.to_min_max_sax,
    }
    return [test_dictionary[test_type](dat_piece) for dat_piece in dataset]

for testDataSet in testFiles:
    trainingData, trainingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TRAIN")
    testingData, testingLabels = loadUCRData(UCR_DIRECTORY + testDataSet + "/" + testDataSet + "_TEST")
    data = [trainingData, trainingLabels, testingData, testingLabels]

    print("Train size:", len(trainingData))
    print("Test size:", len(testingData))

    euclideanCorrect = determineCorrect(*data, classifyMethod="Euclidean")

    tests = {
        "SAX": ("red", {}),
        "Max G": ('cyan', {}),
        "MiMa G": ('green', {}),
        "MiMa SAX": ('black', {})
    }

    wordSizes = [1, 2, 3, 4, 5, 6]
    alphabetSize = 64

    for wordSize in wordSizes:
        for test in tests:
            print("Testing", testDataSet, "with word size", wordSize, "and algorithm", test)
            s = SAX(wordSize=wordSize, alphabetSize=alphabetSize)
            train_data = convert_train_data(s, trainingData, test)
            test_data = convert_train_data(s, testingData, test)

            for i in [0, 3]: # 64, 8
                data_package = [shiftSax(train_data, i), trainingLabels, shiftSax(test_data, i), testingLabels]

                key = test + " a=%d" % (alphabetSize >> i)
                temp = tests[test][1].get(key, [])
                temp.append(determineCorrect(*data_package, classifyMethod="SAX", argList=[s]))
                tests[test][1][key] = temp

    # Convert to correct percentage
    euclideanCorrect = np.array([euclideanCorrect] * len(wordSizes)) / len(testingData)
    for test in tests.keys():
        for key in tests[test][1].keys():
            tests[test][1][key] = np.array(tests[test][1][key]) / len(testingData)
    
    fig, ax = plt.subplots()
    ax.plot(wordSizes, euclideanCorrect, 'k:', label="Euclidean")

    # big and small alphabet
    styles = ['-', '--']

    max_value = 0
    min_value = 0
    for k, test in enumerate(tests.keys()):
        for i, key in enumerate(sorted(tests[test][1].keys())):
            # print(i, test,key)
            max_value = max(max_value, max(tests[test][1][key]))
            min_value = min(min_value, max(tests[test][1][key]))
            ax.plot(wordSizes, tests[test][1][key], color=tests[test][0], ls=styles[i], lw=2, label=key)

    plt.ylim([min_value - .1, max_value + .1])
    plt.title(testDataSet + " (%d classes)" % len(set(trainingLabels.flat)))
    plt.xlabel('Word Size')
    plt.ylabel('Percent Correct')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig("all-four-" + testDataSet + '.png')
    # plt.show()
    







