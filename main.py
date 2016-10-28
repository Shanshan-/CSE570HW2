import sys
import matplotlib
from sklearn.naive_bayes import GaussianNB
from random import *
import random
import math
import numpy as np
import pylab as pl


""" CLASSES """
class Transmitter:
    # create a point class to store info about a tranmitter

    def __init__(self, xCoor, yCoor):
        self.x = xCoor
        self.y = yCoor

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def isEqual(self, other):
        if self.x == other.x & self.y == other.y:
            return True
        else:
            return False

class gridCell:
    # create a class to fill in the grid with

    def __init__(self, xCoor, yCoor, numTrans):
        self.x = xCoor
        self.y = yCoor
        self.trans = 0  # 0 if square is empty, 1 if has a transmitter
        self.featureVec = [0] * numTrans    # fingerprint feature vector
        self.locError = 1.0

    def __str__(self):
        string = "[" + str(self.featureVec) + "]"
        return string

""" FUNCTIONS """
# print out the results of the current grid, where 0 = location; 1 = transmitters; 2 = vectors; 3 = errors
def printGrid(grid, contentOpt):
    if contentOpt == 0:  # print out x, y locations
        print "Printing out grid locations"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "(" + str(grid[numx][numy].x) + "," + str(grid[numx][numy].y) + ")",
            print "\n"
    elif contentOpt == 1:  # print out boolean for transmitter
        print "Printing out transmitter locations"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "[ " + str(grid[numx][numy].trans) + " ]",
            print "\n"
    elif contentOpt == 2:  # print out all feature vectors
        print "Printing out all fingerprint feature vectors"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "[" + str(grid[numx][numy].featureVec) + "]",
            print "\n"
    elif contentOpt == 3:  # print out all location errors
        print "Printing out all location errors"
        print "Not supported yet"
    else:  # this is not an option
        print "Not an option"

# print out all of the transmitters
def printTrans(transmitters):
    print "Printing out transmitters"
    for trans in transmitters:
        print trans
    print "\n"

# setup the grid and the transmitters
def setupSim(resolution, numTrans):
    # setup the grid
    grid = [[gridCell(0, 0, numTrans) for x in range(resolution)] for y in range(resolution)]
    for numx in range(0, resolution):
        for numy in range(0, resolution):
            grid[numx][numy].x = numx
            grid[numx][numy].y = numy

    # place transmitters
    transmitters = [0] * numTrans
    for num in range(0, numTrans):
        #generate the coordinates for the transmitter
        transX = randint(0, resolution-1)
        transY = randint(0, resolution-1)

        #check to see if it overlaps
        while grid[transX][transY].trans == 1:
            transX = randint(0, resolution-1)
            transY = randint(0, resolution-1)

        # initialize the transmitter and mark the grid space as used
        transmitters[num] = Transmitter(transX, transY)
        grid[transX][transY].trans = 1  # temporary setting

    return grid, transmitters

# determine the distance between two given transmitters (in meters)
def calcDist(gridx, gridy, transx, transy, cellLength):
    # calculate the distance in terms of grid squares
    xlen = abs(gridx - transx)
    ylen = abs(gridy - transy)
    if xlen == 0 and ylen == 0:
        return 0  # return 0 if negligible
    dist = math.sqrt(xlen**2 + ylen**2)
    return cellLength * dist  # convert to meters and return

# calculate and return the path loss (in dB)
def getRecStrength(shadowDev, dist):
    """ PL = Pt - Pr = 10*alpha*log(d) + noise
        --> Pr = Pt - 10*alpha*log(d) - noise where:
        PL = path loss
        Pr = received signal strength (returned value)
        Pt = transmitted power in dBm = 10*log(P_t/1mW)
        P_t = transmitted power in watts (constant = 16mW = 0.016W)
        alpha = constant = 2.5
        d = distance between receiver and transmitter (dist)
        noise = variance to function (pull from random.gauss(0, shadowDev)
    """
    transmit = 10 * math.log10(16)
    if dist == 0:
        return transmit
    log = 10 * 2.5 * math.log10(dist)
    noise = random.gauss(0, shadowDev)
    receivedStrength = transmit - log - noise
    return receivedStrength

# generate a fingerprint vector for each grid cell
def genFingerprints(grid, transmitters, resolution, shadowDev):
    for numx in range(0, resolution):  # iterate through each row
        for numy in range(0, resolution):  # iterate through each column
            cell = grid[numx][numy]
            for x in range(0, len(transmitters)):  # for each transmitter at each cell
                trans = transmitters[x]
                dist = calcDist(cell.x, cell.y, trans.x, trans.y, 200/resolution)
                strength = getRecStrength(shadowDev, dist)
                cell.featureVec[x] = strength  # set the received signal strength

# create a feature vector for a given cell, and return it can probably use this to replace parts of genFingerprints
def genFeatVec(cell, transmitters, resolution, shadowDev):
    vec = [0] * len(transmitters)
    for x in range(0, len(transmitters)):  # for each transmitter at each cell
        trans = transmitters[x]
        dist = calcDist(cell.x, cell.y, trans.x, trans.y, 200/resolution)
        strength = getRecStrength(shadowDev, dist)
        vec[x] = strength  # store the received signal strength
    return vec

# generate the testing and training data feature vectors
def genData(grid, transmitters, resolution, shadowDev):
    # generate return variables'
    trainPredList = []
    trainTarList = []
    testPredList = []
    testTarList = []

    # for each grid cell, choose a random number between 10-20
    for numx in range(0, resolution):  # iterate through each row
        for numy in range(0, resolution):  # iterate through each column
            rand = randrange(10, 20)  # generate rand number of feature vectors
            for x in range(0, rand):  # for each vector to create
                vec = genFeatVec(grid[numx][numy], transmitters, resolution, shadowDev)
                coor = numx * 200/resolution + numy  # following C array notation

                # determine if the new vector is training or testing data, and store appropriately
                testRand = random.random()
                if testRand <= 0.1:  # this is testing data
                    testPredList.append(vec)
                    testTarList.append(coor)
                else:  # this is training data
                    trainPredList.append(vec)
                    trainTarList.append(coor)

    return np.asarray(trainPredList), np.asarray(trainTarList), np.asarray(testPredList), np.asarray(testTarList)

# change the values in an array from numbers to coordinate tuples
def changeToTuples(tupList, resolution):
    ans = []
    for x in range(0, len(tupList)):
        val = tupList[x]
        xcoor = val / resolution
        ycoor = val % resolution
        ans.append([xcoor, ycoor])
    return ans

# calculate the Euclidean distance between two tuple grids and return in separate array
def calcError(predictVals, testVals, resolution):
    errors = []
    sumErr = 0
    for x in range(0, len(predictVals)):
        # calculate the Euclidean distance
        dist = calcDist(predictVals[x][0], predictVals[x][1], testVals[x][0], testVals[x][1], resolution)
        errors.append(dist)
        sumErr += dist
    return errors, sumErr/len(predictVals)

# run the simulation once, and return average error
def beginSim(numTrans, resolution, shadowDev):
    # determine resolution and create grid and transmitters
    gridLength = 200
    # print "Total # of rows/columns: " + str(gridLength / int(resolution))
    simGrid, simTrans = setupSim(gridLength / resolution, numTrans)
    genFingerprints(simGrid, simTrans, gridLength / resolution, shadowDev)

    # perform Naive Bayes algorithm to generate classifier
    trainPredictor, trainTarget, testPredictor, testTarget = genData(simGrid, simTrans, 200/resolution, shadowDev)
    model = GaussianNB()
    # print "Performing Naive Bayes"
    model.fit(trainPredictor, trainTarget)

    # perform predictions, and calculate error
    # print "Predicting classes"
    predicted = model.predict(testPredictor)
    predictTup = changeToTuples(predicted, resolution)
    testTup = changeToTuples(testTarget, resolution)
    allErrs, avgErr = calcError(predictTup, testTup, resolution)

    """
    "# printing out results for debugging purposes
    printGrid(simGrid, 0)
    printGrid(simGrid, 1)
    printTrans(transmitters)
    printGrid(simGrid, 2)
    print type(predicted)
    print predicted
    print testTarget
    print predictTup
    print "\n"
    print testTup
    print "\n"
    print allErrs"""
    #print "Average Error = " + str(avgErr)

    return avgErr

# run the simulation x number of times, and return the errors and average errors across all runs
def simNum(iters, numTrans, resolution, shadowDev):
    errors = []
    sumErrs = 0
    for num in range(0, iters):
        # print "Currently running iteration " + str(num + 1)
        tmp = beginSim(numTrans, resolution, shadowDev)
        errors.append(tmp)
        sumErrs += tmp
    return errors, sumErrs/iters

def genHeatmap(resolution):
    data = []
    deviations = [1, 2, 3, 4, 5, 10]

    # generate data for each standard deviation, for each # of transmitters
    for stdDev in deviations:
        stdDevData = []
        for numTrans in range(1, 11):
            error = beginSim(numTrans, resolution, int(stdDev))
            stdDevData.append(error)
        data.append(stdDevData)
        print "Data for standard deviation " + str(stdDev) + " generated"
    print "Data for resolution " + str(resolution) + " generated"

    #display the data in a heat map
    data = np.asarray(data)
    print data
    pl.pcolor(data, cmap=matplotlib.cm.Blues)
    pl.show()


""" PROGRAM BEGINS HERE """
"""
# or start here to get options from user (NOTE: no value verification is done here)
simNumTrans = raw_input("Enter number of transmitters = [1:10]: ")
simRes = raw_input("Enter grid resolution = [1, 5, 10, 15, 20]: ")  # == size of each cell; 200/this = num cells per row
simShadowDev = raw_input("Enter the shadowing noise standard deviation = [1, 2, 3, 4, 5, 10]: ")
if not simNumTrans.isdigit() or not simRes.isdigit() or not simShadowDev.isdigit():
    print "Must be a whole number"
    sys.exit(1)
simNumTrans = int(simNumTrans)
simRes = int(simRes)
simShadowDev = int(simShadowDev)"""
# start here to use preset values
simNumTrans = 1
simRes = 1
simShadowDev = 1
iterations = 10

# allErrs, avgErr = simNum(iterations, simNumTrans, simRes, simShadowDev)

"""
allErrs, avgErr = simNum(iterations, simNumTrans, simRes, 1)
print "Total Average Error For s.d ==  1: " + str(avgErr)
allErrs, avgErr = simNum(iterations, simNumTrans, simRes, 2)
print "Total Average Error For s.d ==  2: " + str(avgErr)
allErrs, avgErr = simNum(iterations, simNumTrans, simRes, 5)
print "Total Average Error For s.d ==  5: " + str(avgErr)
allErrs, avgErr = simNum(iterations, simNumTrans, simRes, 10)
print "Total Average Error For s.d == 10: " + str(avgErr)
allErrs, avgErr = simNum(iterations, simNumTrans, simRes, 20)
print "Total Average Error For s.d == 20: " + str(avgErr)"""

#generate heatmap
#genHeatmapCellData(1)
#genHeatmapCellData(2)
#genHeatmapCellData(5)
#genHeatmapCellData(10)
heatData = genHeatmap(20)
