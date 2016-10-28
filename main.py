import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.naive_bayes import GaussianNB
from random import *
import random
import math
import numpy as np


""" CLASSES AND FUNCTIONS"""
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

# print out the results of the current grid
# 0 = location; 1 = transmitters; 2 = vectors; 3 = errors)
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
    for x in range(0, len(list)):
        val = tupList[x]
        xcoor = val / resolution
        ycoor = val % resolution
        ans.append([xcoor, ycoor])
    return ans

# calculate the Euclidean distance between two tuple grids and return in separate array
def calcError(predictVals, testVals):
    ans = []
    for x in range(0, len(predictVals)):
        # calculate the Euclidean distance
        xdiff = abs(predictVals[x][0] - testVals[x][0])
        ydiff = abs(predictVals[x][1] - testVals[x][1])
        dist = math.sqrt(xdiff**2 + ydiff**2)
        ans.append(dist)

""" PROGRAM BEGINS HERE"""
# start here to use preset values
simNumTrans = 3
simRes = 10
simShadowDev = 1

"""
# or start here to get options from user (NOTE: no value verification is done here)
simNumTrans = raw_input("Enter number of transmitters = [1:10]: ")
simRes = raw_input("Enter grid resolution = [1, 5, 10, 15, 20]: ")
simShadowDev = raw_input("Enter the shadowing noise standard deviation = [1, 2, 3, 4, 5, 10]: ")
if not simNumTrans.isdigit() or not simRes.isdigit() or not simShadowDev.isdigit():
    print "Must be a whole number"
    sys.exit(1)
simNumTrans = int(simNumTrans)
simRes = int(simRes)
simShadowDev = int(simShadowDev)"""

# determine resolution and create grid and transmitters
gridLength = 200
print "Total # of rows/columns: " + str(gridLength / int(simRes)) + "\n"
simGrid, simTrans = setupSim(gridLength / simRes, simNumTrans)
genFingerprints(simGrid, simTrans, gridLength / simRes, simShadowDev)

# perform Naive Bayes algorithm to generate classifier
trainPredictor, trainTarget, testPredictor, testTarget = genData(simGrid, simTrans, simRes, simShadowDev)
model = GaussianNB()
print "Performing Naive Bayes"
model.fit(trainPredictor, trainTarget)

# perform predictions, and calculate error
# for num in range(0, len(testPredictor)):
print "Predicting classes"
predicted = model.predict(testPredictor)
predictTup = changeToTuples(predicted, simRes)
testTup = changeToTuples(testTarget, simRes)
# calcError(predictTup, testTup)


# printing out results
#printGrid(grid, 0)
#printGrid(simGrid, 1)
#printTrans(transmitters)
#printGrid(simGrid, 2)
print type(predicted)
print predicted
print testTarget
print predictTup
print "\n"
print testTup
