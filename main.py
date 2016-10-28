import sys
import matplotlib.pyplot as plt
import matplotlib
from random import *
import random
import math

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
def printGrid(grid, content):
    if content == 0:  # print out x, y locations
        print "Printing out grid locations"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "(" + str(grid[numx][numy].x) + "," + str(grid[numx][numy].y) + ")",
            print "\n"
    elif content == 1:  # print out boolean for transmitter
        print "Printing out transmitter locations"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "[ " + str(grid[numx][numy].trans) + " ]",
            print "\n"
    elif content == 2:  # print out all feature vectors
        print "Printing out all fingerprint feature vectors"
        for numx in range (0, len(grid)):  # iterate through each row
            for numy in range (0, len(grid)):  # iterate through each column
                print "[" + str(grid[numx][numy].featureVec) + "]",
            print "\n"
    elif content == 3:  # print out all location errors
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

# generate the testing and training data feature vectors
def genData():
    return [0], [1]

""" PROGRAM BEGINS HERE"""
# start here to use preset values
numTrans = 1
resolution = 10
shadowDev = 0

"""
# or start here to get options from user (NOTE: no value verification is done here)
numTrans = raw_input("Enter number of transmitters = [1:10]: ")
resolution = raw_input("Enter grid resolution = [1, 5, 10, 15, 20]: ")
shadowDev = raw_input("Enter the shadowing noise standard deviation = [1, 2, 3, 4, 5, 10]: ")
if not numTrans.isdigit() or not resolution.isdigit() or not shadowDev.isdigit():
    print "Must be a whole number"
    sys.exit(1)
numTrans = int(numTrans)
resolution = int(resolution)
shadowDev = int(shadowDev)"""

# determine resolution and create grid and transmitters
gridLength = 200
print "Total # of rows/columns: " + str(gridLength/int(resolution)) + "\n"
grid, transmitters = setupSim(gridLength/resolution, numTrans)
genFingerprints(grid, transmitters, gridLength/resolution, shadowDev)

# perform Naive Bayes algorithm
train, test = genData()

# printing out results
#printGrid(grid, 0)
printGrid(grid, 1)
#printTrans(transmitters)
printGrid(grid, 2)
