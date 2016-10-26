import sys
import matplotlib.pyplot as plt
import matplotlib
from random import randint

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

def setupSim(resolution, numTrans):
    # setup the grid
    grid = [[0 for x in range(resolution)] for y in range(resolution)]

    # place transmitters
    transmitters = [0] * numTrans
    for num in range(0, numTrans):
        #generate the coordinates for the transmitter
        transX = randint(0, resolution-1)
        transY = randint(0, resolution-1)

        #check to see if it overlaps
        while grid[transX][transY] == 1:
            transX = randint(0, resolution-1)
            transY = randint(0, resolution-1)

        # initialize the transmitter and mark the grid space as used
        transmitters[num] = Transmitter(transX, transY)
        grid[transX][transY] = 1  # temporary setting

    # clear the grid again
    for trans in transmitters:
        # grid[trans.x][trans.y] = 0
        x = 1

    return grid, transmitters

""" PROGRAM BEGINS HERE"""
# get options from user (NOTE: no value verification is done here)
numTrans = raw_input("Enter number of transmitters = [1:10]: ")
resolution = raw_input("Enter grid resolution = [1, 5, 10, 15, 20]: ")
shadowDev = raw_input("Enter the shadowing noise deviation = [1, 2, 3, 4, 5, 10]: ")
if not numTrans.isdigit() or not resolution.isdigit() or not shadowDev.isdigit():
    print "Must be a whole number"
    sys.exit(1)

# determine resolution and create grid and transmitters
gridLength = 200
print gridLength/int(resolution)
grid, transmitters = setupSim(gridLength/int(resolution), int(numTrans))
print grid
for trans in transmitters:
    print trans
