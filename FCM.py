# Rashmi Chaudhary

from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centroids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


#
#
#    Fuzzy C-Means Implementation
#
#
class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=4, max_iterations=15, min_distance=5.0, size=200, m=2, epsilon=.5):
        self.k = k  # initialize k clusters
        self.max_iterations = max_iterations  # intialize max_iterations
        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = .001
        self.max_diff = 0.0
        self.image = 0

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)
            # self.beta = self.calculate_beta(self.image)

        print "********************************************************************"
        for i in range(self.s):
            print self.pixels[i]

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))
        randomPixels = random.sample(self.pixels, self.k)
        print"INTIALIZE RANDOM PIXELS AS CENTROIDS"
        print randomPixels
        #    print"================================================================================"
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]
                # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print "________", self.clusters[0].pixels[0]
        iterations = 0

        self.oldClusters = [cluster.centroid for cluster in self.clusters]
        print "HELLO I AM ITERATIONS:", iterations
        self.calculate_centre_vector()
        self.update_degree_of_membership()
        iterations += 1


        # shouldExit(iterations) checks to see if the exit requirements have been met.
            # - max iterations has been reached OR the centers have converged.
        while self.shouldExit(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membership()
            iterations += 1

        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]


    def selectSingleSolution(self):
        self.max_iterations = 10

    def shouldExit(self, iterations):
        
        if iterations >= self.max_iterations:
            return True
        # if (self.max_diff > self.epsilon):
        #   return False
        # Perform normalization
        #self.normalization()
        # for i in self.s:
        return False

    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        t = []
        for i in range(self.s):
            t.append([])
            for j in range(self.k):
                t[i].append(pow(self.degree_of_membership[i][0][j], self.m))
        # print"\n\nCALC_CENTRE_VECTOR INVOKED:"

        for cluster in range(self.k):
            # print"*********************************************************************************"
            numerator = 0.0
            denominator = 0.0
            for i in range(self.s):
                # print "+++++++++", self.clusters[cluster].pixels[i], t[i][cluster], (t[i][cluster] * self.clusters[cluster].pixels[i])
                numerator += t[i][cluster] * self.clusters[cluster].pixels[i]
                denominator += (t[i][cluster])
                # print " ______ ", numerator/denominator
            self.clusters[cluster].centroid = (numerator / denominator)

    # Updates the degree of membership for all of the data points.
    def update_degree_of_membership(self):
        self.max_diff = 0.0

        for idx in range(self.k):
            for i in range(self.s):
                new_uij = self.get_new_value(self.pixels[i], self.clusters[idx].centroid)
                if (i == 0):
                    print "This is the Updatedegree centroid number:", idx, self.clusters[idx].centroid
                diff = new_uij - self.degree_of_membership[i][0][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                self.degree_of_membership[i][0][idx] = new_uij
        return self.max_diff

    def get_new_value(self, i, j):
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in self.clusters:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)
            sum += val
        return (1.0 / sum)

    def normalization(self):
        max = 0.0
        highest_index = 0
        for i in range(self.s):
            # Find the index with highest probability
            for j in range(self.k):
                if (self.degree_of_membership[i][0][j] > max):
                    max = self.degree_of_membership[i][0][j]
                    highest_index = j
            # Normalize, set highest prob to 1 rest to zero
            for j in range(self.k):
                if (j != highest_index):
                    self.degree_of_membership[i][0][j] = 0
                else:
                    self.degree_of_membership[i][0][j] = 1

    # Shows the image.
    def showImage(self):
        self.image.show()

    def showClustering(self):
        localPixels = [None] * len(self.image.getdata())
        for idx, pixel in enumerate(self.pixels):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster
            #print "cluster ", cluster, nearest.centroid


            localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w, 3))
        colourMap = Image.fromarray(localPixels)
        #colourMap.show()

        plt.imsave("outputimage.png",colourMap)

    def showScatterPlot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sum_of_overlapeed_pixels = 0

        for i in range(self.s):
            # Find the index with highest probability

            status = False
            for j in range(self.k):
                if self.degree_of_membership[i][0][j] >=0.5:
                    status = True

            if status == False:
                sum_of_overlapeed_pixels = sum_of_overlapeed_pixels+1

        print "sum of overlapped pixels ", sum_of_overlapeed_pixels 


if __name__ == "__main__":
    image = Image.open("inputimage.jpg")
    f = fcm()
    result = f.run(image)
    f.showScatterPlot()
    f.showClustering()
    # print f.I_index()
    # print f.JmFunction()
    # print f.XBindex()