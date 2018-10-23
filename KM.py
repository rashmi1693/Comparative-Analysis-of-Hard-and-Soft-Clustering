'''
	Rashmi Chaudhary
    IMAGE SEGMENTATION USING K-MEANS (UNSUPERVISED LEARNING)
	Note: K is greater than 2
'''

import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

iterations = 15

#	Parse command-line arguments
#	sets K, input & output
if len(sys.argv) < 4:
	print "Error: Insufficient arguments, imageSegmentation takes three arguments"
	sys.exit()
else:
	K = int(sys.argv[1])
	if K < 3:
		print "Error: K has to be greater than 2"
		sys.exit()
	input = sys.argv[2]
	output = sys.argv[3]

#	Open input image
image = Image.open(input)
imageW = image.size[0]
imageH = image.size[1]

#	Initialise data vector with attribute r,g,b,x,y for each pixel
dataVector = np.ndarray(shape=(imageW * imageH, 5), dtype=float)
#	Initialise vector that holds which cluster a pixel is currently in
pixelCluster = np.ndarray(shape=(imageW * imageH), dtype=int)

#	Populate data vector with data from input image
#	dataVector has 5 fields: red, green, blue, x coord, y coord
for y in range(0, imageH):
      for x in range(0, imageW):
      	xy = (x, y)
      	rgb = image.getpixel(xy)
      	dataVector[x + y * imageW, 0] = rgb[0]
      	dataVector[x + y * imageW, 1] = rgb[1]
      	dataVector[x + y * imageW, 2] = rgb[2]
      	dataVector[x + y * imageW, 3] = x
      	dataVector[x + y * imageW, 4] = y

#	Standarize the values of our features
dataVector_scaled = preprocessing.normalize(dataVector)

#	Set centers
minValue = np.amin(dataVector_scaled)
maxValue = np.amax(dataVector_scaled)

centers = np.ndarray(shape=(K,5))
for index, center in enumerate(centers):
	centers[index] = np.random.uniform(minValue, maxValue, 5)

for iteration in xrange(iterations):
	#	Set pixels to their cluster
	for idx, data in enumerate(dataVector_scaled):
		distanceToCenters = np.ndarray(shape=(K))
		for index, center in enumerate(centers):
			distanceToCenters[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
		pixelCluster[idx] = np.argmin(distanceToCenters)

	##################################################################################################
	#	Check if a cluster is ever empty, if so append a random datapoint to it
	clusterToCheck = np.arange(K)		#contains an array with all clusters
										#e.g for K=10, array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	clustersEmpty = np.in1d(clusterToCheck, pixelCluster)
										#^ [True True False True * n of clusters] False means empty
	for index, item in enumerate(clustersEmpty):
		if item == False:
			pixelCluster[np.random.randint(len(pixelCluster))] = index
			# ^ sets a random pixel to that cluster as mentioned in the homework writeup
	##################################################################################################

	#	Move centers to the centroid of their cluster
	for i in xrange(K):
		dataInCenter = []

		for index, item in enumerate(pixelCluster):
			if item == i:
				dataInCenter.append(dataVector_scaled[index])
		dataInCenter = np.array(dataInCenter)
		centers[i] = np.mean(dataInCenter, axis=0)

	#TODO check for convergence
	print "Centers Iteration num", iteration, ": \n", centers

#	set the pixels on original image to be that of the pixel's cluster's centroid
for index, item in enumerate(pixelCluster):
	dataVector[index][0] = int(round(centers[item][0] * 255))
	dataVector[index][1] = int(round(centers[item][1] * 255))
	dataVector[index][2] = int(round(centers[item][2] * 255))

#	Save image
image = Image.new("RGB", (imageW, imageH))

for y in xrange(imageH):
	for x in xrange(imageW):
	 	image.putpixel((x, y), (int(dataVector[y * imageW + x][0]),
	 							int(dataVector[y * imageW + x][1]),
	 							int(dataVector[y * imageW + x][2])))
image.save(output)
