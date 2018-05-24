#Zena Nuhaily
#A5
import time                   #provides timing for benchmarks
import numpy as np      		#provides complex math and array functions
from sklearn import svm     	#provides Support Vector Regression
import csv
import math
import sys
import random
#import mkl
import mlr
import FromDataFileMLR
import FromFitnessFileMLR

class BinaryParticleSwarmO:

	def __init__(self, numOfPop, numOfFea, model):
		# Retain dimensions
		self.numOfPop	= numOfPop;
		self.numOfFea	= numOfFea;

		self.model = model; 

		self.fileW = 0

		# Initialize data matrices
		self.population = np.zeros((numOfPop, numOfFea))		# Current population. 					 (nPop x nFea)
		self.fitness = np.zeros(numOfPop);							# Fitness of current population. (nPop x 1)
		self.localBestMatrix = np.zeros((numOfPop, numOfFea)) # Local Best matrix 						 (nPop x nFea)
		self.localBestFitness = np.zeros(numOfPop)				# Local best fitness 						 (nPop x 1)
		self.globalBestRow = np.zeros(numOfFea)					# Global best row								 (nFea x 1)
		self.globalBestFitness = 0										# Global best fitness						 (scalar)
		self.velocity = np.zeros((numOfPop, numOfFea))			# Current model velocity				 (nPop x nFea)


	def getAValidRow(self, eps=0.015):
		sum = 0
		row = np.zeros(self.numOfFea)
		while(sum < 3):
			for i in range(self.numOfFea):
				r = random.random()
				if r < eps:
					row[i] = 1
				else:
					row[i] = 0
			sum = row.sum()
		return row

	def createAnOutputFile(self):
	    file_name = None
	    algorithm = None

	    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
	    if ((file_name == None) and (algorithm != None)):
	        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
	                                                alg.model.__class__.__name__, alg.gen_max, timestamp)
	    elif file_name == None:
	        file_name = "{}.csv".format(timestamp)
	    fileOut = file(file_name, 'wb')
	    fileW = csv.writer(fileOut)

	    fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', \
	                    'R2Pred_Validation', 'R2Pred_Test'])
	    self.fileW = fileW

	def initPopulation(self): 
		for i in range(self.numOfPop):
			self.population[i] = self.getAValidRow();

	def initVelocity(self): 
		for i in range(self.numOfPop):
			for j in range(self.numOfFea):
				self.velocity[i,j] = random.random()

	def initLocalBestMatrix(self): 
		self.localBestMatrix	= self.population
		self.localBestFitness = self.fitness

	def initGlobalBest(self): 
		self.globalBestRow = self.population[np.argmin(self.fitness)]
		self.globalBestFitness	= self.fitness[np.argmin(self.fitness)]

	def loadData(self):
		self.TrainX, self.TrainY, self.ValidateX, self.ValidateY, self.TestX, \
		self.TestY = FromDataFileMLR.DataMLR.getAllOfTheData()
		self.TrainX, self.ValidateX, self.TestX = \
		FromDataFileMLR.DataMLR.rescaleTheData(self.TrainX, \
				self.ValidateX, self.TestX)
				
	def initialize(self): 
		self.initPopulation()
		self.initVelocity()
		self.loadData()
		self.createAnOutputFile()
		self.updateFitness()
		self.initLocalBestMatrix()
		self.initGlobalBest()

	def updatePopulation(self): 
		alpha = 0.5
		p = (0.5 * (1+alpha))
		newPop = np.zeros((self.numOfPop, self.numOfFea))
		for i in range(self.numOfPop):
			for j in range(self.numOfFea):
				if(self.velocity[i][j] <= alpha):
					newPop[i][j] = self.population[i][j]
				elif(self.velocity[i][j] <= p):
					newPop[i][j] = self.localBestMatrix[i][j]	
				elif(self.velocity[i][j] <= 1):
					newPop[i][j] = self.globalBestRow[j]
				else:
					newPop[i][j] = self.population[i][j]
		self.population = newPop

	def updateVelocity(self):
		c1 = .5
		c2 = .5
		inertia = 0.8

		for i in range(self.numOfPop):
			for j in range(self.numOfFea):
				t1 = c1*random.random()*(self.localBestMatrix[i][j]-self.population[i][j])
				t2 = c2*random.random()*(self.globalBestRow[j]-self.population[i][j])
				self.velocity[i][j] = (inertia*self.velocity[i][j]+t1+t2)

	def updateLocalBest(self):
		for i in range(self.numOfPop):
				if self.fitness[i] < self.localBestFitness[i]:
					self.localBestFitness[i] = self.fitness[i]
					self.localBestMatrix[i] = self.population[i]

	def updateGlobalBest(self):
		localBestFit = self.localBestFitness[np.argmin(self.localBestFitness)];
		if localBestFit < self.globalBestFitness:
			self.globalBestFitness = localBestFit
			self.globalBestRow = self.localBestMatrix[np.argmin(self.localBestFitness)];

	def updateFitness(self): 
		fittingStatus, self.fitness = FromFitnessFileMLR.FitnessMLR.validate_model(True, self.model, 
													self.fileW, self.population, self.TrainX, self.TrainY,
			 										self.ValidateX, self.ValidateY, self.TestX, self.TestY)
	def performIteration(self):
		self.updatePopulation()
		self.updateFitness()
		self.updateLocalBest()
		self.updateGlobalBest()
		self.updateVelocity()

	def performIterations(self, numOfGen):
		for i in range(numOfGen):
			self.performIteration()
			print "Processing Generation " + str(i) + " of " + str(numOfGen) \
			+ "\t Gloabal Best: " + str(self.globalBestFitness)
			print str(self.velocity.max()) + "\t" + str(self.velocity.mean()) \
			+ "\t" + str(self.velocity.min())

def main():
#	mkl.set_num_threads(96)
	numOfPop	= 1000
	numOfFea	= 385
	numOfGen	= 2000
	model = mlr.MLR()

	# New BPSO model
	BPSO = BinaryParticleSwarmO(numOfPop, numOfFea, model)

	# Perform initialiation and firs iteration
	BPSO.initialize()
	BPSO.performIterations(numOfGen)

#Run Main
main()