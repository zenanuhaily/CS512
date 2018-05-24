#Zena Nuhaily
#A5
# MainMLR.py
import time                 #provides timing for benchmarks
import numpy as np      #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv
import math
import sys
import random

import mlr
import FromDataFileMLR
import FromFinessFileMLR

# ----------------------------------------------------------------------------
class BPSO:
# ----------------------------------------------------------------------------
	@staticmethod
	def getAValidrow(numOfFea, eps=0.015):
	    sum = 0
	    while (sum < 3):
	        V = np.zeros(numOfFea)
	        for j in range(numOfFea):
	            r = random.uniform(0, 1)
	            if (r < eps):
	                V[j] = 1
	            else:
	                V[j] = 0
	        sum = V.sum()
	    return V

# ----------------------------------------------------------------------------
	@staticmethod
	def CreateInitialPopulation(numOfPop, numOfFea):
		population = np.zeros((numOfPop,numOfFea))
 	  	for i in range(numOfPop):
			population[i] = BPSO.getAValidrow(numOfFea)
		return population

# ----------------------------------------------------------------------------
	@staticmethod
	def CreateANewPopulation(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, localfitness, \
	velocity, numOfGen, model, fileW):
		V = BPSO.CreateInitialVelocity(numOfPop, numOfFea) #initialize velocity
		alpha = 0.5
		newPop = np.zeros((numOfPop, numOfFea))
		p = ((0.5 * (1 + alpha)))
		for i in range(0, numOfPop):
			for j in range(0, numOfFea):
				if (V[i][j] <= alpha):
					newPop[i][j] = population[i][j]
				elif (alpha < V[i][j] <= p):
					newPop[i][j] = localBestMatrix[i][j]
				elif (p < V[i][j] <= V[i][j]):
					newPop[i][j] = globalBestRow[j]
				else:
					newPop[i][j] = population[i][j]

		# generate new population fitness
		#TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.DataMLR.getAllOfTheData()
		#TrainX, ValidateX, TestX = FromDataFileMLR.DataMLR.rescaleTheData(TrainX, ValidateX, TestX)
		#fittingStatus, newFitness = FromFinessFileMLR.FitnessMLR.validate_model(False, model,fileW, population, \
		#						 TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

		# update local best matrix
		#if BPSO.UpdateNewLocalBest(numOfPop, numOfFea, localfitness, localBestMatrix, population, globalBestRow,\
		#newFitness, newPop, velocity, numOfGen, model, fileW):
		#	population = newPop
		#	populationFitness = newFitness

		#BPSO.UpdateVelocityMatrix(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, velocity)

		#alpha = alpha - (.17/numOfGen) #update alpha

		return newPop
# ----------------------------------------------------------------------------
	@staticmethod
	def CreateInitialVelocity(numOfPop, numOfFea):
		V = np.zeros((numOfPop, numOfFea))
		for i in range(0, numOfPop):
			for j in range(0, numOfFea):
				V[i,j] = random.uniform(0,1)
		return V

# ----------------------------------------------------------------------------
	@staticmethod
	def CreateInitialLocalBestMatrix(localBestMatrix, population, localfitness, fitness):
		localBestMatrix = population
		localfitness = fitness
		return localBestMatrix, localfitness

# ----------------------------------------------------------------------------
	@staticmethod
	def CreateInitialGlobalBestRow(globalBestRow, population, fitness):
		globalBestRow = population[np.argmin(fitness)]
		return population[np.argmin(fitness)], fitness[np.argmin(fitness)]

# ----------------------------------------------------------------------------
	@staticmethod
	def UpdateVelocityMatrix(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, velocity):
		c1 = 2
		c2 = 2
		inertiaWeight = 0.9

		for i in range(0, numOfPop):
			for j in range(0, numOfFea):
				term1 = c1 * random.random() * (localBestMatrix[i][j]-population[i][j])
				term2 = c2*random.random() * (globalBestRow[j]-population[i][j])
				velocity[i][j] = (inertiaWeight * velocity[i][j] + term1 + term2)
# ----------------------------------------------------------------------------
	@staticmethod
	def UpdateNewLocalBest(numOfPop, numOfFea, localfitness, localBestMatrix, population, globalBestRow, newFitness):
		#newPop = BPSO.CreateANewPopulation(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, localfitness,\
		 #velocity, numOfGen, model, fileW)
		for i in range(0, numOfPop):
			if newFitness[i] < localfitness[i]:
				localfitness[i] = newFitness[i]
				localBestMatrix[i] = population[i]


		return localBestMatrix, localfitness
		
	@staticmethod
	def UpdateNewGlobalBest(numOfPop, localFitness, localBestMatrix, globalBestRow, globalBestFitness):
		for i in range(0, numOfPop):
			if localFitness[i] < globalBestFitness:
				globalBestFitness = localFitness[i]
				globalBestRow = localBestMatrix[i]; 
		return globalBestRow, globalBestFitness
# ----------------------------------------------------------------------------
	def iterate():
		while numOfGen <= numOfIterations: # not at max fit
			BPSO.CreateANewPopulation() # get new population and validate
			print 
			numOfGen = numOfGen + 1 # update generation count
# ----------------------------------------------------------------------------
	@staticmethod
	def createAnOutputFile():
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

	    return fileW

#--------------------------------------------------------------------------
def main():
	model = mlr.MLR() # create an object of MLR model

	# initialize variables
	fileW = BPSO.createAnOutputFile()
	numOfPop = 5000
	numOfFea = 385
	numOfGen = 1
	"""
	population = BPSO.CreateInitialPopulation(numOfPop, numOfFea)
	velocity = BPSO.CreateInitialVelocity(numOfPop, numOfFea)
	populationFitness = BPSO.CreateInitialPopulation(numOfPop, numOfFea)
	localBestMatrix = population
	TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.DataMLR.getAllOfTheData()
	TrainX, ValidateX, TestX = FromDataFileMLR.DataMLR.rescaleTheData(TrainX, ValidateX, TestX)
	fittingStatus, fitness = FromFinessFileMLR.FitnessMLR.validate_model(True, model,fileW, population, \
							 TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
	localfitness = fitness
	globalBestRow = population[np.argmin(fitness)]
	newPop = BPSO.CreateANewPopulation(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, \
	localfitness, velocity, numOfGen, model, fileW)
	fittingStatus, newFitness = FromFinessFileMLR.FitnessMLR.validate_model(False, model,fileW, newPop, \
						 TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


	#BPSO.CreateInitialPopulation(numOfPop, numOfFea) # step 1
	#BPSO.getAValidrow(numOfFea, eps=0.015) # step 2
	#BPSO.CreateInitialVelocity(numOfPop, numOfFea) # step 4
	#BPSO.CreateInitialLocalBestMatrix(localBestMatrix, population, localfitness, fitness) # step 5
	#BPSO.CreateInitialGlobalBestRow(globalBestRow, population, fitness) # step 6
	BPSO.CreateANewPopulation(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, localfitness,\
	 velocity, numOfGen, model, fileW) # step 7
	BPSO.UpdateNewLocalBest(numOfPop, numOfFea, localfitness, localBestMatrix, population, globalBestRow,\
	 newFitness, newPop, velocity, numOfGen, model, fileW) # step 8 & 9
	BPSO.UpdateVelocityMatrix(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, velocity) # step 10

	BPSO.iterate() # training matrix (step 3) """
	
	population = BPSO.CreateInitialPopulation(numOfPop, numOfFea)
	velocity = BPSO.CreateInitialVelocity(numOfPop, numOfFea)
	
	TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.DataMLR.getAllOfTheData()
	TrainX, ValidateX, TestX = FromDataFileMLR.DataMLR.rescaleTheData(TrainX, ValidateX, TestX)
	fittingStatus, fitness = FromFinessFileMLR.FitnessMLR.validate_model(True, model,fileW, population, \
							 TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
							 

	localBestMatrix = np.zeros((numOfPop, numOfFea))
	localFitness = np.zeros((numOfPop, 1)); 
	globalBestRow = np.zeros(numOfFea); 
	
	localBestMatrix, localFitness = BPSO.CreateInitialLocalBestMatrix(localBestMatrix, population, localFitness, fitness)
	globalBestRow, globalBestFitness = BPSO.CreateInitialGlobalBestRow(globalBestRow, population, fitness) # step 6
	
	for i in range(2000):
		BPSO.UpdateVelocityMatrix(numOfPop,numOfFea, localBestMatrix, population, globalBestRow, velocity)
		
		population = BPSO.CreateANewPopulation(numOfPop, numOfFea, localBestMatrix, population, globalBestRow, localFitness,\
			velocity, numOfGen, model, fileW) # step 7
			
		fittingStatus, fitness = FromFinessFileMLR.FitnessMLR.validate_model(True, model,fileW, population, \
				TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
				
		localBestMatrix, localFitness, BPSO.UpdateNewLocalBest(numOfPop, numOfFea,
			localFitness, localBestMatrix, population, globalBestRow, fitness)
		
		globalBestRow, globalBestFitness = BPSO.UpdateNewGlobalBest(numOfPop,
				localFitness, localBestMatrix, globalBestRow, globalBestFitness);
		
		print "Global Best: " + str(globalBestFitness)
#----------------------------------------------------------------------------
main()
