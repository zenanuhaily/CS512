import time                 #provides timing for benchmarks
import numpy as np      #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv
import math
import sys
import random

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

# ----------------------------------------------------------------------------------------------------------------------
class GA:
	@staticmethod
	def crossover(dad, mom):
		pos = random.randint(1, dad.shape[0] - 1)

		child1 = np.append(dad[:pos], mom[pos:])
		child2 = np.append(mom[:pos], dad[pos:])

		GA.mutation(child1)
		GA.mutation(child2)

		return child1, child2

	@staticmethod
	def mutation(child, chance=.0005):
		for i in range(len(child)):
			if random.uniform(0,1) < chance:
				child[i] = 1 - child[i]
		return child


	# ------------------------------------------------------------------------------
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

	    # ------------------------------------------------------------------------------

	@staticmethod
	def Create_A_Population(numOfPop, numOfFea):
	    population = np.zeros((numOfPop, numOfFea))
	    for i in range(numOfPop):
	        V = GA.getAValidrow(numOfFea)
	        population[i] = V
	    return population

# ------------------------------------------------------------------------------
# The following creates an output file. Every time a model is created the
# descriptors of the model, the ame of the model (ex: "MLR" for multiple
# linear regression of "SVM" support vector machine) the R^2 of training, Q^2
# of training,R^2 of validation, and R^2 of test is placed in the output file

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

# -------------------------------------------------------------------------------------------

	@staticmethod
	def createANewPopulation(numOfPop, numOfFea, OldPopulation, fitness):
	    #   NewPopulation = create a 2D array of (numOfPop by num of features)
	    #   sort the OldPopulation and their fitness value based on the asending
	    #   order of the fitness. The lower is the fitness, the better it is.
	    #   So, Move two rows with of the OldPopulation with the lowest fitness
	    #   to row 1 and row 2 of the new population.
	    #
	    #   Name the first row to be Dad and the second row to be mom. Create a
	    #   one point or n point cross over to create at least couple of children.
	    #   children should be moved to the third, fourth, fifth, etc rows of the
	    #   new population.
	    #   The rest of the rows should be filled randomly the same way you did when
	    #   you created the initial population.
	    # -------------
	    # OldPopulation = createANewPopulation()
	    # np.sort()
	    # -----------------
	    NewPopulation = np.zeros((numOfPop, numOfFea))
	    idx = fitness.argsort(axis=0)
	    OldPopulation = OldPopulation[idx]

	    NewPopulation[0] = OldPopulation[0]
	    NewPopulation[1] = OldPopulation[1]

	    mom = NewPopulation[0]
	    dad = NewPopulation[1]
	    child1, child2 = GA.crossover(mom, dad)

	    NewPopulation[2] = child1
	    NewPopulation[3] = child2

	    for i in range(4, numOfPop):
	        row = GA.getAValidrow(numOfFea)
	        NewPopulation[i] = row

	        return NewPopulation
	    # -------------------------------------------------------------------------------------------
	@staticmethod
	def PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW, \
	                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
	    NumOfGenerations = 1
	    OldPopulation = population
	    while (NumOfGenerations < 1000):
	        population = GA.createANewPopulation(numOfPop, numOfFea, OldPopulation, fitness)
	        fittingStatus, fitness = FromFinessFileMLR.FitnessMLR.validate_model(model, fileW, population, \
	                                                                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
	        NumOfGenerations = NumOfGenerations + 1
	        OldPopulation = population
	        print('gen #' + str(NumOfGenerations) + '\t\t' + str(fitness.min()))

#-----------------------------------------------------------------------------------------------------------------------
def main():
    random.seed()
    # create an output file. Name the object to be FileW
    fileW = GA.createAnOutputFile()


    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()

    #Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385


    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train    = .6
    R2req_validate = .5
    R2req_test     = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.DataMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.DataMLR.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    population = GA.Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = FromFinessFileMLR.FitnessMLR.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    GA.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW, \
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#main routine ends in here

#------------------------------------------------------------------------------

main()
