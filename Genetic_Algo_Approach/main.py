import pandas as pd
import numpy as np
from DNA import WindFarm_DNA
import concurrent.futures
from Eval_fitness import loadPowerCurve, binWindResourceData
from Utilities import readFromFile


def firstIteration(pop_size, power_curve, wind_inst_freq, filename):
	DNA_arr, fitness_arr = np.array([]), np.array([])

	DNA_arr     = np.array([])
	fitness_arr = np.array([])
	max_DNA = None

	for i in range(1, pop_size):
		curr_DNA = WindFarm_DNA(generate_genes = True)
		curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, 525)
		
		DNA_arr     = np.append(DNA_arr, curr_DNA)
		fitness_arr = np.append(fitness_arr, curr_fitness)

		if (max_DNA is None) or (curr_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_DNA
	
	return DNA_arr, fitness_arr, max_DNA



def GetBestAndPrepareNext(parents_DNA, max_fitness):
	pop_size = 0
	DNA_arr, fitness_arr = np.array([]), np.array([])
	max_DNA = None

	for i in range(len(parents_DNA)):
		parent, partner = parents_DNA[i], parents_DNA[(i+1)%len(parents_DNA)]
		curr_DNA = parent.crossover(partner)
		if(curr_DNA != None):
			pop_size += 1
			curr_DNA.mutate()
			curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, max_fitness)
			
			DNA_arr     = np.append(DNA_arr, curr_DNA)
			fitness_arr = np.append(fitness_arr, curr_fitness)

			if (max_DNA is None) or (curr_DNA.fitness > max_DNA.fitness):
				max_DNA = curr_DNA

	return pop_size, DNA_arr, fitness_arr, max_DNA


if __name__ == "__main__":
	power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
	wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')   
	max_iterations = 500
	pop_size 	   = 512
	

	# 0th iteration
	DNA_arr, fitness_arr, max_DNA = firstIteration(pop_size, power_curve, wind_inst_freq, "test.csv")
	print("Iteration: ", 0, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)


	# Rest of the iterations
	for iteration in range(1, max_iterations):
		# Selection
		if(pop_size <= 1 or np.sum(fitness_arr) == 0):	break

		parents_DNA = np.random.choice(DNA_arr, size=pop_size, p=fitness_arr/(np.sum(fitness_arr)), replace=True)
		pop_size, DNA_arr, fitness_arr, max_DNA = GetBestAndPrepareNext(parents_DNA, max_DNA.fitness)

		print("Iteration: ", iteration, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)

	print(max_DNA.fitness)
	print(max_DNA.gene_binary)
	max_DNA.writeToFile("yup.csv")
