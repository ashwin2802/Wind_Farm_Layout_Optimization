import pandas as pd
import numpy as np
from DNA import WindFarm_DNA
import concurrent.futures
from Eval_fitness import loadPowerCurve, binWindResourceData


def firstIteration(pop_size, power_curve, wind_inst_freq):
	DNA_arr, fitness_arr = np.array([]), np.array([])
	max_DNA = None

	for i in range(pop_size):
		curr_DNA = WindFarm_DNA(generate_genes = True)
		curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, 515)
		
		DNA_arr     = np.append(DNA_arr, curr_DNA)
		fitness_arr = np.append(fitness_arr, curr_fitness)

		if(max_DNA==None or curr_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_DNA
	
	return DNA_arr, fitness_arr, max_DNA



def GetBestAndPrepareNext(parents_DNA, max_fitness):
	pop_size = 0
	DNA_arr, fitness_arr = np.array([]), np.array([])
	max_DNA = None

	for parent in parents_DNA:
		curr_DNA = parent.parthegenesis(radius=40)
		if(curr_DNA != None):
			pop_size += 1
			curr_DNA.mutate()
			curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, max_fitness)
			
			DNA_arr     = np.append(DNA_arr, curr_DNA)
			fitness_arr = np.append(fitness_arr, curr_fitness)

			if(max_DNA == None or curr_DNA.fitness > max_DNA.fitness):
				max_DNA = curr_DNA

	return pop_size, DNA_arr, fitness_arr, max_DNA


if __name__ == "__main__":
	power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
	wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')   
	max_iterations = 200
	pop_size 	   = 1000
	

	# 0th iteration
	DNA_arr, fitness_arr, max_DNA = firstIteration(pop_size, power_curve, wind_inst_freq)
	print("Iteration: ", 0, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)


	# Rest of the iterations
	workers = 2
	with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
		for iteration in range(1, max_iterations):
			# Selection
			if(pop_size <= 20 or np.sum(fitness_arr) == 0):	break

			ratios = (fitness_arr - np.sort(fitness_arr)[-20])
			ratios[ratios < 0] = 0
			parents_DNA = np.random.choice(DNA_arr, size=pop_size, p=ratios/(np.sum(ratios)), replace=True)
			DNA_arr, fitness_arr = np.array([]), np.array([])

			# Multithreading
			batch_size = max(32, pop_size//workers)
			future_dict = {executor.submit(GetBestAndPrepareNext, parents_DNA[i:i+batch_size], max_DNA.fitness): i for i in range(0, pop_size, batch_size)}
			
			pop_size = 0
			for future in concurrent.futures.as_completed(future_dict):
				curr_pop_size, curr_DNA_arr, curr_fitness_arr, curr_max_DNA = future.result()
				pop_size += curr_pop_size
				DNA_arr = np.concatenate((DNA_arr, curr_DNA_arr))
				fitness_arr = np.concatenate((fitness_arr, curr_fitness_arr))
				if (curr_max_DNA is not None) and (curr_max_DNA.fitness > max_DNA.fitness):
					max_DNA = curr_max_DNA
			
			print("Iteration: ", iteration, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)

	print(max_DNA.fitness)
	max_DNA.writeToFile("test2.csv")