import pandas as pd
import numpy as np
from DNA import WindFarm_DNA
import concurrent.futures
from Eval_fitness import loadPowerCurve, binWindResourceData
from Novelty_Search import Novelty


def firstIteration(pop_size, power_curve, wind_inst_freq):
	DNA_arr, fitness_arr = np.array([]), np.array([])
	max_DNA = None

	for i in range(pop_size):
		curr_DNA = WindFarm_DNA(generate_genes = True)

		# For novelty search
		Novelty_Obj.addListToArchive(curr_DNA.genes, offset = curr_DNA.initial_placed_count)


		curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq)
		
		DNA_arr     = np.append(DNA_arr, curr_DNA)
		fitness_arr = np.append(fitness_arr, curr_fitness)

		if(max_DNA==None or curr_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_DNA
	
	return DNA_arr, fitness_arr, max_DNA



def GetBestAndPrepareNext(DNA_arr, p, pop_size):
	curr_pop_size = 0
	curr_DNA_arr, fitness_arr = np.array([]), np.array([])
	max_DNA = None

	while curr_pop_size != pop_size:
		parent = np.random.choice(DNA_arr, p=p)
		curr_DNA = parent.parthegenesis(radius=40)
		if(curr_DNA != None):
			curr_pop_size += 1
			curr_DNA.mutate()
			
			if(curr_pop_size == 1):
				curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, rebuild=True, novelty_obj=Novelty_Obj, rho=1, offset=curr_DNA.initial_placed_count)
			else:
				curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, rebuild=False, novelty_obj=Novelty_Obj, rho=1, offset=curr_DNA.initial_placed_count)
			
			Novelty_Obj.addListToArchive(curr_DNA.genes, offset = curr_DNA.initial_placed_count)

			
			curr_DNA_arr = np.append(curr_DNA_arr, curr_DNA)
			fitness_arr  = np.append(fitness_arr, curr_fitness)

			if(max_DNA == None or curr_DNA.fitness > max_DNA.fitness):
				max_DNA = curr_DNA

	return curr_pop_size, curr_DNA_arr, fitness_arr, max_DNA


if __name__ == "__main__":
	power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
	wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')   
	max_iterations = 300
	pop_size 	   = 500
	k = 5
	Novelty_Obj = Novelty(k)
	

	# 0th iteration
	DNA_arr, fitness_arr, max_DNA = firstIteration(pop_size, power_curve, wind_inst_freq)
	print("Iteration: ", 0, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)


	# Rest of the iterations
	for iteration in range(1, max_iterations):
		# Selection
		if(pop_size <= 80 or np.sum(fitness_arr) == 0):	break

		ratios = (fitness_arr - np.sort(fitness_arr)[-80])
		ratios[ratios < 0] = 0
		# parents_DNA = np.random.choice(DNA_arr, size=pop_size, p=ratios/(np.sum(ratios)), replace=True)

		pop_size, DNA_arr, fitness_arr, curr_max_DNA = GetBestAndPrepareNext(DNA_arr, ratios/np.sum(ratios), pop_size)
		if(curr_max_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_max_DNA

		print("Iteration: ", iteration, ", pop size: ", pop_size, ", best AEP: ", curr_max_DNA.fitness)


	print(max_DNA.fitness)
	max_DNA.writeToFile("test2.csv")
