import pandas as pd
import numpy as np
from DNA import WindFarm_DNA
import concurrent.futures
from Eval_fitness import loadPowerCurve, binWindResourceData
from Novelty_Search import Novelty
import copy


def firstIteration(pop_size, power_curve, wind_inst_freq):
	DNA_arr, fitness_arr = [], []
	max_DNA = None

	for i in range(pop_size):
		curr_DNA = WindFarm_DNA(generate_genes = True)

		# For novelty search
		Novelty_Obj.addListToArchive(curr_DNA.genes, offset = curr_DNA.initial_placed_count)


		curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq)
		
		DNA_arr.append(curr_DNA)
		fitness_arr.append(curr_fitness)

		if(max_DNA==None or curr_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_DNA
	
	return np.array(DNA_arr), np.array(fitness_arr), max_DNA



def GetBestAndPrepareNext(DNA_arr, p, Novelty_Obj, pop_size):
	curr_pop_size = 0
	curr_DNA_arr, fitness_arr = [], []
	max_DNA = None

	elite_count = 5
	curr_DNA_arr = DNA_arr[-elite_count:]
	fitness_arr  = [D.fitness for D in curr_DNA_arr]
	max_DNA = DNA_arr[-1]
	curr_pop_size = elite_count

	while curr_pop_size < pop_size:
		parent1, parent2 = np.random.choice(DNA_arr, size=2, p=p, replace=False)
		# curr_DNA = parent.parthegenesis(radius=40)
		if(parent1.fitness > parent2.fitness):	curr_DNA = copy.deepcopy(parent1)
		else:	curr_DNA = copy.deepcopy(parent2)

		if(curr_DNA != None):
			curr_DNA.mutate()
			
			if(curr_pop_size == elite_count):
				curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, rebuild=True, novelty_obj=None, rho=0.5, offset=curr_DNA.initial_placed_count)
			else:
				curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, rebuild=False, novelty_obj=None, rho=0.5, offset=curr_DNA.initial_placed_count)
			
			curr_pop_size += 1
			# Novelty_Obj.addListToArchive(curr_DNA.genes, offset = curr_DNA.initial_placed_count)

			
			curr_DNA_arr.append(curr_DNA)
			fitness_arr.append(curr_fitness)

			if(max_DNA == None or curr_DNA.fitness > max_DNA.fitness):
				max_DNA = curr_DNA

	return curr_pop_size, np.array(curr_DNA_arr), np.array(fitness_arr), max_DNA


if __name__ == "__main__":
	power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
	wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')   
	max_iterations = 500
	pop_size 	   = 200
	k = 5
	Novelty_Obj = Novelty(k)
	

	# 0th iteration
	DNA_arr, fitness_arr, max_DNA = firstIteration(pop_size, power_curve, wind_inst_freq)
	print("Iteration: ", 0, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)


	# Rest of the iterations
	for iteration in range(1, max_iterations):
		# Selection
		if(pop_size <= 20 or np.sum(fitness_arr) == 0):	break
		
		DNA_arr, fitness_arr = zip(*(sorted(zip(DNA_arr, fitness_arr), key=lambda pair: pair[1])))
		DNA_arr, fitness_arr = list(DNA_arr), list(fitness_arr)

		ratios = fitness_arr
		# parents_DNA = np.random.choice(DNA_arr, size=pop_size, p=ratios/(np.sum(ratios)), replace=True)

		pop_size, DNA_arr, fitness_arr, curr_max_DNA = GetBestAndPrepareNext(DNA_arr, ratios/np.sum(ratios), Novelty_Obj, pop_size)
		if(curr_max_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_max_DNA

		if iteration%50 == 0:
			print("Iteration: ", iteration, ", pop size: ", pop_size, ", best AEP: ", curr_max_DNA.fitness)


	print(max_DNA.fitness)
	max_DNA.writeToFile("test2.csv")
