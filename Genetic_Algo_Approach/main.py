import pandas as pd
import numpy as np
from DNA import WindFarm_DNA
from Eval_fitness import loadPowerCurve, binWindResourceData

if __name__ == "__main__":
	power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
	wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')   
	
	max_DNA = None
	max_iterations = 500 # for debugging
	
	pop_size = 1024
	# pop_size =  32 # for debugging

	# Will be used for Selection and Crossover
	DNA_arr, fitness_arr = [], []

	# 0th iteration
	for i in range(pop_size):
		curr_DNA = WindFarm_DNA(num_of_turbines=50, x_min=50, x_max=3950, y_min=50, y_max=3950, D_min=400)
		curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, 518)
		
		DNA_arr.append(curr_DNA)
		fitness_arr.append(curr_fitness)

		if(max_DNA==None or curr_DNA.fitness > max_DNA.fitness):
			max_DNA = curr_DNA

	print("Iteration: ", 0, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)

	# Rest of the iterations
	for iteration in range(1, max_iterations):
		# Selection
		if(np.sum(fitness_arr) == 0):
			break

		parents_DNA = np.random.choice(DNA_arr, size=pop_size, p=fitness_arr/(np.sum(fitness_arr)), replace=True)

		# Crossover
		pop_size = 0
		DNA_arr, fitness_arr = [], []
		for parent in parents_DNA:
			curr_DNA = parent.parthegenesis(radius=10/((iteration//40)+1))
			if(curr_DNA != None):
				pop_size += 1
				curr_DNA.mutate()
				curr_fitness = curr_DNA.calcFitness(power_curve, wind_inst_freq, max_DNA.fitness)
				
				DNA_arr.append(curr_DNA)
				fitness_arr.append(curr_fitness)

				if(curr_DNA.fitness > max_DNA.fitness):
					max_DNA = curr_DNA

		print("Iteration: ", iteration, ", pop size: ", pop_size, ", best AEP: ", max_DNA.fitness)
		if pop_size <= 1:
			break

	print(max_DNA.fitness)
	max_DNA.writeToFile("test.csv")
