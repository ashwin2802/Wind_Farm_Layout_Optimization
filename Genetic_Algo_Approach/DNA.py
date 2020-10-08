import numpy as np
from Eval_fitness import modAEP
from Utilities import GeneratePointsInCircle, PlacePointsInCorner

class WindFarm_DNA:
	
	# Constructor (makes a random DNA), all dimension in meters
	def __init__(self, generate_genes: bool = True):
		self.genes = []
		self.fitness = 0.0
		self.initial_placed_count = 36

		self.num_of_turbines = 50
		self.x_min = 50
		self.x_max = 3950
		self.y_min = 50
		self.y_max = 3950
		self.D_min = 400

		self.to_place = self.num_of_turbines - self.initial_placed_count
		self.genes = PlacePointsInCorner(self.x_min, self.x_max, self.y_min, self.y_max, self.initial_placed_count)

		self.x_min = 400
		self.x_max = 3600
		self.y_min = 400
		self.y_max = 3600

		self.gene_binary = np.zeros(64) # to be interpreted as boolean array of size 8*8
		self.cell_size = 400

		if(generate_genes == False or self.to_place == 0):
			return

		# generate random integers to be place turbines
		indices = np.random.choice(self.gene_binary.shape[0], self.to_place, replace=False)
		self.gene_binary[indices] = 1


	# Printing the DNA in string format
	def __str__(self):
		break_after = 5	# number of lines before next newline
		for i in range(len(self.genes)):
			print(self.genes[i], end='\t')
			if((i+1)%break_after == 0):
				print()

		print("\nFitness Value: ", self.fitness)


	# Write to file
	def writeToFile(self, filename: str):
		f = open(filename, "w")
		f.write("x,y\n")

		temp_genes = self.getGenesFromBinary()
		genes = np.concatenate((self.genes, temp_genes))
		for i in range(self.num_of_turbines):
			f.write(str(genes[i][0]) + "," + str(genes[i][1]) + "\n")
		f.close()


	def getGenesFromBinary(self):
		filled_indices = np.where(self.gene_binary == 1)[0]
		x, y = filled_indices//8, filled_indices%8
		x_cord, y_cord = self.cell_size*x + (self.cell_size//2), self.cell_size*y + (self.cell_size//2)
		
		res = [[self.x_min + x_cord[i], self.y_min + y_cord[i]] for i in range(self.to_place)]
		return res

	
	# Fitness function (returns floating point % of "correct" characters)
	def calcFitness(self, powerCurve, wind_inst_freq, c):
		temp_genes = self.getGenesFromBinary()
		self.fitness = modAEP(np.concatenate((self.genes, temp_genes)), powerCurve, wind_inst_freq)
	 
		# if(self.fitness < c-0.1):
		#	return self.fitness
	 
		return 5*(10*(self.fitness - c))

	
	# Crossover
	def crossover(self, partner):
		# A new child
		child = WindFarm_DNA(generate_genes = False)

		indices1 = np.where(self.gene_binary == 1)[0]
		indices2 = np.where(partner.gene_binary == 1)[0]

		indices = np.unique(np.concatenate((indices1, indices2)))
		child_filled_indices = np.random.choice(indices, self.to_place, replace=False)

		child.gene_binary[child_filled_indices] = 1
		assert(len(child_filled_indices) == len(set(child_filled_indices)))
		return child

	
	# Based on a mutation probability, shift turbine's x and y=coordinate with normal distribution 
	def mutate(self, mutationRate:float = 0.2):
		filled_indices = np.where(self.gene_binary == 1)[0]
		for i in range(filled_indices.shape[0]):
			if(np.random.rand() < mutationRate):
				opts = findEmptyNeighbours(i, filled_indices)
				new_index = np.random.choice(opts)
				filled_indices[i] = new_index

		self.gene_binary = np.zeros(self.gene_binary.shape[0])
		assert(len(filled_indices) == len(set(filled_indices)))
		self.gene_binary[filled_indices] = 1



def findEmptyNeighbours(i, arr, rows=8, cols=8):
	index = arr[i]
	x, y = index//rows, index%cols

	res_x_y = []

	if x!=0 and (((x-1)*rows + y) not in arr):			res_x_y.append((x-1, y))
	if y!=0 and ((x*rows + (y-1)) not in arr):			res_x_y.append((x, y-1))
	if (x+1)!=rows and (((x+1)*rows + y) not in arr):	res_x_y.append((x+1, y))
	if (y+1)!=cols and ((x*rows + (y+1)) not in arr):	res_x_y.append((x, y+1))

	if(len(res_x_y) == 0):
		return [index]

	res = [a*rows + b for (a,b) in res_x_y]
	return res