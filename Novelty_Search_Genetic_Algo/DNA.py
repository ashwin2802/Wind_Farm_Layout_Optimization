import numpy as np
from Eval_fitness import modAEP
from Utilities import GeneratePointsInCircle, PlacePointsInCorner

class WindFarm_DNA:
	
	# Constructor (makes a random DNA), all dimension in meters
	def __init__(self, generate_genes: bool = True):
		self.genes = []
		self.fitness = 0.0
		self.initial_placed_count = 37
		self.num_of_turbines = 50
		self.x_min = 50
		self.x_max = 3950
		self.y_min = 50
		self.y_max = 3950
		self.D_min = 400

		self.genes = PlacePointsInCorner(self.x_min, self.x_max, self.y_min, self.y_max, self.initial_placed_count)

		self.x_min = 400
		self.x_max = 3600
		self.y_min = 400
		self.y_max = 3600

		if(generate_genes == False):
			return

		max_candidates = self.num_of_turbines*10
		candidates_x = self.x_min + np.random.rand(max_candidates)*(self.x_max - self.x_min)
		candidates_y = self.y_min + np.random.rand(max_candidates)*(self.y_max - self.y_min)

		candidate_index = 0
		for i in range(self.initial_placed_count, self.num_of_turbines):
			found_flag = False
			while(not found_flag and candidate_index < max_candidates):
				dists = [((gene[0] - candidates_x[candidate_index])**2 + (gene[1] - candidates_y[candidate_index])**2) for gene in self.genes]
				if(min(dists) >= self.D_min**2):
					found_flag = True
					self.genes.append([candidates_x[candidate_index], candidates_y[candidate_index]])
				
				candidate_index += 1
			assert candidate_index < max_candidates, "All candidates tested !!!"


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
		for i in range(len(self.genes)):
			f.write(str(self.genes[i][0]) + "," + str(self.genes[i][1]) + "\n")
		f.close()

	
	# Fitness function (returns floating point % of "correct" characters)
	def calcFitness(self, powerCurve, wind_inst_freq, c):
		self.fitness = modAEP(np.array(self.genes), powerCurve, wind_inst_freq)
	 
		# if(self.fitness < c-0.1):
		#	return self.fitness
	 
		return self.fitness

	
	# Crossover
	def parthegenesis(self, p:float = 0.8, radius: float = 5):
		# A new child
		child = WindFarm_DNA(generate_genes = False)

		# Allow single-parent to produce with probability p a 
		# child located in a random position within a circle of 
		# radius r centered at parent
		for i in range(self.initial_placed_count, self.num_of_turbines):
			parent_gene = self.genes[i]

			if(np.random.rand() < (1-p)):
				child.genes.append(parent_gene)
				continue

			# generate 10 random genes in the circle around current parent
			num_points = 20
			candidate_genes = GeneratePointsInCircle(n = num_points, center_x = parent_gene[0], center_y = parent_gene[1], radius = radius)
			
			# choose the one that satisfies the constraints with previous ones, if none does return None
			found_flag = False
			for i in range(num_points):
				dists = [((gene[0] - candidate_genes[i][0])**2 + (gene[1] - candidate_genes[i][1])**2) for gene in child.genes]
				if(candidate_genes[i][0]>=self.x_min and candidate_genes[i][0]<=self.x_max and candidate_genes[i][1]>=self.y_min and candidate_genes[i][1]<=self.y_max and min(dists) >= self.D_min**2):
					found_flag = True
					child.genes.append(candidate_genes[i])
					break

			if found_flag == False:
				return None

		return child

	
	# Based on a mutation probability, shift turbine's x and y=coordinate with normal distribution 
	def mutate(self, mutationRate:float = 0.2, mu: float = 0, sigma: float = 20):
		for i in range(self.initial_placed_count, len(self.genes)):
			if (np.random.rand() < mutationRate):
				self.genes[i][0] += np.random.normal(mu, sigma)
				self.genes[i][1] += np.random.normal(mu, sigma)
