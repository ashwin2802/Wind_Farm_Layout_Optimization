import numpy as np
from eval import getAEP, loadPowerCurve, binWindResourceData, preProcessing, checkConstraints

class Particle:
    def __init__(self, num_turbines: int = 50):
        self.min_sep = 400
        self.length = 4000
        self.fence = 50
        
        self.pos = np.ones([num_turbines, 2]) * self.fence + np.random.uniform(size=[num_turbines, 2]) * (self.length - 2 * self.fence)
        self.pos_history = []
        
        self.best_pos = None
        self.best_aep = 0
        
        self.fitness = 0
        self.shape= [num_turbines, 2]
    
    def calc_violation(self):
        return 0

    def calc_position(self, vel_1, vel_2, global_best):
        return (self.pos + np.multiply(vel_1, (global_best - self.pos)) + np.multiply(vel_2, (self.best_pos - self.pos)))

    def log_data(self, index):
        print("Index: ", index, "Best AEP: ", self.best_aep, "Curr Fitness: ", self.fitness)

class Swarm:
    def __init__(self, num_particles: int = 500, max_iterations: int = 400, epsilon: float = 0.1):
        self.powerCurve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
        self.windBins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
        self.n_wind_instances, self.cos_dir, self.sin_dir, self.wind_sped_stacked, self.C_t = preProcessing(self.powerCurve)
        self.turb_diam = 50

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.eps = epsilon
        
        self.best_pos = None
        self.best_aep = 0
        self.best_fitness = -1
        
        self.iter_count = 0
        self.particles = [Particle() for i in range(num_particles)]
        
    def iterate(self):
        for i in range(len(self.particles)):
            particle = self.particles[i]
            v1 = np.random.uniform(low=0, high=1, size=particle.shape)
            v2 = np.random.uniform(low=-1, high=1, size=particle.shape)
            new_pos = particle.calc_position(v1, v2, self.best_pos)
            
            viol_cost = particle.calc_violation()
            new_aep = self.calc_aep(particle.pos)
            new_fitness = new_aep - viol_cost

            if (new_fitness > particle.fitness):
                particle.best_pos = new_pos
                particle.best_aep = new_aep

            if (new_fitness > self.best_fitness):
                self.best_fitness = new_fitness
                self.best_aep = new_aep
                self.best_pos = new_pos

            particle.fitness = new_fitness
            particle.pos_history.append(particle.pos)
            particle.pos = new_pos

            # particle.log_data(i)

    def log_file(self, filename: str):
        f = open(filename, "w")
        f.write("x,y\n")
        for i in range(len(self.best_pos)):
            f.write(str(self.best_pos[i][0]) + "," + str(self.best_pos[i][1]) + "\n")
        f.close()

    def calc_aep(self, pos):
        # if not checkConstraints(pos, self.turb_diam):
        #     return 0
        
        aep = getAEP(self.turb_diam / 2, pos, self.powerCurve, self.windBins, self.n_wind_instances, self.cos_dir, self.sin_dir, self.wind_sped_stacked, self.C_t)

        return aep

    def run(self):
        for particle in self.particles:
            viol_cost = particle.calc_violation()
            aep = self.calc_aep(particle.pos)
            particle.fitness = aep - viol_cost
            if (particle.fitness > self.best_fitness):
                self.best_fitness = particle.fitness
                self.best_pos = particle.pos
                self.best_aep = aep
            particle.best_pos = particle.pos

        self.log_data()

        self.iter_count = 1
        while (self.iter_count < self.max_iterations):
            self.iterate()
            self.log_data()
            self.iter_count = self.iter_count + 1
            # terminate
        
        self.log_data()
        self.log_file('results/result_' + str(round(self.best_aep,4)) + '.csv')

    def log_data(self):
        print("Iteration: ", self.iter_count, "Best Fitness: ", self.best_fitness, "Best AEP: ", self.best_aep)

