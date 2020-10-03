import numpy as np
from eval import modAEP, loadPowerCurve, binWindResourceData
from utils import Plotter
import concurrent.futures

class Turbine:
    def __init__(self, row, col):
        self.min_sep = 400
        self.grid_length = 400
        self.best_aep = 0
        self.best_fitness = None
        
        self.fitness = 0
        self.row = row
        self.col = col
        self.pos = [self.grid_length * (row + 0.5), self.grid_length * (row + 0.5)]
        self.fence = (row == 0 or col == 0) or (row == 9 or col == 9)

class Particle:
    def __init__(self, num_turbines: int = 50):
        self.grids = 100
        self.sep = 400
        self.shape = [10, 10]

        self.best_conf = np.zeros(self.shape)
        self.best_aep = 0
        self.best_fitness = None
        self.num_turbines = num_turbines
        self.fitness = 0

        pts = [1] * self.num_turbines + [0] * (self.grids - self.num_turbines)
        np.random.shuffle(pts)
        self.conf = np.array(pts).reshape(self.shape)

    def get_pos(self, conf):
        pos = []
        for i in range(len(conf)):
            for j in range(len(conf[i])):
                if (conf[i][j] == 1):
                    pos.append([(i+0.5)*self.sep, (j+0.5)*self.sep])
        return np.array(pos)
    
    def calc_conf(self):
        pts = [1] * self.num_turbines + [0] * (self.grids - self.num_turbines)
        np.random.shuffle(pts)
        return np.array(pts).reshape(self.shape)

class Swarm:
    def __init__(self, num_particles: int = 500, max_iterations: int = 400):
        self.power_curve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
        self.wind_bins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.sep = 400
        
        self.g_best_conf = None
        self.g_best_aep = 0
        self.g_best_fitness = None
        
        self.iter_count = 0
        self.particles = [Particle() for i in range(num_particles)]

        self.best_plotter = Plotter(1)
        self.part_plotter = Plotter(2)
    
    def calc_aep(self, pos):
        return modAEP(pos, self.power_curve, self.wind_bins)

    def calc_fitness(self, aep):
        return aep ** 2
        
    def iterate(self):
        for i in range(len(self.particles)):
            new_conf = self.particles[i].calc_conf()
            new_aep = self.calc_aep(self.get_pos(new_conf))
            new_fitness = self.calc_fitness(new_aep)

            if (new_fitness > self.particles[i].best_fitness):
                self.particles[i].best_fitness = new_fitness
                self.particles[i].best_aep = new_aep
                self.particles[i].best_conf = new_conf
            
            if (new_fitness > self.g_best_fitness):
                self.g_best_fitness = new_fitness
                self.g_best_aep = new_aep
                self.g_best_conf = new_conf
            
            self.particles[i].fitness = new_fitness
            self.particles[i].conf = new_conf
        
        return 1

    def run(self):
        for particle in self.particles:
            aep = self.calc_aep(self.get_pos(particle.conf))
            fitness = self.calc_fitness(aep)
            if (self.g_best_fitness is None or fitness > self.g_best_fitness):
                self.g_best_fitness = fitness
                self.g_best_aep = aep
                self.g_best_conf = particle.conf
            particle.best_aep = aep
            particle.best_fitness = fitness
            particle.best_conf = particle.conf

        self.log_data()
        self.draw_plot()
        self.iter_count = 0
        
        while (self.iter_count < self.max_iterations):
            self.iterate()
            self.log_data()
            self.draw_plot()
            self.iter_count = self.iter_count + 1

        self.log_data()
        self.log_file('results/grid/' + str(round(self.g_best_aep,4)) + "_" + str(round(self.g_best_fitness, 4)) + "_" + str(self.num_particles) + "_" + str(self.max_iterations) + '.csv')

    def get_pos(self, conf):
        pos = []
        for i in range(len(conf)):
            for j in range(len(conf[i])):
                if (conf[i][j] == 1):
                    pos.append([(i+0.5)*self.sep, (j+0.5)*self.sep])
        return np.array(pos)

    def draw_plot(self):
        for particle in self.particles:
            self.part_plotter.plot(self.get_pos(particle.conf))
        self.best_plotter.plot(self.get_pos(self.g_best_conf))

    def log_file(self, filename: str):
        f = open(filename, "w")
        f.write("x,y\n")
        best_pos = self.get_pos(self.g_best_conf)
        for i in range(len(best_pos)):
            f.write(str(best_pos[i][0]) + "," + str(best_pos[i][1]) + "\n")
        f.close()

    def log_data(self):
        print("Iteration: ", self.iter_count, "Best Fitness: ", self.g_best_fitness, "Best AEP: ", self.g_best_aep)