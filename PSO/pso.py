import numpy as np
from eval import getAEP, loadPowerCurve, binWindResourceData, preProcessing, checkConstraints
from utils import Plotter
import csv

class Particle:
    def __init__(self, num_turbines: int = 50):
        self.min_sep = 400
        self.length = 4000
        self.fence = 50

        self.shape= [num_turbines, 2]
        self.best_pos = np.zeros(self.shape)
        self.best_aep = 0
        self.best_fitness = None
        
        self.fitness = 0
        self.pos_history = []
        
        self.pos = np.ones([num_turbines, 2]) * self.fence + np.random.uniform(size=[num_turbines, 2]) * (self.length - 2 * self.fence)
        self.pos = self.calc_position(np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape))
    
    def calc_violation(self, pos):
        cost = 0
        for i in range(len(pos)):
            x_1, y_1 = pos[i]
            if (x_1 < self.fence):
                cost = cost + (self.fence - x_1)**2
            if (y_1 < self.fence):
                cost = cost + (self.fence - y_1)**2
            if (x_1 > self.length - self.fence):
                cost = cost + (self.length - self.fence - x_1)**2
            if (y_1 > self.length - self.fence):
                cost = cost + (self.length - self.fence - y_1)**2

            for j in range(len(pos)):
                if (i == j):
                    continue
                x_2, y_2 = pos[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                if (dist < self.min_sep):
                    cost = cost + (self.min_sep - dist)
        return cost

    def calc_position(self, vel_1, vel_2, global_best):
        new_pos = self.pos + np.multiply(vel_1, (global_best - self.pos)) + np.multiply(vel_2, (self.best_pos - self.pos))
        for i in range(len(new_pos)):
            for j in range(len(new_pos)):
                if (i == j):
                    continue

                x_1, y_1 = new_pos[i]
                x_2, y_2 = new_pos[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                if (dist < self.min_sep):
                    radius = (self.min_sep - dist) / 2
                    angle = np.arctan((y_2 - y_1) / (x_2 - x_1 + 1e-3))
                    # turn = np.random.uniform(low = -1.57, high = 1.57)
                    turn = 0
                    c=(x_2 * y_1 - x_1 * y_2) / (x_2 - x_1 + 1e-3)
                    
                    s_x_1 = x_1
                    s_y_1 = y_1 + c
                    s_x_2 = x_2
                    s_y_2 = y_2 + c

                    r_x_1 = s_x_1 * np.cos(angle) + s_y_1 * np.sin(angle) + radius*np.cos(turn)
                    r_y_1 = s_y_1 * np.cos(angle) - s_x_1 * np.sin(angle) + radius*np.sin(turn)
                    r_x_2 = s_x_2 * np.cos(angle) + s_y_2 * np.sin(angle) - radius*np.cos(turn)
                    r_y_2 = s_y_2 * np.cos(angle) - s_x_2 * np.sin(angle) - radius * np.sin(turn)
                    
                    new_pos[i] = [r_x_1 * np.cos(angle) - r_y_1 * np.sin(angle), r_x_1 * np.sin(angle) + r_y_1 * np.cos(angle) - c]
                    new_pos[j] = [r_x_2 * np.cos(angle) - r_y_2 * np.sin(angle), r_x_2 * np.sin(angle) + r_y_2 * np.cos(angle) - c]

            new_pos[i][0] = min(new_pos[i][0], self.length - self.fence)
            new_pos[i][1] = min(new_pos[i][1], self.length - self.fence)
            new_pos[i][0] = max(self.fence, new_pos[i][0])
            new_pos[i][1] = max(self.fence, new_pos[i][1])
                
        return new_pos

    def log_data(self, index):
        print("Index: ", index, "Best AEP: ", self.best_aep, "Curr Fitness: ", self.fitness)

class Swarm:
    def __init__(self, num_particles: int = 500, max_iterations: int = 400, epsilon: float = 0.1):
        self.power_curve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
        self.wind_bins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.eps = epsilon
        
        self.g_best_pos = None
        self.g_best_aep = 0
        self.g_best_fitness = None
        
        self.iter_count = 0
        self.particles = [Particle() for i in range(num_particles)]
        
    def iterate(self):
        for i in range(len(self.particles)):
            v1 = np.random.uniform(low=0, high=1, size=self.particles[i].shape)
            v2 = np.random.uniform(low=0, high=1, size=self.particles[i].shape)
            new_pos = self.particles[i].calc_position(v1, v2, self.g_best_pos)
            
            viol_cost = self.particles[i].calc_violation(self.particles[i].pos)
            new_aep = self.calc_aep(new_pos)
            new_fitness = self.calc_fitness(new_aep, viol_cost)

            if (new_fitness > self.particles[i].best_fitness):
                self.particles[i].best_fitness = new_fitness
                self.particles[i].best_pos = new_pos
                self.particles[i].best_aep = new_aep

            if (new_fitness > self.g_best_fitness):
                self.g_best_fitness = new_fitness
                self.g_best_aep = new_aep
                self.g_best_pos = new_pos

            self.particles[i].fitness = new_fitness
            self.particles[i].pos_history.append(self.particles[i].pos) # check these and local bests also
            self.particles[i].pos = new_pos

    def log_file(self, filename: str):
        f = open(filename, "w")
        f.write("x,y\n")
        for i in range(len(self.g_best_pos)):
            f.write(str(self.g_best_pos[i][0]) + "," + str(self.g_best_pos[i][1]) + "\n")
        f.close()

    def calc_aep(self, pos):
        return modAEP(pos, self.power_curve, self.wind_bins)

    def calc_fitness(self, aep, vc):
        if (vc != 0):
            return -vc
        else:
            return aep

    def run(self):
        best_plotter = Plotter(1)
        particle_plotter = Plotter(2)

        for particle in self.particles:
            viol_cost = particle.calc_violation(particle.pos)
            aep = self.calc_aep(particle.pos)
            particle.fitness = self.calc_fitness(aep, viol_cost)
            if (self.g_best_fitness is None or particle.fitness > self.g_best_fitness):
                self.g_best_fitness = particle.fitness
                self.g_best_pos = particle.pos
                self.g_best_aep = aep
            particle.best_pos = particle.pos
            particle.best_aep = aep
            particle.best_fitness = particle.fitness

        self.log_data()
        for particle in self.particles:
            particle_plotter.plot(particle.pos)
        best_plotter.plot(self.g_best_pos)

        self.iter_count = 1

        while (self.iter_count < self.max_iterations):
            self.iterate()
            self.log_data()
            self.iter_count = self.iter_count + 1

            for particle in self.particles:
                particle_plotter.plot(particle.pos)
            best_plotter.plot(self.g_best_pos) 
        
        self.log_data()
        self.log_file('results/' + str(round(self.g_best_aep,4)) + "_" + str(round(self.g_best_fitness, 4)) + "_" + str(self.num_particles) + "_" + str(self.max_iterations) + '.csv')

    def log_data(self):
        print("Iteration: ", self.iter_count, "Best Fitness: ", self.g_best_fitness, "Best AEP: ", self.g_best_aep)

