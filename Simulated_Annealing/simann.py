import numpy as np
from eval import modAEP, binWindResourceData, loadPowerCurve
from utils import Plotter, log_file

class Annealer:
    def __init__(self, num_turbines: int = 50, max_iterations: int = 5000):
        self.fence = 50
        self.length = 4000
        self.min_sep = 400
       
        self.temp = 600 ** 2
        self.cooldown = 0.98
        self.cycles = 20
        self.step = self.min_sep
        
        self.dims = num_turbines * 2
        self.pop_size = self.dims
        self.max_iterations = max_iterations
        self.iterations = max(100, 5 * self.dims)
        self.shape = [num_turbines, 2]
        self.pos = np.ones(self.shape) * self.shape + np.random.uniform(size=self.shape)*(self.length - 2*self.fence)
        
        self.aep = 0
        self.viol = 0
        self.fitness = np.inf
        self.best_aep = 0
        self.iter_count = 0
        self.best_pos = self.pos
        self.best_fitness = np.inf
        
        self.power_curve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
        self.wind_bins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
        # self.bplot = Plotter(1)
        # self.pplot = Plotter(2)

    def run(self):
        self.aep = self.calc_aep(self.pos)
        self.viol = self.calc_viol(self.pos)
        self.fitness = self.calc_fitness(self.aep, self.viol)

        while (self.iter_count < self.max_iterations):
            for k in range(self.iterations):
                ni = 0
                for j in range(self.cycles):
                    for i in range(self.dims):
                        R = np.random.uniform()
                        new_pos = self.pos
                        new_pos[i // 2][i % 2] = new_pos[i // 2][i % 2] + R * self.step
                        new_pos[i // 2][i % 2] = max(new_pos[i // 2][i % 2], self.fence)
                        new_pos[i // 2][i % 2] = min(new_pos[i // 2][i % 2], self.length - self.fence)
                        new_aep = self.calc_aep(new_pos)
                        new_viol = self.calc_viol(new_pos)
                        new_fitness = self.calc_fitness(new_aep, new_viol)
                        
                        if (self.viol > 0 and new_viol > 0 and new_viol < self.viol):
                            self.update(new_pos, new_fitness, new_aep, new_viol)
                            ni = ni + 1
                        elif (self.viol == 0 and new_viol == 0 and new_fitness < self.fitness):
                            self.update(new_pos, new_fitness, new_aep, new_viol)
                            ni = ni + 1
                        elif (self.viol > 0 and new_viol == 0):
                            self.update(new_pos, new_fitness, new_aep, new_viol)
                            ni = ni + 1
                        else:
                            P = np.exp(np.floor(self.fitness - new_fitness) / self.temp)
                            R = np.random.uniform()
                            if (P < R):
                                self.update(new_pos, new_fitness, new_aep, new_viol)
                                ni = ni + 1
                r = ni / (self.cycles * self.dims)
                if (r > 0.6):
                    self.step = self.step * (1 + (2 * (r - 0.6)) / 0.4)
                else:
                    self.step = self.step * (1 + (2 * (r - 0.4)) / 0.4)
                print("Iteration: ", self.iter_count + k, "Best Fitness: ", self.best_fitness, "Best AEP: ", self.best_aep)                

            self.temp = self.temp * self.cooldown
            self.iter_count = self.iter_count + self.iterations

        log_file(self.best_pos, 'results/temp/' + str(round(self.best_aep, 4)) + "_" + str(round(-self.best_fitness, 4)) + "_" + str(self.max_iterations) + '.csv')

    def update(self, new_pos, new_fitness, new_aep, new_viol):
        self.pos = new_pos
        self.viol = new_viol
        self.aep = new_aep
        self.fitness = new_fitness
        if (self.fitness < self.best_fitness):
            self.best_aep = self.aep
            self.best_pos = self.pos
            self.best_fitness = self.fitness
        # self.pplot.plot(self.pos)
        # self.bplot.plot(self.best_pos)     

    def calc_fitness(self, aep, viol):
        if (viol > 0):
            return viol
        else:
            return -aep**2

    def calc_aep(self, pos):
        return modAEP(pos, self.power_curve, self.wind_bins)    

    def calc_viol(self, pos):
        cost = 0
        for i in range(len(pos)):
            x_1, y_1 = pos[i]
            if (x_1 < self.fence):
                cost = cost + (self.fence - x_1)
            if (y_1 < self.fence):
                cost = cost + (self.fence - y_1)
            if (x_1 > self.length - self.fence):
                cost = cost + (self.length - self.fence - x_1)
            if (y_1 > self.length - self.fence):
                cost = cost + (self.length - self.fence - y_1)

            for j in range(len(pos)):
                if (i == j):
                    continue
                x_2, y_2 = pos[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                if (dist < self.min_sep):
                    cost = cost + (self.min_sep - dist)
        return cost