import numpy as np
from eval import modAEP, loadPowerCurve, binWindResourceData
from utils import Plotter

max_viol_time = 10
max_life = 80
max_g_time = 50
reset_ratio = 0.5

class Particle:
    def __init__(self, num_turbines: int = 50):
        self.min_sep = 400
        self.length = 4000
        self.fence = 50
        self.num_turbines = num_turbines

        self.cpts = np.random.choice(range(1,5))
        self.epts = np.random.choice(range(4,9))
        self.fpts = num_turbines - self.cpts - self.epts*4
        self.shape = [num_turbines, 2]

        self.max_viol_time = max_viol_time
        self.max_life = max_life

        self.corners = [[self.fence, self.fence], [self.fence, self.length - self.fence], [self.length - self.fence, self.fence], [self.length - self.fence, self.length - self.fence]]
        
        self.best_pos = np.zeros(self.shape)
        self.best_aep = 0
        self.best_fitness = -np.inf

        self.viol = 0
        self.viol_time = 0
        self.life = 0

        idx = np.random.choice(4, size=min(self.cpts, 4), replace=False)
        self.pos = []
        for i in idx:
            self.pos.append(self.corners[i])
        
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([e, self.fence])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([self.fence, e])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([e, self.length - self.fence])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([self.length - self.fence, e])

        for i in range(self.fpts):
            self.pos.append([self.fence+np.random.uniform()*(self.length - 2*(self.fence)),self.fence+np.random.uniform()*(self.length - 2*(self.fence))])

        self.pos = np.array(self.pos)
        self.constrain(self.pos)
        self.viol_time = 0

    def reset(self):
        self.best_pos = np.zeros(self.shape)
        self.best_aep = 0
        self.best_fitness = -np.inf

        self.viol = 0
        self.viol_time = 0
        self.life = 0

        self.cpts = np.random.choice(range(1,5))
        self.epts = np.random.choice(range(4,9))
        self.fpts = self.num_turbines - self.cpts - self.epts * 4
        
        idx = np.random.choice(4, size=min(self.cpts, 4), replace=False)
        self.pos = []
        for i in idx:
            self.pos.append(self.corners[i])
        
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([e, self.fence])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([self.fence, e])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([e, self.length - self.fence])
        ept = np.ones(self.epts) * self.fence + np.random.uniform(size=self.epts) * (self.length - 2 * self.fence)
        for e in ept:
            self.pos.append([self.length - self.fence, e])

        for i in range(self.fpts):
            self.pos.append([self.fence + np.random.uniform() * (self.length - 2 * (self.fence)), self.fence + np.random.uniform() * (self.length - 2 * (self.fence))])
            
        self.pos = np.array(self.pos)
        self.constrain(self.pos)
        self.viol_time = 0

    def move(self, g_best):
        if (g_best is None):
            return self.pos
        
        if(self.viol > 0):
            return self.constrain(self.pos)

        # vel_1 = []
        # vel_2 = []

        # for c in range(self.cpts):
        #     vel_1.append([0, 0])
        #     vel_2.append([0, 0])
        
        # for e in range(self.epts):
        #     vel_1.append([np.random.uniform(low=0, high=1), 0])
        #     vel_2.append([np.random.uniform(low=-1, high=1), 0])
        # for e in range(self.epts):
        #     vel_1.append([0, np.random.uniform(low=0, high=1)])
        #     vel_2.append([0, np.random.uniform(low=-1, high=1)])
        # for e in range(self.epts):
        #     vel_1.append([np.random.uniform(low=0, high=1), 0])
        #     vel_2.append([np.random.uniform(low=-1, high=1), 0])
        # for e in range(self.epts):
        #     vel_1.append([0, np.random.uniform(low=0, high=1)])
        #     vel_2.append([0, np.random.uniform(low=-1, high=1)])

        # for f in range(self.fpts):
        #     vel_1.append([np.random.uniform(low=0, high=1),np.random.uniform(low=0, high=1)])
        #     vel_2.append([np.random.uniform(low=-1, high=1),np.random.uniform(low=-1, high=1)])

        # vel_1 = np.array(vel_1)
        # vel_2 = np.array(vel_2)

        # global_move = np.multiply(vel_1, (g_best - self.pos))
        # local_move = np.multiply(vel_2, (self.best_pos - self.pos))

        # return self.constrain(self.pos + local_move)
        
        vel_1 = np.random.uniform(low=0, high=1, size=self.shape)
        vel_2 = np.random.uniform(low=-1, high=1, size=self.shape)
        
        global_move = np.multiply(vel_1, (g_best - self.pos))
        local_move = np.multiply(vel_2, (self.best_pos - self.pos))

        return self.constrain(self.pos + global_move + local_move)

    def constrain(self, pos):
        for i in range(len(pos)):
            for j in range(len(pos)):
                if (i == j):
                    continue

                x_1, y_1 = pos[i]
                x_2, y_2 = pos[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                if (dist < self.min_sep):
                    radius = max((self.min_sep - dist), 100)
                    angle = np.arctan((y_2 - y_1) / (x_2 - x_1 + 1e-6))
                    # turn = np.random.uniform(low = -1.57, high = 1.57)
                    turn = 0
                    c=(x_2 * y_1 - x_1 * y_2) / (x_2 - x_1 + 1e-6)
                    
                    s_x_1 = x_1
                    s_y_1 = y_1 + c
                    s_x_2 = x_2
                    s_y_2 = y_2 + c

                    r_x_1 = s_x_1 * np.cos(angle) + s_y_1 * np.sin(angle) + radius*np.cos(turn)
                    r_y_1 = s_y_1 * np.cos(angle) - s_x_1 * np.sin(angle) + radius*np.sin(turn)
                    r_x_2 = s_x_2 * np.cos(angle) + s_y_2 * np.sin(angle) - radius*np.cos(turn)
                    r_y_2 = s_y_2 * np.cos(angle) - s_x_2 * np.sin(angle) - radius * np.sin(turn)
                    
                    pos[i] = [r_x_1 * np.cos(angle) - r_y_1 * np.sin(angle), r_x_1 * np.sin(angle) + r_y_1 * np.cos(angle) - c]
                    pos[j] = [r_x_2 * np.cos(angle) - r_y_2 * np.sin(angle), r_x_2 * np.sin(angle) + r_y_2 * np.cos(angle) - c]

        for i in range(len(pos)):
            pos[i][0] = min(pos[i][0], self.length - self.fence)
            pos[i][1] = min(pos[i][1], self.length - self.fence)
            pos[i][0] = max(self.fence, pos[i][0])
            pos[i][1] = max(self.fence, pos[i][1])

        # for i in range(self.cpts + 4 * self.epts, len(pos)):
        #     pos[i][0] = min(pos[i][0], self.length - self.fence-self.min_sep)
        #     pos[i][1] = min(pos[i][1], self.length - self.fence-self.min_sep)
        #     pos[i][0] = max(self.fence+self.min_sep, pos[i][0])
        #     pos[i][1] = max(self.fence+self.min_sep, pos[i][1])
        
        cost = self.calc_viol(pos)
        
        if (cost > self.viol):
            self.viol_time = self.viol_time + 1
            if (self.viol_time > self.max_viol_time):
                self.reset()
                print("viol reset!")
                return self.pos
        elif (cost == 0):
            self.viol_time = 0

        self.viol = cost
        return pos

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

class Swarm:
    def __init__(self, num_particles: int = 500, max_iterations: int = 400):
        self.power_curve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
        self.wind_bins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')

        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.g_best_pos = None
        self.g_best_aep = 0
        self.g_best_fitness = -np.inf

        # self.bplot = Plotter(1)
        # self.pplot = Plotter(2)

        self.g_count = 0
        self.max_g_count = max_g_time
        self.iter_count = 0
        self.particles = [Particle() for i in range(num_particles)]

    def calc_aep(self, pos, power_curve, wind_bins):
        return modAEP(pos, power_curve, wind_bins)

    def calc_fitness(self, new_aep, viol_cost):
        if (viol_cost > 0):
            return - viol_cost
        else:
            return new_aep

    def move_swarm(self):
        prev_g_fit = self.g_best_fitness

        for i in range(self.num_particles):
            new_pos = self.particles[i].move(self.g_best_pos)
            new_viol = self.particles[i].viol
            new_aep = self.calc_aep(new_pos, self.power_curve, self.wind_bins)
            new_fitness = self.calc_fitness(new_aep, new_viol)

            if (new_fitness > self.particles[i].best_fitness):
                self.particles[i].best_fitness = new_fitness
                self.particles[i].best_pos = new_pos
                self.particles[i].best_aep = new_aep

            if (new_fitness > self.g_best_fitness):
                self.g_best_fitness = new_fitness
                self.g_best_aep = new_aep
                self.g_best_pos = new_pos

            self.particles[i].fitness = new_fitness
            self.particles[i].pos = new_pos

            if (self.particles[i].viol == 0 and new_aep < self.particles[i].best_aep):
                self.particles[i].life = self.particles[i].life + 1
                if (self.particles[i].life > self.particles[i].max_life):
                    self.particles[i].reset()
                    print("life reset!")
            
        if (self.g_best_fitness == prev_g_fit):
            self.g_count = self.g_count + 1
            if (self.g_count >= self.max_g_count):
                self.g_count = 0
                print("elite reset!")
                self.reset_worst(reset_ratio)
        else:
            self.g_count = 0

    def reset_worst(self, ratio):
        k = int(self.num_particles * ratio)
        bests = np.array([particle.best_fitness for particle in self.particles])
        idx = np.argpartition(bests, k)
        for i in range(k):
            self.particles[idx[i]].reset()

    def log_file(self, filename: str):
        f = open(filename, "w")
        f.write("x,y\n")
        for i in range(len(self.g_best_pos)):
            f.write(str(self.g_best_pos[i][0]) + "," + str(self.g_best_pos[i][1]) + "\n")
        f.close()

    def log_data(self):
        print("Iteration: ", self.iter_count, "Best Fitness: ", self.g_best_fitness, "Best AEP: ", self.g_best_aep)

    # def draw_plots(self):
    #     self.bplot.plot(self.g_best_pos)
    #     for particle in self.particles:
    #         self.pplot.plot(particle.best_pos)

    def run(self):
        self.iter_count = 0
        while (self.iter_count < self.max_iterations):
            self.move_swarm()
            self.log_data()
            # self.draw_plots()
            self.iter_count = self.iter_count + 1
        
        self.log_data()
        self.log_file('results/bound/'+ str(round(self.g_best_aep,4)) + "_" + str(round(self.g_best_fitness, 4)) + "_" + str(self.num_particles) + "_" + str(self.max_iterations) + '.csv')