import pandas as pd
import numpy as np

import bound_pso as pso

if __name__ == "__main__":
    num_particles, max_iterations = list(map(int, input().split(',')))
    swarm = pso.Swarm(num_particles=num_particles, max_iterations=max_iterations)
    swarm.run()
    