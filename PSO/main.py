import pandas as pd
import numpy as np

import pso

if __name__ == "__main__":
    swarm = pso.Swarm(num_particles= 100, max_iterations=100)
    swarm.run()
    