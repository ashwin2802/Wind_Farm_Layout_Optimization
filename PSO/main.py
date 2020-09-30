import pandas as pd
import numpy as np

import pso

if __name__ == "__main__":
    swarm = pso.Swarm(num_particles= 10, max_iterations=20)
    swarm.run()
    