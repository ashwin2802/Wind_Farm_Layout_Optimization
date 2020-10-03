import numpy as np
from eval import modAEP, loadPowerCurve, binWindResourceData
import concurrent.futures

power_curve = loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
wind_bins = binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
       
def step():
    pts = [1] * 50 + [0] * 350
    np.random.shuffle(pts)
    conf = np.array(pts).reshape([20,20])
    pos = []
    for i in range(len(conf)):
        for j in range(len(conf[i])):
            if (int(conf[i][j]) == 1):
                pos.append([(i + 0.5) * 200, (j + 0.5) * 200])
    aep = modAEP(np.array(pos), power_curve, wind_bins)
    viol = 0
    for i in range(len(pos)):
        for j in range(len(pos)):
            if (i == j):
                continue
            dist = np.sqrt((pos[i][0] - pos[j][0])** 2 + (pos[i][1] - pos[j][1])** 2)
            if (dist < 400):
                viol = viol + (400 - dist)
    # print(viol)
    return conf, aep ** 2 if (viol == 0) else -viol, aep
    
def writeToFile(filename, conf):
    pos = []
    for i in range(len(conf)):
        for j in range(len(conf[i])):
            if (int(conf[i][j]) == 1):
                pos.append([(i + 0.5) * 200, (j + 0.5) * 200])

    f = open(filename, "w")
    f.write("x,y\n")
    for i in range(len(pos)):
        f.write(str(pos[i][0]) + "," + str(pos[i][1]) + "\n")
    f.close()

def iterate(count, num_batch):
    best_conf = None
    best_fit = -np.inf
    best_aep = 0
    chkpt = int(np.sqrt(count))
    for i in range(count):
        conf, fit, aep = step()
        if (fit > best_fit):
            best_conf = conf
            best_aep = aep
            best_fit = fit
        if (i % (chkpt) == 0):
            print("Batch: ", num_batch, ", Samples: ", i, ", Best Fitness: ", best_fit, ", Best AEP: ", best_aep)

    return best_conf, best_fit, best_aep

if __name__ == "__main__":
    workers = 4
    iterations = int(input())
    best_conf = None
    best_fit = -np.inf
    best_aep = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        batch_size = max(32, iterations // workers)
        futures = {executor.submit(iterate, batch_size, i/batch_size): i for i in range(0, iterations, batch_size)}

        batches = 0
        for future in concurrent.futures.as_completed(futures):
            conf, fit, aep = future.result()
            if (fit > best_fit):
                best_conf = conf
                best_aep = aep
                best_fit = fit
            
            print("Batch: ", batches, ", Best Fitness: ", best_fit, ", Best AEP: ", best_aep)
            batches = batches + 1
    
        print("Best AEP: ", best_aep)
    
    writeToFile("results/random_search/" + str(round(best_aep, 4)) + "_" + str(round(best_fit, 4)) + "_" + str(iterations) + ".csv", best_conf)    
        