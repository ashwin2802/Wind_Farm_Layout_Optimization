import csv
import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, index: int = 0, delay: float = 0.0001):
        plt.ion()
        plt.show()
        self.fig = plt.figure(index)
        self.index = index
        self.delay = delay

    def plot_from_file(self, filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            list_data = list(reader)
        list_data = list_data[1:]
        list_data = [[float(x),float(y)] for x,y in list_data]
        self.plot(list_data, file=True)
    
    def plot(self, list_data, file=False):
        plt.figure(self.index)
        plt.clf()

        list_x = []
        list_y = []
        for x, y in list_data:
            list_x.append(x)
            list_y.append(y)

        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.scatter(list_x, list_y)
        plt.axhline(y=3950, color='r', linestyle='-')
        plt.axhline(y=50, color='r', linestyle='-')
        plt.axvline(x=3950, color='r', linestyle='-')
        plt.axvline(x=50, color='r', linestyle='-')

        for i in range(len(list_data)):
            for j in range(len(list_data)):
                if (i == j):
                    continue
                x_1, y_1 = list_data[i]
                x_2, y_2 = list_data[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                
                if (dist < 400):
                    start, end = [x_1, x_2], [y_1, y_2]
                    plt.plot(start, end)

        if (file):
            plt.ioff()
            plt.show()

        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(self.delay)

def log_file(pos, filename: str):
    f = open(filename, "w")
    f.write("x,y\n")
    for i in range(len(pos)):
        f.write(str(pos[i][0]) + "," + str(pos[i][1]) + "\n")
    f.close()

if __name__ == "__main__":
    aep = input("Enter csv filename to plot from results folder: ")
    plotter = Plotter()
    plotter.plot_from_file('results/'+ aep + '.csv')