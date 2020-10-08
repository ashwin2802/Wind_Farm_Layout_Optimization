from matplotlib import pyplot as plt
from Eval_fitness import modAEP, loadPowerCurve, binWindResourceData
import numpy as np
import csv

def readFromFile(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        list_data = list(reader)
    list_data = list_data[1:]

    list_x, list_y = [], []
    for x,y in list_data:
        list_x.append(float(x))
        list_y.append(float(y))
    return list_x, list_y



power_curve    =  loadPowerCurve('../Dataset/Shell_Hackathon Dataset/power_curve.csv')
wind_inst_freq =  binWindResourceData(r'../Dataset/Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv') 
list_x, list_y =  readFromFile("test.csv")
index = None
counter = 0

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('click to build line segments')
ax.set_xlim([0,4000])
ax.set_ylim([0,4000])
ax.axhline(y=3950, color='r', linestyle='-')
ax.axhline(y=50, color='r', linestyle='-')
ax.axvline(x=3950, color='r', linestyle='-')
ax.axvline(x=50, color='r', linestyle='-')
scatter_plot = ax.scatter(list_x, list_y, color='b')


def plot():
    global scatter_plot

    scatter_plot.set_offsets([[list_x[i], list_y[i]] for i in range(50)])
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # for i in range(50):
    #     for j in range(50):
    #         if (i == j):
    #             continue
    #         x_1, y_1 = list_x[i], list_y[i]
    #         x_2, y_2 = list_x[j], list_y[j]
    #         dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
            
    #         if (dist < 400):
    #             start, end = [x_1, x_2], [y_1, y_2]
    #             ax.plot(start, end)


def on_mouse_press(event):
    if event.inaxes != ax: return

    eps = 60
    global index
    print(index)
    print("mouse pressed at x: ", event.xdata, " y: ", event.ydata)
    for i in range(50):
        d = ((list_x[i]-event.xdata)**2) + ((list_y[i]-event.ydata)**2)
        if(d < eps**2):
            index = i
            break
    print(index)


def on_mouse_release(event):
    global list_x, list_y 
    global index

    if event.inaxes != ax: return
    print("mouse released at x: ", event.xdata, " y: ", event.ydata)
    if(index is not None):
        list_x[index], list_y[index] = event.xdata, event.ydata
        plot()
        index = None


def on_key(event):
    global counter
    
    if(event.key == "a"):
        print("AEP: ", modAEP(np.array([[list_x[i], list_y[i]] for i in range(50)]), power_curve, wind_inst_freq))

    elif(event.key == "h"):
        f = open("test"+str(counter)+".csv", "w")
        f.write("x,y\n")

        genes = [[list_x[i], list_y[i]] for i in range(50)]
        for i in range(50):
            f.write(str(genes[i][0]) + "," + str(genes[i][1]) + "\n")
        f.close()
        counter += 1


fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('key_press_event', on_key)

plot()
plt.show()