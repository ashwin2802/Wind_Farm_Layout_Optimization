import numpy as np 
import csv


def GeneratePointsInCircle(n:int, center_x:float, center_y:float, radius:float):
	assert n>0 and isinstance(n, int), "'n' should be a positive integer."

	radius_arr = radius  * np.sqrt(np.random.rand(n))
	theta_arr  = 2*np.pi * np.random.rand(n)

	x_arr = center_x + (radius_arr * np.cos(theta_arr))
	y_arr = center_y + (radius_arr * np.sin(theta_arr))

	return np.squeeze(np.dstack((x_arr,y_arr)))


def readFromFile(filename):
	with open(filename, "r") as f:
		reader = csv.reader(f)
		list_data = list(reader)

	list_data = list_data[1:]
	list_data = [[float(x),float(y)] for x,y in list_data]
	return list_data


def PlacePointsInCorner(x_min, x_max, y_min, y_max, count):
	assert count == 4 or count == 36 or count == 46, "Allowed values of count are 4,36,48"

	# place 4 points in 4 corners
	res = [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]

	# place more points on boundary
	if(count > 4):
		points = 8
		theta_arr = np.linspace(0, 1, points+2)[1:-1]
		for theta in theta_arr:
			res.append([(theta*x_min + (1-theta)*x_max), y_min])
			res.append([(theta*x_min + (1-theta)*x_max), y_max])

			res.append([x_min, (theta*y_min + (1-theta)*y_max)])
			res.append([x_max, (theta*y_min + (1-theta)*y_max)])


	# if(count > 36):
	# 	# place 12 points equidistantly on a circle
	# 	n = 10
	# 	center = [(x_min + x_max)/2.0, (y_min + y_max)/2.0]
	# 	placePointsOnCircle(center, n, res)
		
	return res


def placePointsOnCircle(center, n, res):
	assert (n < 13), "Only less than 13 points allowed on circle"

	radius = 1200
	theta_gap = (2*np.pi)/n
	theta = np.random.rand()*2*np.pi
	for i in range(n):
		curr_x, curr_y = center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta)
		res.append([curr_x, curr_y])
		theta += theta_gap