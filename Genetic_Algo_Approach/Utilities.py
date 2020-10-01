import numpy as np 


def GeneratePointsInCircle(n:int, center_x:float, center_y:float, radius:float):
	assert n>0 and isinstance(n, int), "'n' should be a positive integer."

	radius_arr = radius  * np.sqrt(np.random.rand(n))
	theta_arr  = 2*np.pi * np.random.rand(n)

	x_arr = center_x + (radius_arr * np.cos(theta_arr))
	y_arr = center_y + (radius_arr * np.sin(theta_arr))

	return np.squeeze(np.dstack((x_arr,y_arr)))


def PlacePointsInCorner(x_min, x_max, y_min, y_max):
	# place 4 points in 4 corners
	res = [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]

	# return 4, res

	# place 4 more points in between side boundaries
	res.append([(x_min+x_max)/2, y_min])
	res.append([(x_min+x_max)/2, y_max])
	res.append([x_min, (y_min+y_max)/2])
	res.append([x_max, (y_min+y_max)/2])

	return 8, res
