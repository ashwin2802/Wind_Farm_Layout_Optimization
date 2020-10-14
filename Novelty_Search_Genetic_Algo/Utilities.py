import numpy as np 


def GeneratePointsInCircle(n:int, center_x:float, center_y:float, radius:float):
	assert n>0 and isinstance(n, int), "'n' should be a positive integer."

	radius_arr = radius  * np.sqrt(np.random.rand(n))
	theta_arr  = 2*np.pi * np.random.rand(n)

	x_arr = center_x + (radius_arr * np.cos(theta_arr))
	y_arr = center_y + (radius_arr * np.sin(theta_arr))

	return np.squeeze(np.dstack((x_arr,y_arr)))


def PlacePointsInCorner(x_min, x_max, y_min, y_max, count):
	# assert count == 4 or count == 36 or count == 37, "Allowed values of count are 4,36,37"

	# place 4 points in 4 corners
	# res = [[x_max, y_min], [x_min, y_max], [x_max, y_max]]
	res = [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]

	# place more points on boundary
	if(count > 4):
		points = 8
		theta_arr = np.linspace(0, 1, points+2)[1:-1]
		for theta in theta_arr:
			# res.append([(theta*x_min + (1-theta)*x_max), y_min])
			res.append([x_min, (theta*y_min + (1-theta)*y_max)])
			

		points = 8
		theta_arr = np.linspace(0, 1, points+2)[1:-1]
		for theta in theta_arr:
			res.append([(theta*x_min + (1-theta)*x_max), y_max])
			res.append([x_max, (theta*y_min + (1-theta)*y_max)])

	# place point in center
	if(count > 36):
		res.append([(x_min+x_max)/2, (y_min+y_max)/2])
		
	return res
