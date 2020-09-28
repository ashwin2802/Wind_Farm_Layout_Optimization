import numpy as np 

def GeneratePointsInCircle(n:int, center_x:float, center_y:float, radius:float):
	assert n>0 and isinstance(n, int), "'n' should be a positive integer."

	radius_arr = radius  * np.sqrt(np.random.rand(n))
	theta_arr  = 2*np.pi * np.random.rand(n)

	x_arr = center_x + (radius_arr * np.cos(theta_arr))
	y_arr = center_y + (radius_arr * np.sin(theta_arr))

	return np.squeeze(np.dstack((x_arr,y_arr)))