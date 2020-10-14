import numpy as np
import sklearn
# import sklearn.neighbors.KDTree
from sklearn.neighbors import KDTree


class Novelty:
	def __init__(self, k):
		self.k = k
		self.archive_list = []
		self.kd_tree = None

	def calcNoveltyPoint(self, point):
		dist, ind = self.kd_tree.query(np.array([point]), k=self.k)
		return np.mean(dist)

	def calcNovelty(self, points_list, rebuild = True, offset = 0):
		if rebuild:
			self.kd_tree = KDTree(np.array(self.archive_list), leaf_size=5)

		novelties = []
		for i in range(offset, len(points_list)):
			novelties.append(self.calcNoveltyPoint(points_list[i]))

		return np.max(novelties)

	def addPointToArchive(self, point):
		return self.archive_list.append(point)


	def addListToArchive(self, points_list, offset=0):
		for i in range(offset, len(points_list)):
			self.addPointToArchive(points_list[i])
