"""
Source code for CURIE: A method for protecting SVM Classifier from Poisoning Attack
http://rickylaishram.com/research/curie16.html

@author: Ricky Laishram (rlaishra@syr.edu)
"""

import random
import csv
import data
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats.mstats import zscore
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class FilterAttack(object):
	"""
	Filter out the attack points
	"""
	def __init__(self, source_file, dest_file):
		super(FilterAttack, self).__init__()
		self._sfile = source_file
		self._dfile = dest_file
		self._label_weight = 5000
		self._threshold = 1.645

	def __preprocessing(self):
		self._data = []
		self._labels = []

		_dat = self.__read()

		for x in _dat:
			self._data.append(x[:-1])
			self._labels.append(x[-1])

		self._data = np.array(self._data)
		self._labels = np.array(self._labels)

		self._data_pca = preprocessing.MinMaxScaler().fit_transform(self._data)
		pca = PCA(n_components=0.95)
		pca.fit(self._data)

		self._data_pca = pca.transform(self._data_pca)

	def __cluster(self):
		self._cluster_member = {}

		cls = DBSCAN(eps=5, min_samples=5, metric='euclidean')
		self._pred_cluster = cls.fit_predict(self._data_pca)

		for i,x in enumerate(self._pred_cluster):
			if x in self._cluster_member:
				self._cluster_member[x].append(i)
			else:
				self._cluster_member[x] = [i]
	
	def __distance(self):
		"""
		Compute the average euclidean distance from members of the cluster
		taking the weighted label into account
		"""
		self._data_distance = []
		for x in range(0, len(self._data)):
			cluster = self._pred_cluster[x]
			n = len(self._cluster_member[cluster])
			population = self._cluster_member[cluster]

			if n > 10:
				population = random.sample(self._cluster_member[cluster], 10)
				n = 10

			dist = 0

			for y in self._cluster_member[cluster]:
				vx = np.array(self._data[x] + [self._labels[x] * self._label_weight])
				vy = np.array(self._data[y] + [self._labels[y] * self._label_weight])
				d = euclidean(vx, vy)
				dist += d*d
			self._data_distance.append(dist/n)
		
		self._data_distance = zscore(self._data_distance)

	def __filter(self):
		"""
		Remove data with distance score above the threshold
		"""
		self._cleaned_data = []

		for i, x in enumerate(self._data_distance):
			if x < self._threshold:
				self._cleaned_data.append(np.append(self._data[i], self._labels[i]))

		print('Original Data Points: {}'.format(len(self._data_distance)))
		print('Cleaned Data Points: {}'.format(len(self._cleaned_data)))


	def __intra_cluster_distance(self):
		distances = {}
		icd = []
		#sample = random.sample(range(0,len(self._data)),100)

		for x in range(0, len(self._data)):
			cluster = self._pred_cluster[x]
			n = len(self._cluster_member[cluster])
			population = self._cluster_member[cluster]

			if n > 10:
				population = random.sample(self._cluster_member[cluster], 10)
				n = 10

			dist = 0

			for y in population:
				vx = np.array(self._data[x])
				vy = np.array(self._data[y])

				ed = euclidean(vx, vy)
				dist += ed * ed

			dist = dist/len(population)
			
			if cluster not in distances.keys():
				distances[cluster] = []

			distances[cluster].append(dist)

		for x in distances.keys():
			icd.append(sum(distances[x])/len(distances[x]))

		theta = 1.645
		p = 0.1
		self._label_weight = np.sqrt((theta - 1)*max(icd)/(1 - p))


	def __save_cleaned(self):
		self.__save(self._cleaned_data)


	def __read(self):
		content = []
		with open(self._sfile, 'r') as f:
			reader = csv.reader(f, delimiter=' ', quotechar='|')
			for row in reader:
				content.append(list(map(float,row)))
		return content

	def __save(self, sdata):
		with open(self._dfile, 'w') as f:
			writer = csv.writer(f, delimiter=' ',quotechar='|', \
			quoting=csv.QUOTE_MINIMAL)
			for row in sdata:
				writer.writerow(row)


	def run(self):
		print('Preprocessing')
		self.__preprocessing()
		print('Clustering')
		self.__cluster()
		print('Intra cluster distances')
		self.__intra_cluster_distance()
		print('Distance')
		self.__distance()
		print('Filter')
		self.__filter()
		print('Saving')
		self.__save_cleaned()

		


		