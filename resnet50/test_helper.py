import csv
import numpy as np


def readData(fileName):
    """read csv data"""

    dataSet = []
    with open(fileName, "r") as csvFile:
        reader = csv.reader(csvFile)
        for i,line in enumerate(reader):
                dataSet.append([float(item) for item in line])

    return dataSet

def euclideanDistance(v1, v2):
    """get euclidean distance of 2 vectors"""

    v1, v2 = np.array(v1), np.array(v2)
    return np.linalg.norm(v1-v2)
    #return np.sqrt(np.sum(np.square(v1 - v2)))

def cosDistance(v1, v2):
    """get euclidean distance of 2 vectors"""
    v1, v2 = np.array(v1), np.array(v2)
    return (v1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

"""
    ### distance functions
	欧式距离sqrt((vector1-vector2)*((vector1-vector2).T))#linalg.norm(ector1-vector2)
	曼哈顿距离sum(abs(vector1-vector2))
	切比雪夫距离abs(vector1-vector2).max()
	余弦距离cosV12 = dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))
	cos=v1*v2/(linalg.norm(v1)*linalg.norm(v2))

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        #This is a hot function, hence some optimizations are made. 
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        # This is a hot function, hence some optimizations are made. 
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        # This is a hot function, hence some optimizations are made. 
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
"""