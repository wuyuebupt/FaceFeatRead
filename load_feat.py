import os,sys

import numpy as np
from numpy import linalg as LA
# import scipy
# import cPickle as cp 




if __name__ =='__main__':
	# build knn
	lines = open (sys.argv[1])
	# 
	for line in lines:
		arr = line.strip().split()
		featpath = arr[0]
		feat_ = np.fromfile(featpath, dtype=np.dtype('f4'))
		feat_ = feat_ / LA.norm(feat_)
		# print (feat_)
		print (feat_.shape)
		assert (feat_.shape[0]== 512)
	
