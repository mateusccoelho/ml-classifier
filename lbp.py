import os
import re
import sys
import skimage.io
import skimage.feature
import skimage.exposure
import skimage.external.tifffile
import numpy as np
import util

def generate_csv():
	X, Y = lbp.prepare_data_lbp()
	print(X.shape)
	print(Y.shape)
	np.savetxt("X.csv", X, delimiter=",")
	np.savetxt("Y.csv", Y, delimiter=",")

def prepare_test_lbp():
	arqs = os.listdir('test')
	lines = []
	for arq in arqs:
		hists = lbp(skimage.io.imread('test/' + arq, plugin='matplotlib'), 8, 1.0)
		lines.append(np.hstack(hists))
	np.savetxt("X_test.csv", np.vstack(lines), delimiter=",")
	return arqs

def prepare_data_lbp():
	X = []
	Y = []
	i = 0
	regex = re.compile(r'.*_red\.png')
	for folder in os.walk('train').__next__()[1]:
		print(folder)
		arqs = list(filter(regex.search, os.listdir('train/' + folder)))
		lines = []
		for arq in arqs:
			hists = lbp(skimage.io.imread('train/' + folder + '/' + arq), 8, 1.0)
			lines.append(np.hstack(hists))
		X.append(np.vstack(lines))
		Y.append(np.ones((275)) * i)
		i += 1
	return np.vstack(X), np.hstack(Y)

def lbp(img, pontos, raio):
	hists = []
	for i in range(img.shape[2]):
		lbp = skimage.feature.local_binary_pattern(img[:,:,i],pontos, raio)
		hist = skimage.exposure.histogram(lbp, 2**pontos)[0]
		hists.append(hist)
	return hists
