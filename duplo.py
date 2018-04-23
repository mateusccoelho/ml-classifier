import os
import re
import skimage.io
import skimage.restoration
from skimage import img_as_ubyte
import numpy as np
import lbp
import skimage.feature
import skimage.exposure
import util
import spn


def features_duplo(fp):
	X = []
	Y = []
	i = 0
	regex = re.compile(r'.*_red\.png')
	keys = fp.keys()
	print(keys)
	for folder in os.walk('train').__next__()[1]:
		print(folder)
		arqs = list(filter(regex.search, os.listdir('train/' + folder)))
		lines = []
		for arq in arqs:
			img = skimage.io.imread('train/' + folder + '/' + arq)
			denoised = skimage.restoration.denoise_wavelet(img, sigma=0.019, \
						wavelet='db4', wavelet_levels=4, multichannel=True)
			ruido = img - img_as_ubyte(denoised)

			coefs = []
			for camera in keys:
				for banda in range(3):
					coefs.append(ncc(fp[camera][:,:,banda], ruido[:,:,banda]))
			np.array(coefs)
			lines.append()
		X.append(np.vstack(lines))
		Y.append(np.ones((275)) * i)
		i += 1
	return np.vstack(X), np.hstack(Y)

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
