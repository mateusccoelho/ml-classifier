import os
import re
import skimage.io
import skimage.restoration
from skimage import img_as_ubyte
import numpy as np
import lbp

def features_test(fp):
	regex = re.compile(r'.*_red\.png')
	keys = ['iPhone-6', 'Motorola-Droid-Maxx', 'Motorola-X', 'Motorola-Nexus-6',\
	'iPhone-4s', 'Samsung-Galaxy-S4', 'HTC-1-M7', 'Sony-NEX-7', 'Samsung-Galaxy-Note3', 'LG-Nexus-5x']
	print(keys)

	arqs = os.listdir('test')
	lines = []
	for arq in arqs:
		img = skimage.io.imread('test/' + arq, plugin='matplotlib')
		denoised = skimage.restoration.denoise_wavelet(img, sigma=0.019, \
					wavelet='db4', wavelet_levels=4, multichannel=True)
		ruido = img - img_as_ubyte(denoised)

		coefs = []
		for camera in keys:
			for banda in range(3):
				coefs.append(ncc(fp[camera][:,:,banda], ruido[:,:,banda]))

		lines.append(np.array(coefs))
	np.savetxt("X_coef_test.csv", np.vstack(lines), delimiter=",")
	return arqs

def features(fp):
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

			lines.append(np.array(coefs))
		X.append(np.vstack(lines))
		Y.append(np.ones((275)) * i)
		i += 1
	return np.vstack(X), np.hstack(Y)

def ncc(fp, r):
	fpm = fp.mean()
	fpd = fp.std()
	rm = r.mean()
	rd = r.std()
	return (1/(512*512*fpd*rd)) * ((fp - fpm) * (r - rm)).sum()

def lbp_spn_test():
	arqs = os.listdir('test')
	lines = []
	for arq in arqs:
		img = skimage.io.imread('test/' + arq, plugin='matplotlib')
		denoised = skimage.restoration.denoise_wavelet(img, sigma=0.019, \
					wavelet='db4', wavelet_levels=4, multichannel=True)
		ruido = img - img_as_ubyte(denoised)
		hists = lbp.lbp(ruido, 8, 1.0)
		lines.append(np.hstack(hists))
	np.savetxt("X_ruido_test.csv", np.vstack(lines), delimiter=",")
	return arqs

def lbp_spn_train():
	X = []
	Y = []
	i = 0
	regex = re.compile(r'.*_red\.png')
	for folder in os.walk('train').__next__()[1]:
		print(folder)
		arqs = list(filter(regex.search, os.listdir('train/' + folder)))
		lines = []
		for arq in arqs:
			img = skimage.io.imread('train/' + folder + '/' + arq)
			denoised = skimage.restoration.denoise_wavelet(img, sigma=0.019, \
						wavelet='db4', wavelet_levels=4, multichannel=True)
			ruido = img - img_as_ubyte(denoised)
			hists = lbp.lbp(ruido, 8, 1.0)
			lines.append(np.hstack(hists))
		X.append(np.vstack(lines))
		Y.append(np.ones((275)) * i)
		i += 1
	return np.vstack(X), np.hstack(Y)


def get_fingerprint():
	fp = {"Motorola-Nexus-6":None, "HTC-1-M7":None, "Motorola-Droid-Maxx":None,
	"iPhone-6":None, "LG-Nexus-5x":None, "Samsung-Galaxy-S4":None,
	"Samsung-Galaxy-Note3":None, "iPhone-4s":None, "Motorola-X":None,
	"Sony-NEX-7":None}

	regex = re.compile(r'.*_red\.png')
	for folder in os.walk('train').__next__()[1]:
		print(folder)
		arqs = list(filter(regex.search, os.listdir('train/' + folder)))[:50]
		sum = np.zeros((512,512,3), dtype='int16')
		for arq in arqs:
			img = skimage.io.imread('train/' + folder + '/' + arq)
			denoised = skimage.restoration.denoise_wavelet(img, sigma=0.019, \
						wavelet='db4', wavelet_levels=4, multichannel=True)
			ruido = img - img_as_ubyte(denoised)
			sum += ruido
		fp[folder] = (sum/50).astype('uint8')

	return fp
