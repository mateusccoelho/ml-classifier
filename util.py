import os
import numpy as np
import skimage.io
import csv

classes = {0:"Motorola-Nexus-6", 1:"HTC-1-M7", 2:"Motorola-Droid-Maxx",
3:"iPhone-6", 4:"LG-Nexus-5x", 5:"Samsung-Galaxy-S4", 6:"Samsung-Galaxy-Note3",
7:"iPhone-4s", 8:"Motorola-X", 9:"Sony-NEX-7"}

def export_result(results, arqs):
	fcsv = open('result.csv', 'w')
	writer = csv.writer(fcsv)
	writer.writerow(['fname', 'camera'])
	for i in range(len(results)):
		writer.writerow([arqs[i], classes.get(results[i])])
	fcsv.close()

def cropImage(img):

	largura = len(img)
	altura = len(img[0])

	new_image = img[largura/2 - 256: largura/2 + 256, altura/2 - 256: altura/2 + 256]

	print (new_image.shape)

	return new_image

def crop_all():
	for name in os.walk('train').__next__()[1]:
		for file in os.listdir('train/' + name):
			skimage.io.imsave('train/' + name + '/' + file.split('.')[0] + '_red.png', cropImage(skimage.io.imread('train/' + name + '/' + file)))



def showImage(img):
	skimage.io.imshow(img)
	skimage.io.show()
