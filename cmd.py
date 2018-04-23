import os
import sys
import skimage.io
import skimage.feature
import skimage.exposure
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import util
import lbp
import modelos
from numpy import genfromtxt

x_train = genfromtxt('X.csv', delimiter=',')
y_train = genfromtxt('Y.csv', delimiter=',')
x_test = genfromtxt('X_test.csv', delimiter=',')
x_train, x_cross, y_train, y_cross = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=10)
x_train.shape
y_train.shape
x_cross.shape
y_cross.shape

reg = []
for i in range(1, 11):
	reg.append(0.001 * (10**(1/2))**i)

for reg_i in reg:
	model = modelos.applyLogReg(x_train, y_train, reg_i)
	ac = modelos.getAcuracy(model, x_cross, y_cross)
	print('Reg: ' + str(reg_i) + ' Acuracia: ' + str(ac))


sklearn.metrics.confusion_matrix(y_cross, model.predict(x_cross))

units = [10, 100, 200, 300, 400, 500, 600, ]
