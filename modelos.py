import numpy as np
import sklearn.linear_model


def applyLogReg(X, Y, regularization):
	logreg = sklearn.linear_model.LogisticRegression(C=1/regularization)
	logreg.fit(X, Y)
	return logreg

def applyNeuralNetwork(X, Y, regularization, numberOfHiddenUnits):
	clf = MLPClassifier(solver='lbfgs', alpha=regularization, hidden_layer_sizes=(numberOfHiddenUnits), random_state=1)
	clf.fit(X, Y)
	return clf

def getAcuracy(model, X, response):
	predicted = model.predict(X)
	correct = 0
	for index in range(len(predicted)):
		if predicted[index] == response[index]:
			correct+=1

	return correct/float(len(predicted)) * 100
