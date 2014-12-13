#!/usr/bin/python   

'''
Basic perceptron network.

Code inspired by implementation provided in "Machine Learning: An Algorithmic Perspective"
(1st Edition), by Stephen Marsland (http://stephenmonika.net/).

The overall structure is the same but is not verbatim.
'''

import numpy as np

class perceptron:

	# Constructor #
	def __init__(self,inputs,targets):
		# setup network size:
		if np.ndim(inputs) > 1:
			self.nIn = np.shape(inputs)[1]
		else:
                        self.nIn = 1
                        
                if np.ndim(targets) > 1:
                        self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(inputs)[0]

		# initialize network weights:
		self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

	# Train the network #
	def train(self,inputs,targets,eta,nIter):
		# add a bias node to the inputs:
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		# train:
		change = range(self.nData)

		for n in range(nIter):
			self.outputs = self.fwd(inputs)
			self.weights += eta*np.dot(np.transpose(inputs),targets-self.outputs)
			# randomize order of inputs:
			np.random.shuffle(change)
			inputs = inputs[change,:]
			targets = targets[change,:]

	# Run the network forward one iteration #
	def fwd(self,inputs):
		outputs = np.dot(inputs,self.weights)

		# apply threshold:
		return np.where(outputs>0,1,0)

	# Generate confusion matrix #
	def confmat(self,inputs,targets):
		# add a bias node to the inputs:
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		outputs = np.dot(inputs,self.weights)
		nClass = np.shape(targets)[1]

		if nClass==1:
			nClass = 2
			outputs = np.where(outputs>0,1,0)
		else:
			outputs = argmax(outputs,1)
			targets = argmax(targets,1)

		cm = np.zeros((nClass,nClass))
		for i in range(nClass):
			for j in range(nClass):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print cm
		print np.trace(cm)/np.sum(cm)

# Example 1: Run perceptron network for AND logic function
def ex_AND():
	import perceptron
	
	a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]]) 
        AND_in = a[:,0:2]
        AND_out = a[:,2:]
        p = perceptron.perceptron(AND_in,AND_out)
        p.train(AND_in,AND_out,0.25,10)
        p.confmat(AND_in,AND_out)
        
# Example 2: Run perceptron network for XOR logic function
def ex_XOR():
	import perceptron
	
	a = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
        XOR_in = a[:,0:2]
        XOR_out = a[:,2:]
        p = perceptron.perceptron(XOR_in,XOR_out)
        p.train(XOR_in,XOR_out,0.25,10)
        p.confmat(XOR_in,XOR_out)

        
