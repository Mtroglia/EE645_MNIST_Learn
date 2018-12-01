'''
Implementation of neural network
11/15/2018   version 0.1
'''

import math
import random
from mnist import MNIST
from sklearn import preprocessing
import numpy as np
import MNIST_Process as MP
import pickle
from datetime import datetime
import os

class NeuralNet:

	'''
	The varaible 'Layer' is an integer array which specifies
	the number of neurals in each layer, the first layer is
	the input layer and the last is output layer.
	'''
	def __init__(self, Layer,reg):

		self._layer = len(Layer)
		self.regularization =reg
		self.nets, Father = [], []
		for i in range(self._layer):
			current_layer = []
			for j in range(Layer[i]):
				current_layer.append(self.newNeural(Father))
			self.nets.append(current_layer)
			Father = current_layer
		
		
	# Build a new neural
	def newNeural(self, Father):
		data = {}
		#data.update({'weights':[random.uniform(-0.5,0.5) for i in range(len(Father))]})
		data.update({'weights': [np.random.normal(0,1/(math.sqrt(len(self.nets[0])))) for i in range(len(Father))]})
		data.update({'derivative':[0]*len(Father)})
		data.update({'father': Father})
		data.update({'value': 0})

		return data
		
	# We use sigmoid function as activation function
	def activationSigmoid(self, x, a):
		#x = math.exp(x*a)
		if x*a < 0: #if x is a large negitive, we exceed floating point range
			return(1- 1/(1+math.exp(x*a)))
		else:
			return(1/(1+math.exp(-x*a)))


		#return x/(x+1)
	
	# 'a' is the pointor to previous layer, 'b' is the weights
	def dotProduct(self, a, b):
		l = len(a)
		if(len(b) != l):
			print("Dimension do not match for inner product.")
			return
		res = 0
		for i in range(l):
			res += a[i]['value']*b[i]
		return res
	
	# Evaluate the network with 'input' and output actual output
	def evaluate(self, input):
		n = len(input)
		if len(self.nets[0]) != n:
			print('Inpute size do not match with the nets.')
			return
		# Initialize inpute layer
		for i in range(n):
			self.nets[0][i]['value'] = input[i]
		for j in range(1,self._layer):
			current_layer = self.nets[j]
			for k in current_layer:
				k['value'] = self.activationSigmoid(self.dotProduct(k['father'], k['weights']), 1)
		# Output the actual value, no binarilization 
		output = []
		for t in self.nets[self._layer-1]:
			output.append(t['value'])
		return output
	
	# Print values in each layer nicely
	def printLayers(self):
		for i in self.nets:
			v = []
			for j in i:
				v.append(j['value'])
			print(v)
			
	# Allow to change weights manually
	def modifyWeights(self, i,j, k, newWeight):
		self.nets[i][j]['weights'][k] = newWeight
	
	''' Using back propagation to calculate derivative,
		here we consider the l_2 loss
	'''
	def backPropagation(self, trueLabel):
		# Last layer
		layer = self.nets[self._layer-1]
		for i in range(len(layer)):
			#deriv of sigmoid activation times loss (output-true_label)Output(1-output))
			const = 2*(layer[i]['value'] - trueLabel[i])*layer[i]['value']*(1-layer[i]['value'])
			for j in range(len(layer[i]['derivative'])):
				if self.regularization == 'none':
					layer[i]['derivative'][j] = const*layer[i]['father'][j]['value']
				if self.regularization == 'ridge':
					layer[i]['derivative'][j] = const*layer[i]['father'][j]['value']+0.03*layer[i]['weights'][j]

		# Internal layers
		for k in range(2, self._layer):
			index = self._layer - k
			layer = self.nets[index]
			for n in range(len(layer)):
				neural = layer[n]
				pre_neural = self.nets[index+1]
				#derivative of sigmoid funciton const
				const, temp = (1-neural['value'])*(neural['value']), 0
				for pre_n in pre_neural:
					temp += pre_n['weights'][n]*pre_n['derivative'][n]
				const *= temp
				for t in range(len(neural['derivative'])):
					neural['derivative'][t] = const*neural['father'][t]['value']
	
	# Update weigth according to derivatives
	def updateWeights(self, l_step):
		for i in self.nets:
			for j in i:
				for k in range(len(j['weights'])):
					j['weights'][k] -= l_step*j['derivative'][k]
					
	
	''' Gradient decent algorithm in one step, train[0] is the input,
		train[1] is the lable
	'''
	def gradientDecent(self, train, l_step):

		input, label = train[0], train[1]
		
		# Evaluate on current sample
		self.evaluate(input)
		
		# Compute derivative
		self.backPropagation(label)
		
		# Gradient Decent
		self.updateWeights(l_step)

	'''
	Generate the Magnitude of the weights at each layer for each neuron. 
	'''
	def getMagWeights(self):

		magWeights = []
		magWeights.append([0])  # first layer has no connecting weights, place holder of 0 is added to array.
		for i in range(1, self._layer):
			j = 1
			layersMagWeights = []
			for neuron in self.nets[i]:
				# neuron['weights']
				magWeight = vectorMagnitude(np.array(neuron['weights']))
				#print(j, ' ', magWeight)
				layersMagWeights.append(magWeight)
				# print(j)
				j += 1
			magWeights.append(layersMagWeights)
		return(magWeights)
	
	def isMatch(self, a, b):
		if len(a) != len(b):
			return False
		for i in range(len(a)):
			if a[i] != b[i]:
				return False
		return True

	# simple binarilization
	def naiveBinary(self, a):
		#print('len a ',a)
		for i in range(len(a)):
			#print(a[i])
			if a[i] > 0.5:
				a[i] = 1
			else:
				a[i] = 0
		return a

	
	# take ouput and change  the maximum value to be 1 and others to be zeros
	def max2one(self, a):
		index = 0
		for i in range(len(a)):
			if a[i] > a[index]:
				index = i
		for j in range(len(a)):
			if j != index:
				a[j] = 0
		a[index] = 1
		return a

	''' Calculate error on 'sample_l' with current weights,
		define you own function 'trim' to binarilize the output
	'''
	def errorCalculate(self, sample_l, trim):
		print('Calc error...')
		err = 0
		for i in sample_l:
			b = trim(self.evaluate(i[0]))
			#print(b,i[1],self.isMatch(b,i[1]))
			if not self.isMatch(b,i[1]):
				err += 1
		return float(err) / len(sample_l)
		
	''' Stochastic gradient decent with training sample list 
		'tain_l', the gradient decent step length 'l_step', 
		the maximum epoch in training.
	'''
	def SGD(self, train_1, l_step, n_epoch):
		for i in range(n_epoch):
			random.shuffle(train_1)
			#train = random.sample(train_l, 1)[0]
			j=0
			for train in train_1:

				if j%500==0:
					print('Running epoch ', i,j)
				j+=1
				self.gradientDecent(train, l_step)
	''''
	SGD that takes training samples, step_size, and desired error.
	'''
	def SGD_TrainThreshold(self, train_l, l_step, err):
		ac_err, epoch = 1, 0
		while (ac_err > err):
			random.shuffle(train_l)
			j = 0
			for train in train_l:
				if j % 500 == 0:
					print('Running epoch ', epoch,' Sample No. ', j)
				j += 1
				self.gradientDecent(train, l_step)
			test = random.sample(train_l, 1000)
			ac_err = self.errorCalculate(test, self.max2one)
			print("Acc Error, epoch")
			print(ac_err, epoch)
			epoch += 1



def vectorMagnitude(V):
	return(math.sqrt(sum(i**2 for i in V)))


#%% Loading the traing data from the MNIST dataset.
print('Loading training')
images_train,labels_train = MP.loadTrain()
images_test, labels_test = MP.loadTest() # Current implementation in MP script reloads the data, wasted effort. Not using test right now, fix later.

#%%normalized pixel data
images_train,images_test = np.array(images_train)/255, np.array(images_test)/255

#reshape the labels to nx1 vector before changing it to one hot matrix
labels_train,labels_test = np.transpose(np.reshape(labels_train,(-1,len(labels_train)))), np.transpose(np.reshape(labels_test,(-1,len(labels_test))))

#Change labels to a matrix where each sample has 10 output values representing the single value decimal number. R_1 --> R_10
oneHotLabels = MP.OneHotTransform(labels_train)

oneHotLabels_Test = MP.OneHotTransform((labels_test))


#%% Turn images and labels into sample frame that network expects
samples = MP.getSamples(np.array(images_train),oneHotLabels)

samples_Test = MP.getSamples(np.array(images_test),oneHotLabels_Test)
#%%Get the input/output size , initialize neural network
inputSize = len(images_train[0])
outputSize = len(oneHotLabels[0])
#a = NeuralNet([inputSize,30,outputSize],'none')
a = NeuralNet([inputSize,30,outputSize],'ridge')



#%%
epoch_number = 100
step_size = math.sqrt(1/epoch_number) #0.25 # should be sqrroot(1/epoch)ca
#a.SGD(samples, step_size, epoch_number)
a.SGD_TrainThreshold(samples, 0.05, .045)


#%% Gets the magnitude of the weights at each layer for each neuron

magWeights = a.getMagWeights()
'''
TODO Determine the magnitude of the weights
'''

#%% Evaluate NN on the training, Transform the output for each for a sample by method max2One or naiveBinary
#print(a.errorCalculate(samples, a.max2one)) # achieves the training error to be 0.0

print('Test error: ',a.errorCalculate(samples_Test,a.max2one))
print('Train error: ',a.errorCalculate(samples, a.naiveBinary)) # achieves the training error to be 0.0


fileSave='SavedModel'+os.sep+'NN_savedModel_'+str(datetime.timestamp(datetime.now())).replace('.','')+'.sav'
pickle.dump(a,open(fileSave,'wb'))

#%%
'''
#Load a pickled model
loadFileName= 'NN_saveModel_TrainErr0425_TestErr_0509.sav'
loaded_NN = pickle.load(open(loadFileName,'rb'))
print('Testing Loaded Model')
print(loaded_NN.errorCalculate(samples_Test,loaded_NN.max2one))
'''

#%%
'''
#sample code used to test NN

#a = NeuralNet([3,30,2])
sample_1 = [
[[0,0,0],[0,0]],
[[0,0,1],[0,1]],
[[0,1,0],[0,1]],
[[1,0,0],[0,1]],
[[0,1,1],[1,0]],
[[1,1,0],[1,0]],
[[1,0,1],[1,0]],
[[1,1,1],[1,1]]
]
#a.printLayers()

#a.SDG(sample_l, 1, 10000)
#b = a.evaluate([1,1,1])
#print(a.errorCalculate(sample_l, a.naiveBinary)) # achieves the training error to be 0.0
'''