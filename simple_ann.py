#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
#X=np.array(([1,2,4],[2,4,8],[4,8,16],[8,16,32]), dtype=float)
#X=np.array([[0,0,1]], dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)
#y=np.array(([1],[2],[4],[8]), dtype=float)
#y=np.array([[0]], dtype=float)

# Define useful functions    

# Activation function: ReLU
def ReLU(t):
    return np.maximum(0,t)

# Derivative of ReLU
def ReLU_derivative(p):
    temp = np.copy(p)
    temp[temp<=0]=0
    temp[temp>0]=1
    return temp

# Activation function: tanh
def tanh(t):
    return np.tanh(t)

# Derivative of tanh
def tanh_derivative(p):
    return  1-np.power(np.tanh(p), 2)


# Activation function: sigmoid
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

def activation_function(func_type, x):
    if(func_type == 0):
        return sigmoid(x)
    elif(func_type == 1):
        return ReLU(x)
    elif(func_type == 2):
        return tanh(x)

def activation_function_derivative(func_type, x):
    if(func_type == 0):
        return sigmoid_derivative(x)
    elif(func_type == 1):
        return ReLU_derivative(x)
    elif(func_type == 2):
        return tanh_derivative(x)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y, activation_function_type):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4) # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np. zeros(y.shape)
        self.activation_function_type = activation_function_type

    def feedforward(self):
        self.layer1 = activation_function(self.activation_function_type, np.dot(self.input, self.weights1))
        self.layer2 = activation_function(self.activation_function_type, np.dot(self.layer1, self.weights2))

        return self.layer2
        
    def backprop(self):

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

	#decomposition of derivative of the loss function with respect to weights2
        d_z_output_layer_d_w2 = self.layer1.T

        d_output_layer_d_z_output_layer =  activation_function_derivative(self.activation_function_type, self.output)

        d_cost_d_output_layer = 2*(self.y - self.output)

        d_weights2 = np.dot(d_z_output_layer_d_w2, d_output_layer_d_z_output_layer * d_cost_d_output_layer)

	#decomposition of derivative of the loss function with respect to weights1
        d_z_layer1_d_w1 = self.input.T

        d_layer1_d_z_layer1 = activation_function_derivative(self.activation_function_type,self.layer1)



        d_z_output_layer_d_layer1 = self.weights2.T

	#d_output_layer_d_z_output_layer = sigmoid_derivative(self.output) #we have this value already
		
	#d_cost_d_output_layer = 2*(self.y - self.output) #we have this value already

        d_weights1 = np.dot(d_z_layer1_d_w1, d_layer1_d_z_layer1 * np.dot(d_output_layer_d_z_output_layer * d_cost_d_output_layer, d_z_output_layer_d_layer1) )


        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        
np.random.seed(1)
NN = NeuralNetwork(X,y, 0)
for i in range(1500): # trains the NN 1,000 times
    if i % 100 ==0: 
#    if i == 1499:
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)
