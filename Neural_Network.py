import numpy as np
import pandas as pd
import sklearn.metrics as sm
from sklearn.utils import shuffle

# hidden layer
class hidden_layer():
    # initializing the layer
    def __init__(self, input_size, output_size, activation,weights = "Random"):
      self.input_size = input_size
      self.output_size = output_size
      self.ActivationFunction = activation
      self.dic = {'Relu' : self.Relu, "Tanh" : self.Tanh, "Sigmoid" : self.Sigmoid}
      if(weights=='Zero'):
        self.weights = np.zeros((input_size,output_size))
      elif (weights=="Constant"):
          self.weights = np.full((input_size,output_size),1/input_size)
      else:
        self.weights = np.random.rand(input_size,output_size)
  
      self.bias = np.random.rand(1, output_size)
      return 

    # activation function of layer
    def activate(self, inputs):
      assert(inputs.shape[1]==self.input_size)
      output = np.matmul(inputs, self.weights) + self.bias 
      self.input = output
      output = self.dic[self.ActivationFunction](output)  #check syntax
      self.output = np.average(output,0)[:,None]
      return output 

    # updating the parameters
    def update(self, learning_rate, new_values_weights, new_values_bias):
      self.weights = self.weights - learning_rate*new_values_weights
      self.bias = self.bias - learning_rate*new_values_bias
      return

    # activation functions with their respective gradient descent
    def Relu(self,input):
      self.derivative = np.mean((np.transpose(input)>0)*1,1)
      return np.maximum(np.zeros((100,self.output_size)),input)

    def Tanh(self,input):
      self.derivative = np.mean(((4*np.exp(2*np.transpose(input)))/((1+np.exp(2*np.transpose(input)))**2)),1)
      return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def Sigmoid(self,input):
      result = 1/(1+np.exp(-np.transpose(input)))
      self.derivative = np.mean(np.multiply(result , 1-result),1)
      return 1/(1+np.exp(-input))


# multi-layer perceptron
class mlp():      # determining model and class function?

    # initializing hidden layers
    def __init__(self, input_size, no_of_layers, sizeOfLayers, activation_function, weights, output_size, output_activation):  
              # sizeOfLayers is an array of dim (no of layers) providing their output sizes/no of neurons
              # activation_function and weights can be contant of different for each hidden layers, (assumed the two to take same case here)
      self.input_size = input_size
      self.output_size = output_size
      self.no_of_layers = no_of_layers
      self.layers = []
      if(type(activation_function)==str):
        for i in range(no_of_layers):
          if(i==0):
            self.layers.append(hidden_layer(input_size,sizeOfLayers[i], activation_function, weights))
          else :
            self.layers.append(hidden_layer(sizeOfLayers[i-1], sizeOfLayers[i], activation_function, weights))
      else:
        for i in range(no_of_layers):
          if(i==0):
            self.layers.append(hidden_layer(input_size, sizeOfLayers[i], activation_function[i], weights[i]))
          else :
            self.layers.append(hidden_layer(sizeOfLayers[i-1], sizeOfLayers[i], activation_function[i], weights[i]))

      self.output_layer=hidden_layer(sizeOfLayers[-1],output_size,output_activation) # assuming the output layer to always have random weight initialization
      return

    # training properties
    def SetProperties(self, batch_size, num_epochs, learning_rate, num_classes, features):
      self.batch_size = batch_size
      self.num_epochs = num_epochs
      self.learning_rate = learning_rate  
      self.features = features
      return

    # loss function : cross entropy loss
    def loss(self, predicted, labels):
      sum_score = 0.0
      for i in labels.index:
          sum_score += labels[i] * np.log(1e-15 + predicted.loc[i,labels[i]])
      mean_sum_score = 1.0 / len(labels) * sum_score
      return -mean_sum_score

    def category_loss(self, predicted, labels):
      loss = pd.Series()
      for i in range(max(labels)+1):
        loss.loc[i] = self.loss(predicted[labels==i],labels[labels==i])
      return np.array(loss)

    # forward propogate the input
    def forward_pass(self, input):
      output = input 
      for i in range(self.no_of_layers):
        output = self.layers[i].activate(output)
      return self.output_layer.activate(output)

    # backward propogate the error
    def backward_pass(self, weight,loss):
      #print(weight.shape, loss.shape)
      self.output_layer.update(self.learning_rate, weight,loss)
      weight = np.multiply(weight, self.output_layer.derivative)
      loss = np.transpose(np.matmul(self.output_layer.weights, weight))
      #print(weight.shape)
      for i in range(self.no_of_layers-1,-1,-1):
        self.layers[i].update(self.learning_rate, weight,loss)
        weight = np.multiply(weight, self.layers[i].derivative)
        loss = np.transpose(np.matmul(self.layers[i].weights, weight))
      return 

    # training/testing accuracy
    def accuracy(self, predicted, labels):
        pred = pd.Series()
        for _ in predicted.index:
          pred.loc[_]= predicted.loc[_,:].argmax()
        return sm.accuracy_score(np.array(pred), labels)

    # train 
    def train(self,X_train, y_train):
      self.no_of_batches = int(len(X_train) / self.batch_size)
      for epoch in range(1):#self.num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        training_accuracy = 0.0
        running_loss = 0.0

        for i in range(self.no_of_batches):
          start = i * self.batch_size
          end = start + self.batch_size
          inputs = X_train[start:end]
          labels = y_train[start:end]

          outputs = self.forward_pass(inputs)
          outputs = outputs/np.max(outputs,1)[:,None]   
    
          running_loss+=self.loss(outputs, labels)
          self.backward_pass(self.category_loss(outputs, labels), self.loss(outputs,labels))
          training_accuracy += self.accuracy(outputs, labels)

        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f'  %(epoch+1, running_loss / (i+1), training_accuracy/(i+1)))  
      return 

    # predict 
    def predict(self, X_test):
      output = self.forward_pass(X_test)
      pred = []
      for i in range(len(X_test)):
        pred.append(np.where(output[i] == np.amax(output[i]))) # assigning max probability class as perdicted value
      return np.array(pred)