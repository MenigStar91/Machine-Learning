import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# optimal dimension 
def dimension(model, X, Y, threshold):
    model.Singular_Value_Decomposition(X, Y)
    vector = model.EigenValue.flatten()
    y = vector/np.sum(vector)
    x = range(1,len(X.columns)+1)

    y_step = []
    y_cumulative = 0 
    index = len(X.columns) 
    for i in range(len(X.columns)):
        y_cumulative += y[i]
        y_step.append(y_cumulative)
        if(y_cumulative<=threshold):
            index = i+1

    plt.step(x,y_step)
    plt.bar(x,y)
    plt.plot(x, [threshold]*len(x), linestyle='dashed')
    plt.xlabel("dimention")
    plt.ylabel("data retention ratio")
    plt.show()
    return index

class PCA:
  def __init__(self, X, Y):
    self.features = X.columns
    self.initial_dimention = len(self.features)
    return

  # dataset order reduction
  def PCAreduce(self, X, Y, dim):
    X_use = self.Singular_Value_Decomposition(X, Y)
    vectors = []
    for i in range(dim):
      vectors.append(self.EigenVector[:,i])

    return X.dot(np.transpose(vectors))


  # centralizing the data using mean and std
  def Centralize(self, X):
    mean = X.mean()
    std = X.std()
    X_centralize = pd.DataFrame(columns = self.features)
    for i in self.features:
        X_centralize.loc[:,i]=(X.loc[:,i]-mean[i])/std[i]
    return X_centralize

  # deriving covariance matrix from scratch
  def Covariance(self,X):
    self.covar = pd.DataFrame(columns=self.features)
    N = len(X.index)
    for i in self.features:
      for j in self.features:
        self.covar.loc[i,j] = np.sum(np.dot(np.array(X[i])-self.mean[i], np.transpose(np.array(X[j])-self.mean[j])))/N
    return 


  # principal components

  # 2nd order norm of a vector
  def norm(self, array):
    return np.sqrt(np.sum(array**2))

  # generate a random, unit vector
  def generate_tensor(self, no_of_cols):
    tensor = np.random.rand(no_of_cols,1)
    return tensor/self.norm(tensor)

  # eigen vector and eigen value calculation  
  def create_EigenValue_and_Vector(self):
    EigenVector = np.zeros(shape=(self.initial_dimention, self.initial_dimention))
    EigenValues = np.zeros(shape=(self.initial_dimention, self.initial_dimention))
    eps = 1e-12
    
    vector = self.generate_tensor(self.initial_dimention)
    vector = vector/self.norm(vector)

    EigenVector[:,0] = vector[:,0]
    temp = np.array(np.matmul(self.covar,vector))
    eigenValue = np.matmul(np.transpose(temp),vector)
    EigenValues[0,0]=eigenValue

    past = vector
    vector = temp-float(eigenValue)*vector
    for i in range(self.initial_dimention):
      norm = self.norm(vector)
      if (norm>eps):
        vector = vector/norm 
      else:
        vector = self.generate_tensor(i)

      EigenVector[:,i] = vector[:,0]
      temp = np.array(np.matmul(self.covar,vector))
      eigenValue = np.matmul(np.transpose(temp),vector)
      EigenValues[i,i]=eigenValue

      past = vector
      vector = temp-float(eigenValue)*vector-norm*past

    #self.check(EigenVector)
    return EigenValues, EigenVector

  # to check independence (verify that eigen vectors obtained are correct)
  def check(self, vector):
    if (np.abs(np.linalg.det(vector)) < 1e-8):
      print("Independent")
    else :
      print("Not Independent")

  # sort the order of features along with the eigenVctors and values
  def sort(self, eVal, eVec):
    sorted_eigenValues = np.zeros(shape=(self.initial_dimention,1))
    sorted_eigenVector = np.zeros(shape=(self.initial_dimention, self.initial_dimention))
    featureOrder = []

    values = [np.abs(eVal[i][i]) for i in range(self.initial_dimention)]
    
    for i in range(self.initial_dimention):
      index = values.index(max(values))
      sorted_eigenValues[i] = eVal[index][index]
      sorted_eigenVector[:,i] = eVec[:,index]
      values[index]=0
      featureOrder.append(self.features[index])

    return sorted_eigenValues, sorted_eigenVector, featureOrder

  # Singular Value Decomposition 
  def Singular_Value_Decomposition(self, X, Y):
    X_use = self.Centralize(X)
    self.mean = X_use.mean()
    self.std = X_use.std()
    self.Covariance(X_use)
    var1, var2 = self.create_EigenValue_and_Vector()
    self.EigenValue, self.EigenVector, self.PrincipalComponents = self.sort(var1, var2)
    return X_use