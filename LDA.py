import pandas as pd
import numpy as np

class LDA:
  def __init__(self, X, Y):
    self.features = X.columns
    self.initial_dimention = len(self.features)
    self.classes = Y.unique()
    return 

  # dataset order reduction
  def LDAreduce(self, X, Y, dim):
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

  # witten class scatter metric
  def within_class_scatter(self, X, Y):
    metric = np.zeros(shape=(self.initial_dimention, self.initial_dimention))

    for class_ in self.classes :
      x = X.loc[Y==class_, :]
      for i in x.index:
        temp = np.array(x.loc[i,:])-np.array(self.class_wise_mean[self.class_wise_mean.index==class_])
        metric += np.matmul(np.transpose(temp),temp)
    return metric

  # between class scatter metric
  def between_class_scatter(self, X, Y):
    metric = np.zeros(shape=(self.initial_dimention, self.initial_dimention))

    for class_ in self.classes:
      x = X.loc[Y==class_, :]
      temp = np.array(self.class_wise_mean[self.class_wise_mean.index==class_]) - np.array(x.mean())
      metric += (np.matmul(np.transpose(temp), temp))*len(x.index)
    return metric

  # covariance matrix
  def Covariance(self, X, Y):
    sw, sb = self.within_class_scatter(X, Y), self.between_class_scatter(X, Y)
    self.covar = np.matmul(np.linalg.inv(sw),sb)
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
        vector = self.generate_tensor(self.initial_dimention)

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
    self.class_wise_mean = X_use.groupby(by=Y).mean()
    self.std = X_use.std()
    self.class_wise_std = X_use.groupby(by=Y).std()
    self.Covariance(X_use, Y)
    var1, var2 = self.create_EigenValue_and_Vector()
    self.EigenValue, self.EigenVector, self.PrincipalComponents = self.sort(var1, var2)
    return X_use