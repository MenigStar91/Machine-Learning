import numpy as np
import pandas as pd

# probability distribution of each class
def prior(column):
  value = column.value_counts()
  return value/np.sum(value)

# count occurance table for feature value vs target class
def table(feature, col):
  value=pd.DataFrame(index=col.unique(), columns=np.sort(feature.unique())).fillna(0)
  for i in feature.unique():
    value.loc[col[feature==i].unique(),i]= col[feature==i].value_counts()
  return value

# class wise likelihood function (sampling the table values into probability)
def likelihood(feature, col):
  value=table(feature,col)
  for i in value.index:
    value.loc[i,:]= value.loc[i,:]/np.sum(value.loc[i,:])
  return value

# posterior probability for all classes (giving proper weightage to class distribution, in our likelihood function)
def posterior_probability(feature,col):
  value=pd.DataFrame(index=col.unique(), columns=np.sort(feature.unique())).fillna(0)
  Likelihood=likelihood(feature,col)
  Prior=prior(col)
  for i in Prior.index:
    value.loc[i,:]= Likelihood.loc[i,:]*Prior[i]

  for i in value.columns:
    value.loc[:,i]=value.loc[:,i]/np.sum(value.loc[:,i])
  return value


# naive bayes classifier from scratch
# varient : Guassian
class Naive_Bayes():
  def __init__(self):
    pass
  
  # storing the mean and variance of our features, so that they can be used to calculate probabilty (in future)
  def TrainModel(self, X,Y):
    self.mean = X.groupby(by=Y).mean()
    self.std = X.groupby(by=Y).std()
    self.prior = prior(Y)
    self.classes = Y.unique()
    self.pred = pd.Series()

  # guassian probability for a value, given mean and varience
  def probability(self, mean, std, value):
    return (1/np.sqrt(2*np.pi*(std**2)))*np.exp(-(value-mean)**2/(2*(std**2)))
  
  # return likelihood of all class vs features, as a dictionary
  def create_dict(self, row):
    dictionary ={}
    for x in self.classes :
      data = pd.DataFrame(columns=row.index)
      for feature in row.index:
          data.loc[x,feature]=self.probability(self.mean[feature][x], self.std[feature][x], row[feature])
      dictionary[x]=data
    return dictionary

  # return evidence over all classes
  def evidence(self, row):
    evidence = 0
    for i in self.classes:
      evidence += self.prior[i]*np.prod(np.array(self.create_dict(row)[i]))
    return evidence

  # calculate posterior probabilities for all classes, and return class wise probability for each test sample
  def traverse(self, row):
    dictionary = self.create_dict(row)
    evidence = self.evidence(row)
    array = pd.Series()
    for x in self.classes:
      likelihood = np.prod(np.array(dictionary[x]))
      array.loc[x]=(likelihood*self.prior[x])/ evidence
    return array

  # predict the class with maximum probability
  def Predict(self, test_data):
    prediction = pd.Series(dtype = "float64")
    for row in test_data.index:
       eval = self.traverse(test_data.loc[row,:])
       self.pred.loc[row]=np.max(eval)
       prediction.loc[row]= eval.argmax()
    return prediction  
  
  def topProbability(self, test_data ):
    self.Predict(test_data)
    return self.pred