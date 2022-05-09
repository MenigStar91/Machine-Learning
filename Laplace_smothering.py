import pandas as pd
import numpy as np

# laplace smmothering
class nlp():
  def __init__(self, index_train, index_test):
    self.index_train = index_train
    self.index_test = index_test
    return

  def TrainModel(self, X, Y):
    self.classes = Y.unique()
    self.prior_probabilities = self.prior(Y)
    self.likelihoods = {}
    self.total = {}
    self.smoothered_likelihood = {}
    self.max = max(X.loc[:,'TermId'])

    for class_ in self.classes :
      self.likelihoods[class_] , self.total[class_] = self.likelihood(X[X['DocId'].isin(Y[Y==class_].index)], Y[Y==class_])
    return
 
  def prior(self,column):
    value = column.value_counts()
    return np.log(value/np.sum(value))

  def likelihood(self, X,Y):
    words = X.loc[:,'TermId'].unique()
    total = np.sum(X.loc[:,'count'])
    likelihood = {}
    for word in words:
      likelihood[word] = np.sum(X[X['TermId']==word]['count'])/total 

    return likelihood, total

  # smoothing technique to handle zero probability values
  def laplace_smoothing(self, x, class_):
    map = self.likelihoods[class_]
    if x['TermId'] in map.keys():
      return np.log((map[x['TermId']]*self.total[class_]+1)/(self.total[class_]+self.max))
    else :
      return np.log(1/(self.total[class_]+self.max))

  def predict_probability(self, X, class_):
    value = 0
    for row in X.index:
        value = value + self.laplace_smoothing(X.loc[row,:], class_)
    return value+self.prior_probabilities[class_]

  def traverse(self, X):
    array = pd.Series()
    for x in self.classes:
      array.loc[x]=self.predict_probability(X,x)
    return array

  def Predict(self, test_data):
    prediction = pd.Series(dtype = "float64")
    if self.max<max(test_data.loc[:,'TermId']):
      self.max = max(test_data.loc[:,"TermId"])

    for id in self.index_test:
      eval = self.traverse(test_data[test_data['DocId']==id])
      prediction.loc[id] = eval.idxmax()
    return prediction 