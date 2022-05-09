import pandas as pd
import numpy as np

# bayes classifier
class bayes_classifier():
    def __init__(self):
      pass

    def TrainModel(self, data, label):
      self.classes = label.unique()
      self.features = data.columns

      self.means_ = pd.Series(dtype = 'float64')
      self.covariance_ = pd.Series(dtype = 'float64')

      for class_ in self.classes:
        m,c = self.maximum_likelihood_parameter(data[label==class_])
        self.means_.loc[class_] = np.array(m)
        self.covariance_.loc[class_] = np.array(c)   

      self.pred = pd.Series()
      return

    # computing class wise likelihood
    def compute_likelihood(self, feature_values , mean, covariance):
      d = len(feature_values)
      determinant = np.linalg.det(covariance)
      inverse = np.linalg.inv(covariance)
      a = feature_values - mean 

      temp = np.matmul(a,np.matmul(inverse,np.transpose(a)))
      value = 1/np.sqrt((np.power(2*np.pi,d)*determinant)) * np.exp(-1/2*temp)
      return value

    # estimating best parameters for mean and covariance
    def maximum_likelihood_parameter(self, feature_values):
      # assuming there are only 2 features
      d = len(feature_values.columns)
      min1, min2 = feature_values.min() -1
      max1, max2 = feature_values.max() +1

      x1grid = np.arange(min1, max1, 0.1)
      x2grid = np.arange(min2, max2, 0.1)

      xx, yy = np.meshgrid(x1grid,x2grid)
      X_grid = np.c_[xx.ravel(), yy.ravel()]

      best_mean = []
      max_likelihood = 0
      best_covariance = ''
      for i in range(len(X_grid)):
        covar = np.zeros(shape = (d,d))
        for x,j in enumerate(self.features):
          for y,k in enumerate(self.features):
            covar[x][y]= np.sum(np.dot(feature_values.loc[:,j]-X_grid[i][x], np.transpose(feature_values.loc[:,k]-X_grid[i][y])))/(len(feature_values)-1)

        max =1
        for row in feature_values.index:
            temp = self.compute_likelihood(feature_values.loc[row,:], X_grid[i], covar)
            max=max*temp
        if(max>max_likelihood):
          max_likelihood = max
          best_mean = X_grid[i]
          best_covariance = covar

      return (best_mean, best_covariance)

    # training
    def traverse(self, row):
      array = pd.Series()
      for x in self.classes:
        likelihood = self.compute_likelihood(row, self.means_[x], self.covariance_[x])
        array.loc[x]=likelihood
      return array

    def Predict(self, test_data):
      prediction = pd.Series(dtype = "float64")
      for row in test_data.index:
        eval = self.traverse(test_data.loc[row,:])
        self.pred.loc[row]=eval
        prediction.loc[row]= eval.idxmax()
      return prediction 

    def topProbability(self, test_data ):
      self.Predict(test_data)
      return self.pred
