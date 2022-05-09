import numpy as np

class ica():
  def __init__(self, iterations, X):
    self.W = np.zeros((X.shape[1], X.shape[1])) # random initial value
    self.no_of_features = X.shape[1]
    self.iterations = iterations
    self.prediction = self.analyse(X.T)
    return 

  def analyse(self, X):
    X = self.center(X)
    signal = self.whiten(X)
    # create demixing matrix
    for i in range(self.no_of_features):
      w = np.random.rand(self.no_of_features)
      for j in range(self.iterations):
        # update u
        next_w = self.next(w,signal,i)
        w=next_w
        if(self.update(next_w,w)):
          break  
      self.W[i,:]=w
    # return independent source signals function
    return np.dot(self.W,X)

  def center(self, X):
    # centralize the input signal of all individual sources
    mean = X.mean(axis=1)
    for i in range(X.shape[0]):
      X[i]=X[i]-mean[i]
    return X
  
  def whiten(self, X):
    self.cov = np.cov(X)
    # D == diagnol matrix (eigen values) of corelation matrix
    # E == eigen vectors of corelation matrix
    eigenvalues,E = np.linalg.eigh(self.cov)
    # form diagnol matrix of eigenvalues
    D = np.diag(eigenvalues)

    # whittening formula : x = ED^-0.5 E^t x
    D_inv = np.sqrt(np.linalg.inv(D))
    X = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X
  
  def next(self, W, X,i):
    # update demixing matrix
    matrix = np.dot(W.T, X)
    W = (X * np.tanh(matrix)).mean(axis=1) - (1-np.tanh(matrix)**2).mean() * W 

    # check independency of sources 
    if(i!=0):
      W -= np.dot(np.dot(W, self.W[:i].T), self.W[:i])

    # self normalize
    W = W / np.sqrt((W ** 2).sum()) 
    return W
  
  def update(self,w_new,w):
    # coumpute norm
    distance = np.abs(np.abs((w * w_new).sum()) - 1)

    # check if the identity matrix has been obtained of not (norm==1)
    if (distance<1e-5):
      return True
    else :
      return False
