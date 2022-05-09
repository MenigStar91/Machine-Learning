import pandas as pd
import numpy as np

# implement a kmeans algorithm from scratch (q2 part a,b)
class K_Means():
  # k:n_clusters,
  def __init__(self, n_clusters, max_iterations=-1, centers=pd.DataFrame()):
    # user provided value of k
    self.n_clusters = n_clusters
    self.max_iterations = max_iterations
    # class to store cluster centers (can be initialized via user input)
    self.centers = centers
  
  # initialize categorization and centroids (if not provided)
  def fit(self,X):
    # class to store clusters 
    self.class_=pd.Series([0]*len(X.index), index = X.index)
    if(self.centers.shape[0]==0):
      self.centers = pd.DataFrame(X.loc[np.random.choice(X.index,self.n_clusters),:].values)
    # classify
    self.cluster(X)
    # return categorization learnt
    return self.class_
  
  # classification of training dataset
  def cluster(self, X):
    start = 0
    # stop iteration condition (max iteration reached)
    while(start<self.max_iterations):
      # categorization based on centroids
      self.categorize(X)
      distance = 0
      # recenter based on data distribution
      for i in np.unique(self.class_):
        distance+=self.recenter(X.loc[self.class_[self.class_==i].index],i)
      # best categorization state achieved, no change from previous classification observed
      if(distance<1e-5):
        break
      start+=1
    return 

  # rechoosing the cetroids for next iteration
  def recenter(self, data, index):
    temp = self.centers.loc[index,:]
    # taking mean of points belonging to common category, as next best centroid
    self.centers.loc[index,:] = data.mean()
    return self.distance(data.mean(), temp)

  # classifing the instances to closest centroids
  def categorize(self,X):
    for i in X.index:
      category = 0
      best_class = 0
      min_distance = 1e12
      # finding distance from all centroids
      for c in self.centers.index:
        distance = self.distance(X.loc[i,:],self.centers.loc[c,:])
        # assigning the min_distance as the best category
        if(distance<min_distance):
          min_distance = distance
          best_class = category
        category+=1
      self.class_[i]=best_class
    return

  # return euclidean distance between two points
  def distance(self, row, center):
    return np.sqrt(np.sum((row-center)**2))