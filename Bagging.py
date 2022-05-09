from Decision_Tree import *

# bagging 
class bagging():
  def __init__(self,n_estimators):
    self.n_estimator = n_estimators
    self.model = []

  # train n_estimator classifiers, selecting data based on allowed replacements
  def TrainModel(self, ty, X, Y, n=None, depth=10):
      if(n==None):
        self.n = len(X)
      else :
        self.n = n
      
      for i in range (self.n_estimator):
        index = np.random.choice(X.index, self.n, replace = True)
        X_train=X.loc[index]
        y_train=Y.loc[index]
        classifier = DecisionTree(tree_type = ty, max_depth = depth)
        classifier.TrainModel(X_train,y_train)
        self.model.append(classifier)

  # returning the best predicted output, from all weak classifiers created
  def Predict(self, X):
      self.output = pd.DataFrame()
      i=0
      for model in self.model:# check if i can be replaced with series index
        self.output.insert(i,i,np.array(model.Predict(X)),True)
        i+=1
      if self.model[0].tree_type == "regressor":
        return self.output.mean(axis=1)
      else:
        return self.output.mode(axis = "columns")[0]
  
  # return the predicted output of all classifiers as a dataframe
  def Summarize(self):
    return self.output