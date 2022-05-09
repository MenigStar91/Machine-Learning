import pandas as pd
import sklearn.metrics as sm

# K-fold 
class K_fold():
  def __init__(self, no):
    self.data = pd.DataFrame(columns=range(no))
    self.target = pd.Series(dtype="float64")
    
  def append(self, index, train_row, target_row):
    self.data.loc[index]=train_row
    self.target.loc[index]=target_row

# kfold cross validation implementation
class cross_val_score():

  def __init__(self):
     return 
  # create n folds, based on X,Y and return the accuracy for each fold
  def cross_val_score(self, model, X, Y, cv, random_seed =1):
    self.model=model
    self.splits = cv
    self.folds = self.split_data(len(X.columns))
    self.feed_data(X,Y, random_seed)
    return self.eval_trainModel()

  # creating k-folds
  def split_data(self, no_of_features):
    folds = {}
    for i in range(self.splits):
        folds[i] = K_fold(no_of_features)
    return folds

  # for splitting data into n parts
  def feed_data(self, X, Y, random_seed):
    index  = X.index
    max_iteration = len(Y)
    seed_position = 1
    iteration =0
    key = 0
    row = 0

    while(iteration < max_iteration):
      self.folds[key].append(index[row], X.loc[index[row],:].values,Y.loc[index[row]])
      row = row + random_seed
      if row>=max_iteration:
        row = seed_position
        seed_position+=1
      key = (key+1)%self.splits
      iteration+=1
    return

  # returning the accuracy of a single fold over training data
  def solve(self,model,X_train,y_train,X_test,y_test):
    model.TrainModel(X_train,y_train)
    y_pred=model.Predict(X_test)
    if (model.tree_type == "classifier"):
      return sm.accuracy_score(y_pred,y_test)
    else :
      return sm.mean_squared_error(y_test,y_pred)

  # make training data for each fold, selected from the splitting data ; and train
  def eval_trainModel(self):
    scores =[]
    for key in range (self.splits):
      X_validate =self.folds[key].data
      y_validate = self.folds[key].target
      X_train = pd.concat( [ self.folds[i].data for i in range(self.splits) if i!=key ])
      y_train = pd.concat( [ self.folds[i].target for i in range(self.splits) if i!=key])
      scores.append(self.solve(self.model,X_train,y_train,X_validate,y_validate))
    return scores
