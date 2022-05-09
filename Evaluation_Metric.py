import pandas as pd
import numpy as np

def metric( y_test, y_predicted ):
  df=pd.DataFrame()
  df['y_test']=y_test
  df['y_predicted']=y_predicted

  df[df['y_test']==0 ][df['y_predicted']==0].shape[0] 

  dictionary={}
  dictionary['true_negative'] = df[df['y_test']==0 ][df['y_predicted']==0].shape[0] 
  dictionary['false_positive'] = df[df['y_test']==0 ][df['y_predicted']==1].shape[0] 
  dictionary['false_negative'] = df[df['y_test']==1 ][df['y_predicted']==0].shape[0] 
  dictionary['true_positive'] = df[df['y_test']==1 ][df['y_predicted']==1].shape[0] 

  return dictionary

# evaluation / confusion natrix
def confusion_matrix(y_test, y_predicted):
  dt=metric(y_test, y_predicted)
  return np.array([[dt['true_negative'], dt['false_positive']], [dt['false_negative'], dt['true_positive']]])

# accuracy
def avg_accuracy(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return (dt['true_negative']+dt['true_positive'])/(dt['true_negative'] + dt['false_positive'] + dt['false_negative'] + dt['true_positive']) *100

# class wise accuracy
def class_accuracy(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return ((dt['true_negative']/(dt['true_negative'] + dt['false_positive']))+(dt['true_positive']/(dt['false_negative'] + dt['true_positive']))) *100

# class wise accuracy for multiple classes (more than 2)
def class_wise_accuracy(y_test, y_pred):
  classes ={}
  for x in y_pred.unique():
    classes[x] = y_test[y_test['species']==x][y_pred==x].shape[0] / y_test[y_test['species']==x].shape[0]
  return classes 

# precision
def precision(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return (dt['true_positive']/(dt['true_positive'] + dt['false_positive']))*100

# recall
def recall(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return (dt['true_positive']/(dt['true_positive'] + dt['false_negative']))*100

# f1 - score
def f1_score(y_test,y_predicted):
  return 2/(1/precision(y_test,y_predicted) + 1/recall(y_test, y_predicted))

# senstivity
def sensitivity(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return (dt['true_positive']/(dt['true_positive'] + dt['false_negative']))*100

# specificity
def specificity(y_test,y_predicted):
  dt=metric(y_test, y_predicted)
  return (dt['true_negative']/(dt['true_negative'] + dt['false_positive']))*100