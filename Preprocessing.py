import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''Date Time'''


from datetime import datetime
# input, storing current time as string
date_time = str(datetime.now()) 

# convertion
current=datetime.strptime(date_time , "%Y-%m-%d %H:%M:%S.%f")
Date = current.strftime("%Y %b %d")
Time = current.strftime("%H:%M:%S")

data = pd.load_csv("url")


'''Preprocessing lines'''


# returns name of all columns containg at least 1 null value
data.loc[:, data.isnull().any()].columns

# data.dropna(subset=[]) : subset contain list of all column names, for which the rows have to be dropped
data.dropna(subset=['Rear.seat.room'], inplace=True)

# filling in missing values
data['Luggage.room'].replace(to_replace=[None], value=data['Luggage.room'].mode(), inplace=True)

# setting appropriate datatypes
data["Cylinders"]=data.loc[:,"Cylinders"].astype("int64")

# check multiple column conditions
data.loc[data['bill_length_mm'].isnull() | data['bill_depth_mm'].isnull() | data['flipper_length_mm'].isnull() | data['body_mass_g'].isnull()]

# ordinal encoding for ordinal data
mapping = {"None":0, "Driver only":1, "Driver & Passenger":2}
data["AirBags"] = data["AirBags"].replace(mapping)

# one hot encoding for nominal data of equal bias
data = pd.get_dummies(data, columns = ['DriveTrain', 'Origin', 'Man.trans.avail'])

# label encoding for nominal data
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
data['Type']=lbcode.fit_transform(data['Type'])


''' Visualization'''


# heat map
correlation_matrix = data[data.columns[:-1]].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

# histogram
fig = plt.figure(figsize = (9,9))
ax = fig.gca()
data.hist(ax=ax)
plt.show()


'''SFS'''


# for SFS
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier() # any
sfs = SFS(model,
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'accuracy',
          verbose=2
)


'''Imbalanced Data handling'''


# for imbalanced dataset
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.under_sampling import RandomUnderSampler as RUS

s = SMOTE()
X , Y = s.fit_resample(X,Y.ravel())

def oversample(self,X,y):
    ros=ROS(random_state=42)
    X_ros,y_ros=ros.fit_resample(X,y)
    return X_ros,y_ros

def undersample(self,X,y):
    rus=RUS(random_state=42)
    X_rus,y_rus=rus.fit_resample(X,y)
    return X_rus,y_rus


'''Hyperparameter tuning'''


# hyperparameter tuning : Grid search cv, random_search, bayes optimizer
from sklearn.model_selection import GridSearchCV
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred

model = lgb.LGBMClassifier()
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}

model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, param_grid, cv=5, scoring_fit='accuracy')
print(model.best_score_)
print(model.best_params_)


#Imported BayesianOptimization
from bayes_opt import BayesianOptimization

# Function to calculate and return average of score (scoring parameter manually chosen, here 'roc_auc') after performing 5-fold cross-validation
def gbm_cl_bo(max_depth, n_estimators):
    params_gbm = {}
    params_gbm['max_depth'] = round(max_depth)
    params_gbm['n_estimators'] = round(n_estimators)
    scores = cross_val_score(RFC(random_state=42, **params_gbm),                                    X_train_ros, y_train_ros, scoring='roc_auc', cv=5).mean()
    score = scores.mean()
    return score

params_gbm ={
    'max_depth':(5, 10),
    'n_estimators':(50, 500),
}

gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=42)
gbm_bo.maximize(init_points=20, n_iter=4)