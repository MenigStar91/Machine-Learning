import numpy as np
import pandas as pd

# to convert continuous column into categorical
class Split():
    def __init__(self):
        pass

    # impurity
    def gini(self, column):
        x=np.array(column.value_counts())
        y=sum(x)
        return 1- np.sum((x/y)**2)

    # cont_to_cat() 
    def cont_to_cat(self,column):
        split_value=1
        max_gain=0
        for threshold in column.unique():
            split_a=column[column<threshold]
            split_b=column[column>=threshold]
            gain = self.gini(column)- len(split_a)/(len(split_a)+len(split_b))*self.gini(split_a) - len(split_b)/(len(split_a)+len(split_b))*gini(split_b)
            if(gain>max_gain):
                max_gain=gain
                split_value=threshold
        return split_value

    # split the entire dataframe, converting it into a completly categorical dataframe
    def optimal_split(self, data, inplace=False):
        if(inplace!=True):
            data=data.copy(deep=True)
        for column in data:
            if data[column].dtype=="float64":
                split_value=self.cont_to_cat(data[column])
                data.loc[data[column]<split_value, column]=0
                data.loc[data[column]>=split_value, column]=1
        return data


 
# decision tree for numerical attributes, both classifing and regression prediction depending on its "tree_type"
# default 2 splits for any attribute is taken
class DecisionTree():

    def __init__(self, max_depth =10, value=None, tree_type = "classifier"):
        self.max_depth= max_depth
        self.gain=None
        self.attribute=None
        self.nodes ={}
        self.threshold = None
        self.value = value
        if (tree_type not in ["classifier", "regressor"]):
            return "invalid classifier"
        self.tree_type = tree_type
    
    # for training of model 
    def TrainModel(self, X, Y):
        self.split_tree(X,Y)
        if self.gain==0:
            if(self.tree_type=="classifier"):
                self.value=Y.value_counts().idxmax()
            else:
                self.value = np.mean(Y)   
        elif(self.max_depth <=0):
            if(self.tree_type=="classifier"):
                self.value=Y.value_counts().idxmax()
            else:
                self.value = np.mean(Y)
        else:
                node_data_smaller = X[X[self.attribute]<self.threshold]
                node_target_smaller = Y[X[self.attribute]<self.threshold]

                node_data_greater = X[X[self.attribute]>=self.threshold]
                node_target_greater = Y[X[self.attribute]>=self.threshold]
                
                self.nodes[0].TrainModel(node_data_smaller, node_target_smaller)
                self.nodes[1].TrainModel(node_data_greater, node_target_greater)

    def gini(self,column):
        x=np.array(column.value_counts())
        y=sum(x)
        return 1- np.sum((x/y)**2)

    def impurity(self,column):
        if(self.tree_type=="regressor"):
            return np.var(column)
        else :     
            x=np.array(column.value_counts())
            y=sum(x)
            return 1- np.sum((x/y)**2)

    def information_gain(self, column, target):
        split_value=1
        max_gain=0
        for threshold in column.unique():
            split_a=target[column<threshold]
            split_b=target[column>=threshold]
            gain = self.impurity(target)- len(split_a)/(len(split_a)+len(split_b))*self.impurity(split_a) - len(split_b)/(len(split_a)+len(split_b))*self.impurity(split_b)
            if(gain>max_gain):
                max_gain=gain
                split_value=threshold
        return (max_gain,split_value)

    def best_split_attribute(self,data, target):
        self.gain = 0
        attribute=data.columns[0]
        threshold =0
        for feature in data:
            feature_gain, feature_threshold=self.information_gain(data[feature], target)
            if(feature_gain > self.gain):
                self.gain=feature_gain 
                attribute=feature
                threshold= feature_threshold
        return (attribute,threshold)

    def split_tree(self,data, target):
        self.attribute, self.threshold=self.best_split_attribute(data, target)
        for x in [0,1]:
            self.nodes[x]=  DecisionTree(self.max_depth-1, tree_type=self.tree_type)
        return
        
    # for testing of model
    def traverse(self, row):
        if self.value !=None:
            return self.value
        else:
            if row[self.attribute]<self.threshold:
                return self.nodes[0].traverse(row)
            else:
                return self.nodes[1].traverse(row)

    def Predict(self, test_data):
        prediction = pd.Series(dtype = "float64")
        for row in test_data.index:
            prediction.loc[row]=self.traverse(test_data.loc[row,:])
        return prediction