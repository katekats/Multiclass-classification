

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import itertools
from sklearn.neural_network import MLPClassifier 
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
# this function manually makes grid search using cv for finding the best parameter
#for MLP    
def man_grid_searchCV(X, y):
    k=1 
 # choose folds = 10 for the cross validation   
    cv = KFold(n_splits=10)
 # create an array with three columns   
    df2 = pd.DataFrame({"i" : [], "j" : [], "f1_score" : []})
 # making grid search for the number of units for the two layer of the MLP   
    for i in range(10, 100, 20):
        for j in range (2, 8, 2):
             mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(i, j), random_state=1) 
  # calculate the cross_val_score and taking the mean for each model  
             scores = cross_val_score(mlp_model, X, y, cv = cv, scoring='f1')
             mean_score = np.mean(scores)
# store the mean_scores of all models in a dataframe
             df2.set_value((k), ("i", "j", "f1_score"),
                                 (i, j, mean_score))
             k += 1 
# printing the model parameters that gives us the best mean_score                   
    print(df2[["i", "j", "f1_score"]][df2["f1_score"] == df2["f1_score"].max(axis=0)])        
   
# this function uses a dataframe for making grid search cv for MLP
#   where the records are classified in two classes
# the first is the class with the bigger number of records and the other class includes
#   the records of the rest classes with smaller number of records        
def one_vs_rest(df1):
    df1[295].unique()
    class_samples = {}
 # for each class in column 295   
    for i in df1[295].unique(): 
# find the records that belong in class i and put the pair class-number of records
# in a dictionary    
        class_samples.update({i:(len(df1[df1[295] == i].axes[0]))})
    global maximum 
# find the class with the max number of records    
    maximum = max(class_samples, key=class_samples.get) 
# if the class is this with the max number of records assign to it the value 1   
     # otherwise assign to the class the value 0  
   
    df1[295].loc[df1[295] != maximum] = 10
    df1[295].loc[df1[295] == maximum] = 11
    df1[295].replace(10, 0, inplace=True)
    df1[295].replace(11, 1, inplace=True)
   
# asign to y the column 295    
    y = df1[[295]]
# and drop column 295 from the df    
    df1.drop(df1.columns[295], axis = 1, inplace = True)
# standardize the features of df
    std_scale = preprocessing.StandardScaler().fit(df1)
    df_std = std_scale.transform(df1)
    df_std = pd.DataFrame(df_std, columns = df1.columns)
# assign df to X    
    X = df_std
# perform grid search cv 
    man_grid_searchCV(X.as_matrix(), y.as_matrix().ravel())

     
     
# read a csv file     
df = pd.read_csv(r"C:\Dataset.csv", header=None, delimiter=",")
# trandorm the categorical values of column 295 to numerical
label_encoder = preprocessing.LabelEncoder()
df[[295]] = label_encoder.fit_transform(df[[295]]) 

#df1 = df.drop(df[df[295] == 2].index)
#df1 = df1.drop(df[df[295] == 3].index)
#df1 = df1.drop(df[df[295] == 1].index)
df1 = df.copy()
#df1.index = range(len(df1.axes[0]))
#print(df1)
one_vs_rest(df1) 
