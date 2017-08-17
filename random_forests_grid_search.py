from __future__ import print_function

import os
import subprocess

from time import time
from operator import itemgetter
from scipy.stats import randint

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  cross_val_score
from sklearn.model_selection import KFold
import itertools
import matplotlib.pyplot as plt
from sklearn import tree, preprocessing, metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
   
def report(grid_scores, n_top=1):
    """Report top n_top parameters settings, default n_top=1.

    Args
    ----
    grid_scores -- output from grid search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters 
 
    
def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    grid_search.fit(X, y)


    top_params = report(grid_search.grid_scores_)
    return  top_params
    

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
    # choose the parameters values that we will perform grid search cv
    rf_model = RandomForestClassifier(random_state=30)
    param_grid = { "n_estimators"  : [50, 100, 200],
           "max_features"      : [10, 40, 70, 100],
           "max_depth"         : [10, 20,40, 60],
           "min_samples_split" : [2, 4] ,
           "bootstrap": [True, False]}
 #   run grid search cv      
    run_gridsearch(X.as_matrix(), pd.DataFrame.as_matrix(y).ravel(), rf_model, param_grid, cv=5)
 
     
     
# read a csv file     
df = pd.read_csv(r"C:\Dataset.csv", header=None, delimiter=",")
# trandorm the categorical values of column 295 to numerical
label_encoder = preprocessing.LabelEncoder()
df[[295]] = label_encoder.fit_transform(df[[295]]) 

df1 = df.copy()
one_vs_rest(df1) 
