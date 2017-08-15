
import itertools
import matplotlib.pyplot as plt
#from itertools import cycle
import numpy as np  
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import tree, preprocessing, metrics, linear_model
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    cm = np.around(cm,decimals=2)
    print("Normalized confusion matrix")
    print(cm)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
    
# this function uses a dataframe for classifying  using MLP
#   where the records are classified in two classes
# the first is the class with the bigger number of records and the other class includes
#   the records of the rest classes with smaller number of records   
def one_vs_rest(df):
    df[295].unique()
    class_samples = {}
 # for each class in column 295   
    for i in df[295].unique(): 
# find the records that belong in class i and put the pair class-number of records
# in a dictionary    
        class_samples.update({i:(len(df[df[295] == i].axes[0]))})
    print(class_samples)    
    global maximum 
# find the class with the max number of records    
    maximum = max(class_samples, key=class_samples.get)
# if the class is this with the max number of records assign to it the value 1   
     # otherwise assign to the class the value 0    
    df[295].loc[df[295] != maximum] = 10
    df[295].loc[df[295] == maximum] = 11
    df[295].replace(10, 0, inplace=True)
    df[295].replace(11, 1, inplace=True)
# asign to y the column 295 
    y = df[[295]]
 # and drop column 295 from the df       
    df.drop(df.columns[295], axis = 1, inplace = True)
# standardize the features of df  
    std_scale = preprocessing.StandardScaler().fit(df)
    df_std = std_scale.transform(df)
    df_std = pd.DataFrame(df_std, columns = df.columns)
 # assign df to X       
    X = df_std
#  use 10 folds for cross validation   
    cv = KFold(n_splits=10)
 # use MLP model with L1 with the parameters from grid_search_cv  
    mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(90, 2), random_state=1)  
 # use 10 cross validation for taining & testing the model   
    for train, test in cv.split(X,y):
        X_train, X_test = X.ix[train], X.ix[test]
        y_train, y_test = y.ix[train], y.ix[test]
 # training
        mlp_model.fit(X_train,y_train) 
# testing
        y_predict = mlp_model.predict(X_test) 
# call the confusion matrix 
        cnf_matrix = confusion_matrix(y_test, y_predict)
        print(accuracy_score(y_test, y_predict))

    np.set_printoptions(precision=2)

# Plot normalized confusion matrix
    plt.figure()
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Normalized confusion matrix')
    plt.show()
#
     
# read a csv file     
df = pd.read_csv(r"C:\Dataset.csv", header=None, delimiter=",")
# trandorm the categorical values of column 295 to numerica
label_encoder = preprocessing.LabelEncoder()
df[[295]] = label_encoder.fit_transform(df[[295]]) 
# droping the three classes with the bigger numbers records and making prediction
# for the other two
df1 = df

#df1 = df.drop(df[df[295] == 2].index)
#df1 = df1.drop(df[df[295] == 3].index)
#df1 = df1.drop(df[df[295] == 1].index)
#df1.index = range(len(df1.axes[0]))
one_vs_rest(df1)
