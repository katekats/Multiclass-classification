# Multiclass-classification
A multiclass classifier where the response variable (column256) can be one of the five possible classes A, B, C, D, E. The dataset doesn't include the names of the variables because it's not a public one.  Machine learning algorithms like logistic regression, random forests and MLP are used where for the two last we find the optimal parameters using GridSearch().
The dependent variable (column 296) takes as possible values five discrete values: A, B, C, D, E. As a result we have a multiclass classification problem. By grouping the records per class we find out that the class A has 867 records, class B has 6602, class C has 46882, class D has 9279 and class E has 2507 records. As we can easily see the class C has significantly more records that the other classes. That is the reason why we use one-vs-rest classification by starting with the C as the first class and the rest classes as the second class and transforming the problem into binary classification. Next we leave the C out and we implement again one-vs-rest classification with the next class with the more records after C (that is class D) as the one class and the other three classes as the second class. Scikit- learn has the choice of using one vs rest classification but because we don’t know with which order the classifier uses the classes and because I want the C class to be first the one vs rest classification is implemented manually.
I also notice that many features have sparse data. We standardized the values of the features for making them contribute proportionally and for avoiding that some features being more influent than others.
Firstly I implement one-vs-rest logistic regression using scikit learn that can handle sparse matrices and I choose regularization L1/Lasso. L1 regularization has a built-in feature selection for sparse feature spaces with producing sparse coefficients. Particularly it has the property of producing many coefficients with zero values or very small values with few large coefficients. That
For finding the performance of the algorithm I use 10fold-cross validation and the results are presented in a confusion matrix. In the following figure the class E is the class 1 and the class A is class 0. As we can see we make more accurate predictions for class 1 as for class 0. That can occurs due to the fact that class 1 (class E in the specific case) has significantly more records than class 0 (class A in the specific case). A solution could be using training set with comparable number of records for each class or taking more times the records of the class with the smaller number of records.
A next step is using the random forest classifier (similarly as in logistic regression weuse one-vs-rest classification). For finding the best parameters I use grid search cvand as in one-vs-rest logistic regression I use 10Fold-cross validation and aconfusion matrix forevaluating the performance of the algorithm. In open Literature itseems that the random forests are not used for sparse data but that is something that needs further search. 
A third algorithm that I use is MLP (Multilayer Perceptron) Classifier from thelibrary scikit-learn. Firstly I use a manually grid search for finding the bestparameters. Then as above I use a 10fold cross validation for training and testingthe algorithm that performs binary classification using one-vs-rest method asdescribed above.
