import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sklearn.tree as tree
import sys

my_data = pd.read_csv('static/drug200.csv')
print(my_data.head())

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

y = my_data['Drug']
y[0:5]

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size= 0.3, random_state=3)

# print('Shape of X train set {}'.format(X_trainset), '&', 'Size of Y train set {}'.format(y_trainset))

# print('Shape of X test set {}'.format(X_testset), '&', 'Size of Y test set {}'.format(y_testset))

drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)
print(predTree [0:5])
print(y_testset [0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

from sklearn.tree import export_graphviz
import subprocess

export_graphviz(drugTree, out_file='tree.dot', filled= True, feature_names= ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
subprocess.run(['C:/Program Files/Graphviz/bin/dot', '-Tpng', 'tree.dot', '-o', 'tree.png'], check=True)