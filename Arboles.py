from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()

x=iris.data
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target)
arbol=DecisionTreeClassifier()

arbol.fit(x_train,y_train)
fig=plt.figure(figsize=(25,20))
_=tree.plot_tree(arbol,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)

print(arbol.score(x_train,y_train))
print(arbol.score(x_test,y_test))

y_pred = arbol.predict(x_test)
