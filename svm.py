import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets 
import numpy as np

#Define the col names
colnames=["sepal_length_in_cm", "sepal_width_in_cm","petal_length_in_cm","petal_width_in_cm", "class"]
iris = datasets.load_iris() 
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

plt.figure(1)
sns.heatmap(df.corr())
plt.title('Correlación en Clases de Iris')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

#Fit the model for the data

classifier.fit(X_train, y_train)

y_pred_train=classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_train)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_train,y_pred_train)

from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_train))

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

classifier = SVC(kernel = 'poly', degree=5, random_state = 0)

classifier.fit(X_train, y_train)

y_pred_train=classifier.predict(X_train)

cm = confusion_matrix(y_train, y_pred_train)
print(cm)

accuracy_score(y_train,y_pred_train)

y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

classifier = SVC(kernel = 'rbf')

classifier.fit(X_train, y_train)

y_pred_train=classifier.predict(X_train)

cm = confusion_matrix(y_train, y_pred_train)
print(cm)

accuracy_score(y_train,y_pred_train)

y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Create the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
#Fit the model for the data

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Create the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
#Fit the model for the data

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)